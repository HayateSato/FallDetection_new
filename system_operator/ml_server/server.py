"""
Fall Detection ML Server — System Operator Component

FastAPI server for ML inference with:
  - XGBoost fall detection pipeline (existing)
  - PostgreSQL inference logging (new — Phase 1)
  - Prometheus metrics export at /metrics (new — Phase 2)
  - Redis fall event publication (new — Phase 3)
  - Hot-reload model switching via /model/switch (new — Phase 5)

Run from project root:
    python system_operator/ml_server/server.py

Or with uvicorn:
    uvicorn system_operator.ml_server.server:app --host 0.0.0.0 --port 8001
"""

import asyncio
import json
import logging
import sys
import os
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.inference_engine import PipelineSelector
from app.core.model_registry import get_model_name, get_model_config, get_model_path, list_available_models
from app.data_input.data_converter import (
    convert_lsb_to_g,
    convert_acc_nparray_to_df,
    compose_detection_window,
)
from app.data_input.accelerometer_processor.acc_resampler import AccelerometerResampler

from config.settings import (
    MODEL_VERSION,
    MODEL_PATH,
    ACC_SAMPLE_RATE,
    HARDWARE_ACC_SAMPLE_RATE,
    RESAMPLING_METHOD,
    WINDOW_SIZE_SECONDS,
    PUBLIC_ENDPOINT_ENABLED,
    API_KEYS,
    RATE_LIMIT_PER_MINUTE,
    CORS_ALLOWED_ORIGINS,
    ACC_SENSOR_TYPE,
)
from config.hardware_config import ACC_SENSOR_SENSITIVITY

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log_dir = os.path.join("results", "logs")
os.makedirs(log_dir, exist_ok=True)

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f"ml_server_{MODEL_VERSION}_{timestamp_str}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics (Phase 2) — import after logging so errors are visible
# ---------------------------------------------------------------------------

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    PROMETHEUS_ENABLED = True
except ImportError:
    PROMETHEUS_ENABLED = False
    logger.warning("prometheus_fastapi_instrumentator not installed — /metrics disabled")

try:
    from system_operator.ml_server.services.metrics_collector import record_prediction
except ImportError:
    def record_prediction(*args, **kwargs):
        pass

# ---------------------------------------------------------------------------
# Model (hot-swappable — protected by a threading lock)
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_current_model_version = MODEL_VERSION
_inference_engine = PipelineSelector(MODEL_VERSION, MODEL_PATH)
_model_info = _inference_engine.get_model_info()
_model_config = get_model_config(get_model_name(MODEL_VERSION))

# Mutable runtime config — updated by POST /config, default to settings values
_window_size_seconds:   float = WINDOW_SIZE_SECONDS
_hardware_sample_rate:  int   = HARDWARE_ACC_SAMPLE_RATE   # 25, 50, or 100 Hz
_resampling_method:     str   = RESAMPLING_METHOD          # 'linear', 'decimate', 'average'
_acc_sensor_type:       str   = ACC_SENSOR_TYPE            # 'bosch' or 'non_bosch'

logger.info("=" * 60)
logger.info("Fall Detection ML Server")
logger.info(f"  Model Version:   {MODEL_VERSION.upper()}")
logger.info(f"  Model Path:      {MODEL_PATH}")
logger.info(f"  Model Name:      {_model_info['name']}")
logger.info(f"  Uses Barometer:  {_model_info['uses_barometer']}")
logger.info(f"  Features:        {_model_info['num_features']}")
logger.info(f"  Window:          {WINDOW_SIZE_SECONDS}s @ {ACC_SAMPLE_RATE}Hz")
logger.info(f"  Prometheus:      {'enabled' if PROMETHEUS_ENABLED else 'disabled'}")
logger.info("=" * 60)

# ---------------------------------------------------------------------------
# Redis publisher — optional, gracefully disabled if Redis unavailable
# ---------------------------------------------------------------------------

# Pending emergency alert tasks: inference_id → asyncio.Task
# When patient submits feedback, the task is cancelled.
# NOTE: with --workers > 1, tasks are per-process; works correctly in dev
# (single uvicorn worker) and single-container deployments.
_pending_emergency_tasks: Dict[int, "asyncio.Task[None]"] = {}


def _publish_to_redis(channel: str, payload: dict) -> None:
    """Publish a JSON payload to a Redis channel. Never raises."""
    try:
        import redis as _redis
        from config.settings import REDIS_URL
        if not REDIS_URL:
            return
        r = _redis.from_url(REDIS_URL)
        r.publish(channel, json.dumps(payload))
        logger.debug(f"Redis publish → channel={channel!r}")
    except Exception as e:
        logger.debug(f"Redis publish skipped ({e})")


async def _delayed_emergency_alert(
    inference_id: int,
    patient_id: str,
    confidence: float,
    model_version: str,
    delay_seconds: int = 12,
) -> None:
    """
    Waits delay_seconds. If no patient feedback was received, marks the row as
    no_answer and publishes an emergency alert to Redis 'fall_events'.

    This coroutine is created as an asyncio.Task and can be cancelled by the
    POST /patient/feedback/{inference_id} endpoint when the patient responds.
    """
    try:
        await asyncio.sleep(delay_seconds)
    except asyncio.CancelledError:
        logger.debug(f"Emergency timer cancelled — patient responded (inference_id={inference_id})")
        return

    # Timer fired — patient did not respond
    logger.info(f"No patient feedback in {delay_seconds}s for inference_id={inference_id} — alerting emergency")
    try:
        from shared.db.session import SessionLocal
        from shared.db.models import InferenceLog
        db = SessionLocal()
        try:
            row = db.query(InferenceLog).filter(InferenceLog.id == inference_id).first()
            if row and row.user_fall == 0:
                row.user_fall = 3   # no_answer
                row.need_help = 3   # no_answer
                db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.error(f"DB update for no-answer failed: {e}")

    _publish_to_redis("fall_events", {
        "patient_id":    patient_id,
        "fall_detected": True,
        "confidence":    confidence,
        "model_version": model_version,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "inference_id":  inference_id,
        "alert_reason":  "no_patient_response",
    })
    _pending_emergency_tasks.pop(inference_id, None)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    acc_x: List[float] = Field(..., description="Accelerometer X-axis values")
    acc_y: List[float] = Field(..., description="Accelerometer Y-axis values")
    acc_z: List[float] = Field(..., description="Accelerometer Z-axis values")
    timestamps_ms: List[float] = Field(..., description="Timestamps in milliseconds")
    pressure: Optional[List[float]] = Field(None, description="Barometer pressure values (Pa)")
    pressure_timestamps_ms: Optional[List[float]] = Field(None, description="Barometer timestamps in milliseconds")
    sample_rate: Optional[float] = Field(None, description="Hardware sampling rate in Hz")
    acc_unit: Optional[str] = Field(None, description="'lsb' or 'g'")
    participant: Optional[str] = Field(None, description="Patient/participant name for DB logging")


class PredictResponse(BaseModel):
    fall_detected: bool
    confidence: float
    threshold: float
    result: str
    model_version: str
    model_name: str
    num_features: int
    acc_features: int
    baro_features: int
    window_size: int
    features: Dict[str, float]


class ModelInfoResponse(BaseModel):
    version: str
    name: str
    description: str
    uses_barometer: bool
    acc_preprocessing: str
    baro_preprocessing: str
    num_features: int
    acc_features: int
    baro_features: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class ModelSwitchRequest(BaseModel):
    version: str = Field(..., description="Model version to load (e.g. 'v3')")


class ModelSwitchResponse(BaseModel):
    success: bool
    previous_version: str
    new_version: str
    model_name: str
    num_features: int

# ---------------------------------------------------------------------------
# API Key auth + rate limiting (unchanged from original server.py)
# ---------------------------------------------------------------------------

_server_start_time = datetime.now()
_rate_limit_storage: Dict[str, List[float]] = {}


async def verify_api_key(request: Request):
    if not PUBLIC_ENDPOINT_ENABLED:
        return
    client_ip = request.headers.get("X-Forwarded-For", request.client.host)
    now = time.time()
    window_start = now - 60
    if client_ip not in _rate_limit_storage:
        _rate_limit_storage[client_ip] = []
    _rate_limit_storage[client_ip] = [t for t in _rate_limit_storage[client_ip] if t > window_start]
    if len(_rate_limit_storage[client_ip]) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE} req/min).")
    _rate_limit_storage[client_ip].append(now)
    if not API_KEYS:
        return
    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Authentication required. Provide X-API-Key header.")
    if api_key not in API_KEYS:
        logger.warning(f"Invalid API key from {client_ip}")
        raise HTTPException(status_code=403, detail="Invalid API key.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fall Detection ML Server",
    description="Inference server with PostgreSQL logging, Prometheus metrics, and model hot-swap.",
    version="2.0.0",
)

if PUBLIC_ENDPOINT_ENABLED:
    origins = ["*"] if CORS_ALLOWED_ORIGINS == "*" else [o.strip() for o in CORS_ALLOWED_ORIGINS.split(",")]
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_methods=["GET", "POST", "OPTIONS"],
                       allow_headers=["Content-Type", "X-API-Key", "Authorization"])

# Prometheus auto-instrumentation (Phase 2)
if PROMETHEUS_ENABLED:
    Instrumentator().instrument(app).expose(app)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    uptime = (datetime.now() - _server_start_time).total_seconds()
    return HealthResponse(status="ok", model_loaded=True,
                          model_version=_current_model_version,
                          uptime_seconds=round(uptime, 1))


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    with _model_lock:
        return ModelInfoResponse(**_model_info)


@app.get("/model/list")
async def list_models():
    """Return all model versions available on disk."""
    return {"available_versions": list_available_models()}


@app.post("/model/switch", response_model=ModelSwitchResponse,
          dependencies=[Depends(verify_api_key)])
async def switch_model(req: ModelSwitchRequest):
    """
    Hot-reload the inference engine with a different model version.
    Uses a threading lock so in-flight requests complete before the switch.
    """
    global _inference_engine, _current_model_version, _model_info, _model_config

    new_version = req.version.strip()
    available = list_available_models()
    if new_version not in available:
        raise HTTPException(status_code=404,
                            detail=f"Model '{new_version}' not found. Available: {available}")

    previous = _current_model_version
    try:
        new_model_type = get_model_name(new_version)
        new_path = get_model_path(new_model_type)
        new_engine = PipelineSelector(new_version, new_path)
        new_info = new_engine.get_model_info()
        new_config = get_model_config(get_model_name(new_version))

        with _model_lock:
            _inference_engine = new_engine
            _current_model_version = new_version
            _model_info = new_info
            _model_config = new_config

        logger.info(f"Model switched: {previous} -> {new_version}")
        return ModelSwitchResponse(success=True, previous_version=previous,
                                   new_version=new_version,
                                   model_name=new_info["name"],
                                   num_features=new_info["num_features"])
    except Exception as e:
        logger.error(f"Model switch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model switch failed: {e}")


@app.get("/inferences")
async def recent_inferences(limit: int = 20):
    """
    Return the most recent N rows from inference_log.
    Used by the operator dashboard. No API key required (read-only, non-sensitive).
    """
    from shared.db.session import SessionLocal
    from shared.db.models import InferenceLog
    from sqlalchemy import desc as sa_desc

    if SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database not configured (DATABASE_URL not set)")
    db = SessionLocal()
    try:
        rows = db.query(InferenceLog).order_by(sa_desc(InferenceLog.timestamp)).limit(min(limit, 200)).all()
        return {
            "inferences": [
                {
                    "id":             r.id,
                    "timestamp":      r.timestamp.isoformat(),
                    "participant":    r.participant,
                    "fall_detected":  r.fall_detected,
                    "confidence":     round(r.confidence, 3) if r.confidence is not None else None,
                    "model_version":  r.model_version,
                    "latency_ms":     r.latency_ms,
                    "window_size":    r.window_size,
                    "inference_mode": r.inference_mode,   # 'remote', 'local', or 'replay'
                }
                for r in rows
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")
    finally:
        db.close()


@app.get("/model/comparison")
async def get_model_comparison(since_days: int = Query(30, ge=1, le=365,
                                                        description="Look back N days of replay data")):
    """
    Aggregated model comparison data from replay rows in inference_log.
    Used by the operator dashboard model comparison sub-page.
    Returns per-model stats, per-recording × model matrix, recent sessions, and raw timeseries.
    """
    from shared.db.session import SessionLocal
    from shared.db.models import InferenceLog
    from collections import defaultdict, OrderedDict
    from datetime import timedelta

    if SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database not configured (DATABASE_URL not set)")

    db = SessionLocal()
    try:
        since = datetime.now(timezone.utc) - timedelta(days=since_days)
        rows = (
            db.query(InferenceLog)
            .filter(InferenceLog.inference_mode == "replay")
            .filter(InferenceLog.timestamp >= since)
            .order_by(InferenceLog.id)
            .all()
        )

        if not rows:
            return {
                "summary": {"total_windows": 0, "total_recordings": 0, "models_tested": []},
                "by_model": [], "by_recording": [], "recent_sessions": [], "timeseries": [],
            }

        # Group by model version and by recording
        by_model_rows: dict = defaultdict(list)
        by_recording_rows: dict = defaultdict(lambda: defaultdict(list))
        for r in rows:
            by_model_rows[r.model_version].append(r)
            if r.participant:
                by_recording_rows[r.participant][r.model_version].append(r)

        def _percentile(values: list, p: float):
            if not values:
                return None
            sv = sorted(values)
            idx = min(int(len(sv) * p / 100), len(sv) - 1)
            return round(sv[idx], 4)

        def _mean(values: list):
            return round(sum(values) / len(values), 4) if values else None

        def _stddev(values: list, mean: float):
            if not values or mean is None:
                return None
            return round((sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5, 4)

        # Per-model stats
        model_stats = []
        for mv in sorted(by_model_rows):
            mrows = by_model_rows[mv]
            confs      = [r.confidence  for r in mrows if r.confidence  is not None]
            fall_confs = [r.confidence  for r in mrows if r.fall_detected and r.confidence is not None]
            nfall_confs= [r.confidence  for r in mrows if not r.fall_detected and r.confidence is not None]
            latencies  = [r.latency_ms  for r in mrows if r.latency_ms  is not None]
            falls = sum(1 for r in mrows if r.fall_detected)
            total = len(mrows)

            # Confidence buckets (10 bins 0.0–1.0)
            buckets = {f"{i/10:.1f}-{(i+1)/10:.1f}": 0 for i in range(10)}
            for c in confs:
                idx = min(int(c * 10), 9)
                key = f"{idx/10:.1f}-{(idx+1)/10:.1f}"
                buckets[key] += 1

            mean_conf = _mean(confs)
            # Uncertainty: windows with confidence in [0.4, 0.6] (near decision boundary)
            uncertain = sum(1 for c in confs if 0.4 <= c <= 0.6)

            model_stats.append({
                "model_version":    mv,
                "total_windows":    total,
                "falls_detected":   falls,
                "fall_rate_pct":    round(100 * falls / total, 2) if total > 0 else 0,
                "uncertainty_pct":  round(100 * uncertain / total, 2) if total > 0 else 0,
                "percentiles": {
                    "p10":   _percentile(confs, 10),
                    "p25":   _percentile(confs, 25),
                    "p50":   _percentile(confs, 50),
                    "p75":   _percentile(confs, 75),
                    "p90":   _percentile(confs, 90),
                    "p95":   _percentile(confs, 95),
                    "mean":  mean_conf,
                    "stddev":_stddev(confs, mean_conf),
                },
                "latency": {
                    "avg_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
                    "p95_ms": _percentile(sorted(latencies), 95)        if latencies else None,
                },
                "confidence_buckets": [
                    {"range": k, "count": v} for k, v in sorted(buckets.items())
                ],
                "box_data": {
                    "all":      confs[:2000],
                    "falls":    fall_confs[:1000],
                    "no_falls": nfall_confs[:1000],
                },
                "recordings": sorted(set(r.participant for r in mrows if r.participant)),
            })

        # Per-recording × model matrix
        by_recording = []
        for rec in sorted(by_recording_rows):
            models_map = {}
            for mv, rec_rows in sorted(by_recording_rows[rec].items()):
                rec_confs = [r.confidence for r in rec_rows if r.confidence is not None]
                rec_falls = sum(1 for r in rec_rows if r.fall_detected)
                models_map[mv] = {
                    "total_windows":  len(rec_rows),
                    "falls_detected": rec_falls,
                    "fall_rate_pct":  round(100 * rec_falls / len(rec_rows), 2) if rec_rows else 0,
                    "avg_confidence": _mean(rec_confs),
                }
            by_recording.append({"recording": rec, "models": models_map})

        # Recent sessions (deduplicated by participant + model_version + date)
        session_keys: OrderedDict = OrderedDict()
        for r in reversed(rows):
            date = r.timestamp.date() if r.timestamp else None
            key = (r.participant, r.model_version, date)
            if key not in session_keys:
                session_keys[key] = r
            if len(session_keys) >= 30:
                break

        recent_sessions = []
        for (rec, mv, date), anchor in session_keys.items():
            s_rows = [
                row for row in rows
                if row.participant == rec and row.model_version == mv
                and (row.timestamp.date() == date if row.timestamp and date else True)
            ]
            s_falls = sum(1 for r in s_rows if r.fall_detected)
            recent_sessions.append({
                "timestamp":         anchor.timestamp.isoformat() if anchor.timestamp else None,
                "model_version":     mv,
                "recording":         rec,
                "total_windows":     len(s_rows),
                "falls_detected":    s_falls,
                "fall_rate_pct":     round(100 * s_falls / len(s_rows), 2) if s_rows else 0,
                "step_seconds":      anchor.step_seconds,
                "resampling_method": anchor.resampling_method,
            })

        # Timeseries for scatter plot (capped at 5000 rows)
        ts_rows = rows[-5000:] if len(rows) > 5000 else rows
        timeseries = [
            {
                "id":            r.id,
                "timestamp":     r.timestamp.isoformat() if r.timestamp else None,
                "model_version": r.model_version,
                "recording":     r.participant,
                "confidence":    round(r.confidence, 4) if r.confidence is not None else None,
                "fall_detected": r.fall_detected,
                "latency_ms":    r.latency_ms,
            }
            for r in ts_rows
        ]

        return {
            "summary": {
                "total_windows":     len(rows),
                "total_recordings":  len(by_recording_rows),
                "models_tested":     sorted(by_model_rows.keys()),
            },
            "by_model":        model_stats,
            "by_recording":    by_recording,
            "recent_sessions": recent_sessions,
            "timeseries":      timeseries,
        }

    except Exception as e:
        logger.error(f"Model comparison query failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Database error: {e}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# /patient — patient dashboard SSE + feedback submission
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    user_fall: int = Field(..., description="0=pending, 1=yes(fell), 2=no(not fell), 3=no_answer")
    need_help: int = Field(0,   description="0=pending, 1=yes, 2=no, 3=no_answer")


@app.get("/patient/stream")
async def patient_stream(participant: Optional[str] = Query(None,
                          description="Patient name — filters events to this participant only")):
    """
    SSE stream for the patient dashboard.
    Delivers fall alert events published to Redis 'patient_alerts'.
    Only fires for real-time detections (not replay).
    """
    from fastapi.responses import StreamingResponse as _SR

    async def _generator():
        from shared.redis_client import subscribe_patient_alerts
        try:
            async for event in subscribe_patient_alerts(participant):
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            pass

    return _SR(
        _generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/patient/feedback/{inference_id}")
async def patient_feedback(inference_id: int, req: FeedbackRequest):
    """
    Submit patient feedback for a detected fall.

    Called by the patient dashboard after the 10-second popup.
    Cancels the pending emergency alert timer and decides whether to publish
    an emergency alert based on the patient's response.

    Decision logic:
      - user_fall=1, need_help=1          → alert emergency (fell + needs help)
      - user_fall=1, need_help=3          → alert emergency (fell, no answer on help = treat as needs help)
      - user_fall=1, need_help=2          → no alert (fell but explicitly said no help needed)
      - user_fall=2                       → no alert (not a fall)
      - user_fall=3 (submitted by client) → alert emergency (no response at all)
    """
    # Cancel the pending emergency timer
    task = _pending_emergency_tasks.pop(inference_id, None)
    if task and not task.done():
        task.cancel()

    # Update the database
    from shared.db.session import SessionLocal
    from shared.db.models import InferenceLog

    if SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    db = SessionLocal()
    try:
        row = db.query(InferenceLog).filter(InferenceLog.id == inference_id).first()
        if not row:
            raise HTTPException(status_code=404, detail=f"Inference {inference_id} not found")

        row.user_fall = req.user_fall
        row.need_help = req.need_help
        db.commit()

        # Capture values before closing session
        patient_id    = row.participant or "unknown"
        confidence    = row.confidence
        model_version = row.model_version
        ts            = row.timestamp.isoformat() if row.timestamp else datetime.now(timezone.utc).isoformat()

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=503, detail=f"Database error: {e}")
    finally:
        db.close()

    # Decide whether to publish the emergency alert
    should_alert = (
        req.user_fall == 1 and req.need_help in (1, 3)  # fell + needs help / fell + no answer on help
        or req.user_fall == 3                            # patient gave no answer at all
    )

    if should_alert:
        alert_reason = "patient_confirmed_help" if req.need_help == 1 else "no_patient_response"
        _publish_to_redis("fall_events", {
            "patient_id":    patient_id,
            "fall_detected": True,
            "confidence":    confidence,
            "model_version": model_version,
            "timestamp":     ts,
            "inference_id":  inference_id,
            "alert_reason":  alert_reason,
        })
        logger.info(f"Emergency alert published: inference_id={inference_id}, reason={alert_reason}")

    return {
        "inference_id":     inference_id,
        "user_fall":        req.user_fall,
        "need_help":        req.need_help,
        "emergency_alerted": should_alert,
    }


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)])
async def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    """
    Run fall detection inference on a window of sensor data.

    Pipeline: resample → LSB-to-g → DataFrame → window → features → XGBoost
    After responding: writes result to Postgres + publishes to Redis (background tasks).
    """
    n = len(req.acc_x)
    if len(req.acc_y) != n or len(req.acc_z) != n or len(req.timestamps_ms) != n:
        raise HTTPException(status_code=422,
                            detail="acc_x, acc_y, acc_z, timestamps_ms must all have the same length.")
    if n == 0:
        raise HTTPException(status_code=422, detail="Sensor data arrays cannot be empty.")

    logger.info(f"Predict request: {n} samples, participant={req.participant}")
    t_start = time.monotonic()

    try:
        with _model_lock:
            engine = _inference_engine
            mv = _current_model_version
            mc = _model_config
            mi = _model_info

        # 1. numpy arrays (shape: 3 x N)
        acc_data = np.array([req.acc_x, req.acc_y, req.acc_z])
        acc_time = np.array(req.timestamps_ms)

        # 2. Resampling
        hw_rate = req.sample_rate or _hardware_sample_rate
        if hw_rate != ACC_SAMPLE_RATE:
            resampler = AccelerometerResampler(source_rate=hw_rate, target_rate=ACC_SAMPLE_RATE,
                                               method=_resampling_method)
            acc_data, acc_time = resampler.process(acc_data, acc_time)
            logger.info(f"  Resampled {hw_rate}Hz -> {ACC_SAMPLE_RATE}Hz ({acc_data.shape[1]} samples)")

        # 3. LSB-to-g conversion
        data_is_lsb = (req.acc_unit or "lsb") == "lsb"
        if not mc.acc_in_lsb and data_is_lsb:
            acc_data = convert_lsb_to_g(acc_data)
        elif mc.acc_in_lsb and not data_is_lsb:
            logger.warning("Model expects LSB but received g values. Results may be inaccurate.")

        # 4. DataFrame
        df = convert_acc_nparray_to_df(acc_data, acc_time)

        # 5. Window extraction
        required_samples = int(_window_size_seconds * ACC_SAMPLE_RATE)
        pressure = np.array(req.pressure) if req.pressure else None
        pressure_time = np.array(req.pressure_timestamps_ms) if req.pressure_timestamps_ms else None
        window_df, window_pressure, window_pressure_time = compose_detection_window(
            df, required_samples, pressure, pressure_time)
        logger.info(f"  Window: {len(window_df)} ACC samples")

        # 6. Inference
        result = engine.predict(window_df, pressure=window_pressure,
                                pressure_timestamps=window_pressure_time)

        is_fall = result["is_fall"]
        confidence = result["confidence"]
        features_dict = result["features"]
        threshold = mc.threshold
        latency_ms = int((time.monotonic() - t_start) * 1000)

        logger.info(f"  Result: fall={is_fall}, confidence={confidence:.3f}, latency={latency_ms}ms")

        # Result message
        if is_fall:
            result_message = ("High confidence fall detection" if confidence > 0.75
                              else "Moderate confidence fall detection" if confidence > 0.60
                              else "Low confidence fall detection")
        else:
            result_message = ("Close to threshold - consider manual review" if confidence > 0.40
                              else "Borderline case" if confidence > 0.25
                              else "Clear negative")

        # Phase 2: Prometheus metrics (fire-and-forget)
        record_prediction(mv, is_fall, float(confidence), latency_ms / 1000)

        # Phase 1: PostgreSQL write (background task — response already sent by then)
        background_tasks.add_task(
            _bg_write_and_publish,
            mv=mv, is_fall=is_fall, confidence=float(confidence),
            window_size=len(window_df), latency_ms=latency_ms,
            participant=req.participant, features=features_dict,
        )

        return PredictResponse(
            fall_detected=is_fall,
            confidence=float(confidence),
            threshold=float(threshold),
            result=result_message,
            model_version=mv,
            model_name=mi["name"],
            num_features=mi["num_features"],
            acc_features=mi["acc_features"],
            baro_features=mi["baro_features"],
            window_size=len(window_df),
            features=features_dict,
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


async def _bg_write_and_publish(mv: str, is_fall: bool, confidence: float,
                                window_size: int, latency_ms: int,
                                participant: Optional[str], features: dict) -> None:
    """
    Async background task (runs in the event loop, not a thread pool).

    Steps:
      1. Write inference result to PostgreSQL.
      2. If a fall was detected (real-time mode only):
           a. Publish to Redis 'patient_alerts' → triggers the patient dashboard popup.
           b. Schedule a 12-second delayed emergency alert.
              The timer is cancelled if the patient submits feedback in time.
              If it fires, user_fall=3/need_help=3 are written and 'fall_events' is published.
    """
    from system_operator.ml_server.services.db_writer import write_inference_log

    inference_id = write_inference_log(
        model_version=mv,
        fall_detected=is_fall,
        confidence=confidence,
        window_size=window_size,
        inference_mode="remote",
        latency_ms=latency_ms,
        participant=participant,
        features=features,
    )

    if is_fall and inference_id:
        patient_id = participant or "unknown"
        # Notify the patient dashboard
        _publish_to_redis("patient_alerts", {
            "patient_id":    patient_id,
            "fall_detected": True,
            "confidence":    confidence,
            "model_version": mv,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "inference_id":  inference_id,
        })
        # Schedule the delayed emergency alert (fires in 12s if no patient response)
        task = asyncio.create_task(
            _delayed_emergency_alert(inference_id, patient_id, confidence, mv)
        )
        _pending_emergency_tasks[inference_id] = task
        logger.info(f"Emergency timer started for inference_id={inference_id} (12s)")

# ---------------------------------------------------------------------------
# /config — update detection window size at runtime
# ---------------------------------------------------------------------------

class ConfigRequest(BaseModel):
    window_seconds:      Optional[float] = Field(None, ge=1.0,   le=60.0,
                                                 description="Detection window size in seconds")
    hardware_sample_rate: Optional[int]  = Field(None,
                                                 description="Hardware ACC sample rate in Hz (25, 50, or 100)")
    resampling_method:   Optional[str]   = Field(None,
                                                 description="Resampling algorithm: 'linear', 'decimate', or 'average'")
    acc_sensor_type:     Optional[str]   = Field(None,
                                                 description="Accelerometer sensor type: 'bosch' or 'non_bosch'")


@app.post("/config")
async def set_config(req: ConfigRequest):
    """Update runtime processing config. Only supplied fields are changed."""
    global _window_size_seconds, _hardware_sample_rate, _resampling_method, _acc_sensor_type

    if req.window_seconds is not None:
        _window_size_seconds = req.window_seconds

    if req.hardware_sample_rate is not None:
        if req.hardware_sample_rate not in (25, 50, 100):
            raise HTTPException(status_code=422, detail="hardware_sample_rate must be 25, 50, or 100")
        _hardware_sample_rate = req.hardware_sample_rate

    if req.resampling_method is not None:
        if req.resampling_method not in ("linear", "decimate", "average"):
            raise HTTPException(status_code=422,
                                detail="resampling_method must be 'linear', 'decimate', or 'average'")
        _resampling_method = req.resampling_method

    if req.acc_sensor_type is not None:
        if req.acc_sensor_type not in ("bosch", "non_bosch"):
            raise HTTPException(status_code=422, detail="acc_sensor_type must be 'bosch' or 'non_bosch'")
        _acc_sensor_type = req.acc_sensor_type

    logger.info(
        f"Config updated: window={_window_size_seconds}s, hw_rate={_hardware_sample_rate}Hz, "
        f"resample={_resampling_method}, sensor={_acc_sensor_type}"
    )
    return _build_config_response()


@app.get("/config")
async def get_config():
    """Return the current runtime processing configuration."""
    return _build_config_response()


def _build_config_response() -> dict:
    return {
        "window_seconds":       _window_size_seconds,
        "window_samples":       int(_window_size_seconds * ACC_SAMPLE_RATE),
        "model_sample_rate_hz": ACC_SAMPLE_RATE,
        "hardware_sample_rate": _hardware_sample_rate,
        "resampling_method":    _resampling_method,
        "acc_sensor_type":      _acc_sensor_type,
    }


# ---------------------------------------------------------------------------
# /datalake — MinIO file management + CSV offline inference replay
# ---------------------------------------------------------------------------

@app.get("/datalake/files")
async def datalake_list_files():
    """List CSV files available in the MinIO datalake bucket."""
    try:
        from datalake.minio_client import list_csv_files
        files = list_csv_files()
        return {"files": files, "bucket": os.getenv("MINIO_BUCKET", "sensor-recordings")}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MinIO unavailable: {e}")


@app.post("/datalake/upload")
async def datalake_upload(file: UploadFile = File(...)):
    """Upload a SmarKo CSV recording to the MinIO datalake."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")
    try:
        from datalake.minio_client import upload_file
        upload_file(file.filename, file.file)
        return {"success": True, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.post("/datalake/replay")
async def datalake_replay(
    filename: str = Query(..., description="CSV filename in MinIO bucket"),
    window_seconds: Optional[float] = Query(None, ge=1.0, le=60.0,
                                            description="Window size in seconds (default: server config)"),
    step_seconds: float = Query(3.0, ge=0.5, le=30.0,
                                description="Step between windows in seconds"),
    participant: str = Query("replay", description="Participant label for DB logging"),
    sample_rate: float = Query(25.0, ge=1.0, le=200.0,
                               description="Hardware ACC sample rate of the recording in Hz"),
):
    """
    Run fall detection on every window of a CSV file stored in MinIO.

    Downloads the CSV, splits it into sliding windows, runs each through
    the currently loaded model, and returns all predictions. This is the
    core of the offline model comparison workflow — same CSV, different
    models, compare the results.
    """
    win_s = window_seconds if window_seconds is not None else _window_size_seconds

    try:
        from datalake.minio_client import download_file_bytes
        from datalake.csv_converter import load_csv, extract_acc, extract_pressure, split_into_windows

        logger.info(f"Replay start: file={filename!r}, window={win_s}s, step={step_seconds}s")
        csv_bytes = download_file_bytes(filename)
        df        = load_csv(csv_bytes)
        acc_data, acc_time  = extract_acc(df)
        pressure, pres_time = extract_pressure(df)
        windows = split_into_windows(
            acc_data, acc_time, pressure, pres_time,
            window_seconds=win_s,
            step_seconds=step_seconds,
            sample_rate=sample_rate,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MinIO unavailable or CSV invalid: {e}")

    with _model_lock:
        engine = _inference_engine
        mv     = _current_model_version
        mc     = _model_config

    predictions = []
    falls_detected = 0

    for i, w in enumerate(windows):
        try:
            t_start = time.monotonic()

            acc = np.array([w["acc_x"], w["acc_y"], w["acc_z"]])
            ts  = np.array(w["timestamps_ms"])

            hw_rate = w.get("sample_rate", sample_rate)
            if hw_rate != ACC_SAMPLE_RATE:
                resampler = AccelerometerResampler(
                    source_rate=hw_rate, target_rate=ACC_SAMPLE_RATE,
                    method=_resampling_method,
                )
                acc, ts = resampler.process(acc, ts)

            if not mc.acc_in_lsb:
                acc = convert_lsb_to_g(acc)

            df_win = convert_acc_nparray_to_df(acc, ts)
            required = int(win_s * ACC_SAMPLE_RATE)
            pres_arr = np.array(w["pressure"])               if w["pressure"]               else None
            pres_ts  = np.array(w["pressure_timestamps_ms"]) if w["pressure_timestamps_ms"] else None
            df_win, w_pressure, w_pres_ts = compose_detection_window(
                df_win, required, pres_arr, pres_ts)

            result     = engine.predict(df_win, pressure=w_pressure, pressure_timestamps=w_pres_ts)
            latency_ms = int((time.monotonic() - t_start) * 1000)
            is_fall    = result["is_fall"]
            confidence = float(result["confidence"])
            if is_fall:
                falls_detected += 1

            predictions.append({
                "window_index":    i,
                "window_start_ms": w["window_start_ms"],
                "window_end_ms":   w["window_end_ms"],
                "fall_detected":   is_fall,
                "confidence":      round(confidence, 4),
                "latency_ms":      latency_ms,
            })

        except Exception as e:
            logger.warning(f"Replay window {i} failed: {e}")
            predictions.append({
                "window_index":    i,
                "window_start_ms": w.get("window_start_ms"),
                "window_end_ms":   w.get("window_end_ms"),
                "fall_detected":   None,
                "confidence":      None,
                "latency_ms":      None,
                "error":           str(e),
            })

    logger.info(f"Replay complete: {len(predictions)} windows, {falls_detected} falls")

    # Write all predictions to Postgres in one transaction (inference_mode='replay')
    from system_operator.ml_server.services.db_writer import write_inference_batch
    from datetime import datetime, timezone as _tz
    db_rows = [
        {
            "timestamp":         datetime.fromtimestamp(p["window_start_ms"] / 1000, tz=_tz.utc)
                                 if p.get("window_start_ms") else datetime.now(_tz.utc),
            "model_version":     mv,
            "fall_detected":     p["fall_detected"],
            "confidence":        p["confidence"],
            "window_size":       int(win_s * ACC_SAMPLE_RATE),
            "inference_mode":    "replay",
            "latency_ms":        p.get("latency_ms"),
            "participant":       filename,
            "step_seconds":      step_seconds,
            "resampling_method": _resampling_method,
            "acc_sensor_type":   _acc_sensor_type,
        }
        for p in predictions
    ]
    written = write_inference_batch(db_rows)
    logger.info(f"Saved {written} replay rows to inference_log")

    return {
        "filename":       filename,
        "participant":    participant,
        "model_version":  mv,
        "total_windows":  len(predictions),
        "falls_detected": falls_detected,
        "window_seconds": win_s,
        "step_seconds":   step_seconds,
        "db_rows_saved":  written,
        "predictions":    predictions,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("SERVER_PORT", "8001"))
    logger.info(f"Starting ML server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
