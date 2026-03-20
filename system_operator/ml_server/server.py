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

import logging
import sys
import os
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.inference_engine import PipelineSelector
from app.core.model_registry import get_model_name, get_model_config, list_available_models
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
# Redis publisher (Phase 3) — optional, gracefully disabled if Redis unavailable
# ---------------------------------------------------------------------------

def _publish_fall_event(patient_id: str, fall_detected: bool, confidence: float,
                        model_version: str, inference_id: Optional[int]) -> None:
    """Publish fall event to Redis channel 'fall_events'. Never raises."""
    try:
        import redis
        from config.settings import REDIS_URL
        if not REDIS_URL:
            return
        r = redis.from_url(REDIS_URL)
        import json
        payload = json.dumps({
            "patient_id": patient_id,
            "fall_detected": fall_detected,
            "confidence": confidence,
            "model_version": model_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inference_id": inference_id,
        })
        r.publish("fall_events", payload)
        logger.debug(f"Published fall_event to Redis: fall={fall_detected}, patient={patient_id}")
    except Exception as e:
        logger.debug(f"Redis publish skipped ({e})")

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
        hw_rate = req.sample_rate or HARDWARE_ACC_SAMPLE_RATE
        if hw_rate != ACC_SAMPLE_RATE:
            resampler = AccelerometerResampler(source_rate=hw_rate, target_rate=ACC_SAMPLE_RATE,
                                               method=RESAMPLING_METHOD)
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
        required_samples = int(WINDOW_SIZE_SECONDS * ACC_SAMPLE_RATE)
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


def _bg_write_and_publish(mv: str, is_fall: bool, confidence: float,
                           window_size: int, latency_ms: int,
                           participant: Optional[str], features: dict) -> None:
    """Background task: write to Postgres, then publish to Redis if fall detected."""
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
    # Phase 3: publish to Redis regardless of fall/not-fall so care-giver sees all events
    _publish_fall_event(
        patient_id=participant or "unknown",
        fall_detected=is_fall,
        confidence=confidence,
        model_version=mv,
        inference_id=inference_id,
    )

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("SERVER_PORT", "8001"))
    logger.info(f"Starting ML server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
