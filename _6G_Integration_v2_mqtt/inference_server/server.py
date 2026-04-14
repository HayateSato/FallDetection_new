"""
Fall Detection Inference Server — 6G / Charite Integration Build
================================================================
Minimal FastAPI server for delivering fall detection results in FHIR R4 format
to a medical partner (Charite / FOCUS stack).

What is fixed (not configurable at runtime):
  - Model version        : set in .env → MODEL_VERSION
  - Sensor type          : set in .env → ACC_SENSOR_TYPE  (bosch | non_bosch)
  - Hardware sample rate : set in .env → HARDWARE_ACC_SAMPLE_RATE  (25 Hz)
  - ACC unit             : always LSB on input; server converts to g internally
  - Detection window     : 9 s × 50 Hz = 450 samples  (fixed, cannot be changed)
  - Target sample rate   : 50 Hz  (fixed — what the XGBoost models were trained on)

What is NOT included (compared to full system):
  - No model switching UI / /model/switch endpoint
  - No MQTT patient feedback / emergency service
  - No Prometheus metrics / Grafana
  - No MinIO datalake / CSV replay
  - No operator or caregiver dashboards
  - No patient popup / 12-second emergency timer

Endpoints
---------
  POST /predict   — run inference, return JSON + embedded FHIR Observation
  GET  /health    — liveness check
  GET  /model/info — loaded model metadata (includes uses_barometer flag)
  GET  /docs      — OpenAPI / Swagger UI

Run (from _6G_Integration/ as working directory):
  uvicorn inference_server.server:app --host 0.0.0.0 --port 8001
"""

import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    _PROMETHEUS_ENABLED = True
except ImportError:
    _PROMETHEUS_ENABLED = False

try:
    from inference_server.services.metrics_collector import record_prediction as _record_prediction
except ImportError:
    def _record_prediction(*args, **kwargs):
        pass

# ---------------------------------------------------------------------------
# Shared pipeline imports  (app/ and config/ live in _6G_Integration/)
# ---------------------------------------------------------------------------
from app.core.inference_engine import PipelineSelector
from app.core.model_registry import get_model_config, get_model_name, get_model_path, list_available_models
from app.data_input.accelerometer_processor.acc_resampler import AccelerometerResampler
from app.data_input.data_converter import (
    compose_detection_window,
    convert_acc_nparray_to_df,
    convert_lsb_to_g,
)
from config.settings import (
    ACC_SAMPLE_RATE,
    ACC_SENSOR_TYPE,
    API_KEYS,
    CORS_ALLOWED_ORIGINS,
    HARDWARE_ACC_SAMPLE_RATE,
    MODEL_PATH,
    MODEL_VERSION,
    PUBLIC_ENDPOINT_ENABLED,
    RATE_LIMIT_PER_MINUTE,
    RESAMPLING_METHOD,
    WINDOW_SIZE_SECONDS,
)
from fhir_converter import build_fhir_observation

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_dir = os.path.join("results", "logs")
os.makedirs(log_dir, exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(log_dir, f"6g_server_{MODEL_VERSION}_{_ts}.log"),
            mode="w", encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FHIR server config (optional — push results to partner's FHIR server)
# ---------------------------------------------------------------------------
FHIR_SERVER_URL = os.getenv("FHIR_SERVER_URL", "").rstrip("/")
FHIR_BASE_URL   = os.getenv("FHIR_BASE_URL", "")     # base for resource references
FHIR_AUTH_TOKEN = os.getenv("FHIR_AUTH_TOKEN", "")   # Bearer token if required
FHIR_PUSH_ON_FALL_ONLY = os.getenv("FHIR_PUSH_ON_FALL_ONLY", "true").lower() != "false"

# ---------------------------------------------------------------------------
# MQTT config (optional — publish fall events for live dashboards / mobile app)
# ---------------------------------------------------------------------------
MQTT_BROKER_HOST   = os.getenv("MQTT_BROKER_HOST", "").strip()
MQTT_BROKER_PORT   = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_FALL_TOPIC    = os.getenv("MQTT_FALL_TOPIC", "fall/events")
MQTT_USERNAME      = os.getenv("MQTT_USERNAME", "").strip()
MQTT_PASSWORD      = os.getenv("MQTT_PASSWORD", "").strip()
_mqtt_client = None  # populated on startup if MQTT_BROKER_HOST set

# ---------------------------------------------------------------------------
# Model — hot-swappable via POST /model/switch
# All reads/writes to these globals must hold _model_lock.
# ---------------------------------------------------------------------------
_model_lock            = threading.Lock()
_current_model_version = MODEL_VERSION
_model_type            = get_model_name(MODEL_VERSION)
_model_path            = MODEL_PATH
_engine                = PipelineSelector(MODEL_VERSION, _model_path)
_model_config          = get_model_config(_model_type)
_model_info            = _engine.get_model_info()

logger.info("=" * 60)
logger.info("Fall Detection Server  [6G / Charite Integration]")
logger.info(f"  Model:            {MODEL_VERSION.upper()}  ({_model_info['name']})")
logger.info(f"  Uses barometer:   {_model_info['uses_barometer']}")
logger.info(f"  Features:         {_model_info['num_features']}")
logger.info(f"  Sensor type:      {ACC_SENSOR_TYPE}  ({HARDWARE_ACC_SAMPLE_RATE} Hz → 50 Hz)")
logger.info(f"  Window:           {WINDOW_SIZE_SECONDS}s × {ACC_SAMPLE_RATE}Hz = "
            f"{int(WINDOW_SIZE_SECONDS * ACC_SAMPLE_RATE)} samples")
logger.info(f"  FHIR server:      {FHIR_SERVER_URL or '(not configured — results not pushed)'}")
logger.info(f"  MQTT broker:      {f'{MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}  topic={MQTT_FALL_TOPIC}' if MQTT_BROKER_HOST else '(not configured — no MQTT publishing)'}")
logger.info(f"  Prometheus:       {'enabled  (/metrics)' if _PROMETHEUS_ENABLED else 'disabled (pip install prometheus-fastapi-instrumentator)'}")
logger.info("=" * 60)

# ---------------------------------------------------------------------------
# API key auth + rate limiting
# ---------------------------------------------------------------------------
_server_start_time   = datetime.now()
_rate_limit_storage: Dict[str, List[float]] = {}


async def verify_api_key(request: Request):
    if not PUBLIC_ENDPOINT_ENABLED:
        return
    client_ip = request.headers.get("X-Forwarded-For", request.client.host)
    now = time.time()
    _rate_limit_storage.setdefault(client_ip, [])
    _rate_limit_storage[client_ip] = [
        t for t in _rate_limit_storage[client_ip] if t > now - 60
    ]
    if len(_rate_limit_storage[client_ip]) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429,
                            detail=f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE} req/min).")
    _rate_limit_storage[client_ip].append(now)
    if not API_KEYS:
        return
    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if not api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required.")
    if api_key not in API_KEYS:
        logger.warning(f"Invalid API key from {client_ip}")
        raise HTTPException(status_code=403, detail="Invalid API key.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fall Detection — FHIR Integration Server",
    description=(
        "POST /predict with accelerometer (+ optional barometer) data.\n"
        "Returns the inference result AND a FHIR R4 Observation resource.\n\n"
        f"Fixed model: **{MODEL_VERSION.upper()}**  "
        f"({'ACC + barometer' if _model_info['uses_barometer'] else 'ACC only'})\n\n"
        f"Sensor: **{ACC_SENSOR_TYPE}** at **{HARDWARE_ACC_SAMPLE_RATE} Hz** "
        f"→ resampled to 50 Hz internally.\n\n"
        "Input ACC values must be raw **LSB integers** as recorded by the SmarKo app."
    ),
    version="1.0.0",
)

origins = ["*"] if CORS_ALLOWED_ORIGINS == "*" else [o.strip() for o in CORS_ALLOWED_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

if _PROMETHEUS_ENABLED:
    Instrumentator().instrument(app).expose(app)  # adds GET /metrics


@app.on_event("startup")
async def _connect_mqtt():
    """Connect to MQTT broker at startup if MQTT_BROKER_HOST is configured."""
    global _mqtt_client
    if not MQTT_BROKER_HOST:
        return
    if mqtt is None:
        logger.warning("MQTT_BROKER_HOST is set but 'paho-mqtt' is not installed — "
                       "skipping MQTT publishing. `pip install paho-mqtt`.")
        return
    try:
        client = mqtt.Client(client_id="fall-detection-server")
        if MQTT_USERNAME:
            client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD or None)
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
        client.loop_start()  # background thread handles network I/O
        _mqtt_client = client
        logger.info(f"Connected to MQTT broker at {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}"
                    f"  topic={MQTT_FALL_TOPIC}")
    except Exception as exc:
        logger.warning(f"Could not connect to MQTT broker "
                       f"({MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}): {exc}")
        _mqtt_client = None


@app.on_event("shutdown")
async def _disconnect_mqtt():
    if _mqtt_client is not None:
        try:
            _mqtt_client.loop_stop()
            _mqtt_client.disconnect()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    # ── Identifiers ────────────────────────────────────────────────────────
    patient_id: str = Field(
        ...,
        description=(
            "Patient identifier — used as FHIR Patient reference (Patient/<patient_id>). "
            "Must match the patient registered in your FHIR server."
        ),
        example="charite-patient-007",
    )
    device_id: Optional[str] = Field(
        None,
        description="SmarKo device/wearable identifier. Added to FHIR Observation as Device reference.",
        example="smarko-wearable-42",
    )

    # ── Sensor data ────────────────────────────────────────────────────────
    acc_x: List[float] = Field(
        ...,
        description=(
            f"Raw accelerometer X-axis values in **LSB integers** "
            f"({'bosch_acc_x' if ACC_SENSOR_TYPE == 'bosch' else 'acc_x'} field from InfluxDB). "
            f"Hardware rate: {HARDWARE_ACC_SAMPLE_RATE} Hz. "
            "Server resamples to 50 Hz internally."
        ),
    )
    acc_y: List[float] = Field(..., description="Raw accelerometer Y-axis values (LSB).")
    acc_z: List[float] = Field(..., description="Raw accelerometer Z-axis values (LSB).")
    timestamps_ms: List[float] = Field(
        ...,
        description="Timestamps for each ACC sample in milliseconds (Unix epoch ms).",
    )

    # ── Barometer (required for v3 model, ignored for v0) ──────────────────
    pressure: Optional[List[float]] = Field(
        None,
        description=(
            f"Barometer pressure values in Pa (bmp_pressure field from InfluxDB). "
            f"{'REQUIRED for the configured model (v3 uses barometer).' if _model_info['uses_barometer'] else 'Not used by the configured model (v0 — ACC only).'}"
        ),
    )
    pressure_timestamps_ms: Optional[List[float]] = Field(
        None,
        description="Timestamps for barometer samples in milliseconds.",
    )


class InferenceResult(BaseModel):
    """Standard inference result — same fields as EcoSystem integration."""
    fall_detected: bool
    confidence:    float
    threshold:     float
    result:        str       # human-readable label
    model_version: str
    window_size:   int       # number of ACC samples used


class PredictResponse(BaseModel):
    """
    Combined response: inference result + FHIR R4 Observation.
    The `fhir_observation` field is a complete, valid FHIR R4 Observation
    resource that can be POSTed directly to a FHIR server.
    """
    patient_id:       str
    device_id:        Optional[str]
    timestamp:        str
    inference:        InferenceResult
    fhir_observation: dict    # FHIR R4 Observation resource
    fhir_pushed:      bool    # True if server successfully POSTed to FHIR_SERVER_URL


class ModelSwitchRequest(BaseModel):
    version: str = Field(..., description="Model version to load (e.g. 'v3')")


class ModelSwitchResponse(BaseModel):
    success:          bool
    previous_version: str
    new_version:      str
    model_name:       str
    num_features:     int
    uses_barometer:   bool

# ---------------------------------------------------------------------------
# FHIR push helper
# ---------------------------------------------------------------------------

async def _publish_fall_event(payload: dict) -> bool:
    """
    Publish a fall event to the MQTT broker so dashboards and the mobile app
    can react in real time.  Never raises — MQTT failures must not break inference.
    """
    if _mqtt_client is None:
        return False
    try:
        import json as _json
        topic   = f"{MQTT_FALL_TOPIC}/{payload.get('patient_id', 'unknown')}"
        message = _json.dumps(payload, default=str)
        result  = _mqtt_client.publish(topic, message)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.warning(f"MQTT publish returned error code {result.rc}  topic={topic}")
            return False
        logger.info(f"MQTT publish OK  topic={topic}  "
                    f"patient={payload.get('patient_id')}")
        return True
    except Exception as exc:
        logger.warning(f"MQTT publish error: {exc}")
        return False


async def _push_to_fhir_server(observation: dict) -> bool:
    """
    POST the FHIR Observation to the configured FHIR server.
    Returns True on success, False on failure (never raises — inference
    result is returned to caller regardless of FHIR push outcome).
    """
    if not FHIR_SERVER_URL:
        return False
    headers = {"Content-Type": "application/fhir+json"}
    if FHIR_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {FHIR_AUTH_TOKEN}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{FHIR_SERVER_URL}/Observation",
                json=observation,
                headers=headers,
            )
        if resp.is_success:
            logger.info(f"FHIR push OK  status={resp.status_code}  "
                        f"id={observation.get('id')}")
            return True
        else:
            logger.warning(f"FHIR push failed  status={resp.status_code}  "
                           f"body={resp.text[:200]}")
            return False
    except Exception as exc:
        logger.warning(f"FHIR push error: {exc}")
        return False

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    uptime = (datetime.now() - _server_start_time).total_seconds()
    return {
        "status":        "ok",
        "model_version": MODEL_VERSION,
        "uses_barometer": _model_info["uses_barometer"],
        "sensor_type":   ACC_SENSOR_TYPE,
        "sample_rate_hz": HARDWARE_ACC_SAMPLE_RATE,
        "uptime_seconds": round(uptime, 1),
    }


@app.get("/model/info")
async def model_info():
    """
    Metadata about the loaded model.
    The `uses_barometer` field tells the trigger client whether
    to fetch `bmp_pressure` from InfluxDB alongside ACC data.
    """
    return _model_info


@app.post("/predict", response_model=PredictResponse,
          dependencies=[Depends(verify_api_key)])
async def predict(req: PredictRequest):
    """
    Run fall detection on one window of sensor data.

    **Input units:** raw LSB integers as written by the SmarKo app to InfluxDB.
    The server converts LSB → g and resamples to 50 Hz internally.

    **Minimum data required:**
    At least `window_seconds × hardware_rate` samples must be provided.
    Default: 9 s × 25 Hz = 225 raw samples per axis (→ 450 after resampling).

    **FHIR output:**
    The response includes a complete FHIR R4 Observation resource in
    `fhir_observation`. If `FHIR_SERVER_URL` is configured in `.env`,
    the server also POSTs the observation to your FHIR server automatically.
    """
    # Snapshot model state under lock so a hot-swap mid-request is safe
    with _model_lock:
        engine       = _engine
        model_config = _model_config
        model_info   = _model_info
        model_version = _current_model_version

    n = len(req.acc_x)
    if len(req.acc_y) != n or len(req.acc_z) != n or len(req.timestamps_ms) != n:
        raise HTTPException(status_code=422,
                            detail="acc_x, acc_y, acc_z, timestamps_ms must all have equal length.")
    if n == 0:
        raise HTTPException(status_code=422, detail="Sensor data arrays cannot be empty.")

    # Barometer required check
    if model_info["uses_barometer"] and not req.pressure:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Model '{model_version}' requires barometer data. "
                "Provide 'pressure' and 'pressure_timestamps_ms' in the request body."
            ),
        )

    timestamp = datetime.now(timezone.utc).isoformat()
    obs_id    = str(uuid.uuid4())
    t_start   = time.monotonic()

    logger.info(f"Predict: patient={req.patient_id}  device={req.device_id}  "
                f"n_samples={n}")

    try:
        # 1. Numpy arrays
        acc_data = np.array([req.acc_x, req.acc_y, req.acc_z])
        acc_time = np.array(req.timestamps_ms)

        # 2. Resample hardware rate → 50 Hz
        if HARDWARE_ACC_SAMPLE_RATE != ACC_SAMPLE_RATE:
            resampler = AccelerometerResampler(
                source_rate=HARDWARE_ACC_SAMPLE_RATE,
                target_rate=ACC_SAMPLE_RATE,
                method=RESAMPLING_METHOD,
            )
            acc_data, acc_time = resampler.process(acc_data, acc_time)
            logger.info(f"  Resampled {HARDWARE_ACC_SAMPLE_RATE}Hz → {ACC_SAMPLE_RATE}Hz "
                        f"({acc_data.shape[1]} samples)")

        # 3. LSB → g  (skip for models that expect raw LSB)
        if not model_config.acc_in_lsb:
            acc_data = convert_lsb_to_g(acc_data)

        # 4. DataFrame
        df = convert_acc_nparray_to_df(acc_data, acc_time)

        # 5. Extract detection window (last 9 s = 450 samples)
        required = int(WINDOW_SIZE_SECONDS * ACC_SAMPLE_RATE)
        pressure      = np.array(req.pressure)               if req.pressure               else None
        pressure_time = np.array(req.pressure_timestamps_ms) if req.pressure_timestamps_ms else None
        window_df, window_pressure, window_pressure_time = compose_detection_window(
            df, required, pressure, pressure_time)

        if len(window_df) < required:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Insufficient data after resampling: got {len(window_df)} samples, "
                    f"need {required} ({WINDOW_SIZE_SECONDS}s × {ACC_SAMPLE_RATE}Hz). "
                    f"Provide at least {int(WINDOW_SIZE_SECONDS * HARDWARE_ACC_SAMPLE_RATE)} "
                    f"raw samples ({WINDOW_SIZE_SECONDS}s × {HARDWARE_ACC_SAMPLE_RATE}Hz)."
                ),
            )

        # 6. XGBoost inference
        result     = engine.predict(window_df, pressure=window_pressure,
                                    pressure_timestamps=window_pressure_time)
        is_fall    = result["is_fall"]
        confidence = result["confidence"]
        threshold  = model_config.threshold
        latency_ms = int((time.monotonic() - t_start) * 1000)

        # 6a. Prometheus metrics (best-effort — never blocks inference)
        _record_prediction(model_version, is_fall, float(confidence), latency_ms / 1000)

        if is_fall:
            logger.info("")
            logger.info(f"  *** FALL DETECTED ***  patient={req.patient_id}  "
                        f"confidence={confidence:.3f}  latency={latency_ms}ms")
            logger.info("")
        else:
            logger.info(f"  → no fall  confidence={confidence:.3f}  latency={latency_ms}ms")

        # 7. Human-readable label
        if is_fall:
            label = ("High confidence fall"    if confidence > 0.75 else
                     "Moderate confidence fall" if confidence > 0.60 else
                     "Low confidence fall")
        else:
            label = ("Near threshold — review recommended" if confidence > 0.40 else
                     "No fall detected")

        # 8. Build FHIR Observation
        fhir_obs = build_fhir_observation(
            fall_detected=is_fall,
            confidence=confidence,
            model_version=model_version,
            patient_id=req.patient_id,
            device_id=req.device_id,
            timestamp=timestamp,
            observation_id=obs_id,
            fhir_base_url=FHIR_BASE_URL,
        )

        # 9. Push to FHIR server (only if fall detected, or if configured to push all)
        fhir_pushed = False
        if FHIR_SERVER_URL:
            if is_fall or not FHIR_PUSH_ON_FALL_ONLY:
                fhir_pushed = await _push_to_fhir_server(fhir_obs)

        # 10. Publish to MQTT broker — caregiver dashboards and mobile app subscribe to this
        if is_fall:
            await _publish_fall_event({
                "patient_id":       req.patient_id,
                "device_id":        req.device_id,
                "timestamp":        timestamp,
                "fall_detected":    True,
                "confidence":       round(float(confidence), 4),
                "model_version":    model_version,
                "observation_id":   obs_id,
                "fhir_observation": fhir_obs,
            })

        return PredictResponse(
            patient_id=req.patient_id,
            device_id=req.device_id,
            timestamp=timestamp,
            inference=InferenceResult(
                fall_detected=is_fall,
                confidence=round(float(confidence), 4),
                threshold=float(threshold),
                result=label,
                model_version=model_version,
                window_size=len(window_df),
            ),
            fhir_observation=fhir_obs,
            fhir_pushed=fhir_pushed,
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# ---------------------------------------------------------------------------
# Model management endpoints
# ---------------------------------------------------------------------------

@app.get("/model/list")
async def model_list():
    """Return all model versions available on disk."""
    return {"available_versions": list_available_models()}


@app.post("/model/switch", response_model=ModelSwitchResponse,
          dependencies=[Depends(verify_api_key)])
async def model_switch(req: ModelSwitchRequest):
    """
    Hot-swap the loaded model without restarting the server.

    Uses a threading lock so any in-flight /predict calls complete before
    the switch takes effect.  The new model is used for all subsequent
    /predict calls.
    """
    global _engine, _current_model_version, _model_info, _model_config

    new_version = req.version.strip()
    available   = list_available_models()
    if new_version not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{new_version}' not found. Available: {available}",
        )

    previous = _current_model_version
    try:
        new_type   = get_model_name(new_version)
        new_path   = get_model_path(new_type)
        new_engine = PipelineSelector(new_version, new_path)
        new_info   = new_engine.get_model_info()
        new_config = get_model_config(new_type)

        with _model_lock:
            _engine                = new_engine
            _current_model_version = new_version
            _model_info            = new_info
            _model_config          = new_config

        logger.info(f"Model switched: {previous} → {new_version}  ({new_info['name']})")
        return ModelSwitchResponse(
            success=True,
            previous_version=previous,
            new_version=new_version,
            model_name=new_info["name"],
            num_features=new_info["num_features"],
            uses_barometer=new_info["uses_barometer"],
        )
    except Exception as exc:
        logger.error(f"Model switch failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model switch failed: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("SERVER_PORT", 8001))
    uvicorn.run("inference_server.server:app",
                host="0.0.0.0", port=port, workers=1, reload=False)
