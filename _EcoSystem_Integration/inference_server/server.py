"""
Fall Detection — Inference Server (Integration Build)
======================================================
Stripped-down FastAPI server exposing only the ML inference pipeline.
All multi-user components (Redis, PostgreSQL, Prometheus, patient feedback,
subprocess management, datalake, model comparison) have been removed.

Endpoints
---------
  POST /predict         — run inference, return result
  GET  /health          — liveness check
  GET  /model/info      — loaded model metadata
  GET  /model/list      — models available on disk
  POST /model/switch    — hot-swap model (requires X-API-Key)
  GET  /config          — current runtime processing config
  POST /config          — update runtime processing config
  GET  /docs            — auto-generated OpenAPI / Swagger UI

Run (from project root):
    uvicorn _EcoSystem_Integration.inference_server.server:app --host 0.0.0.0 --port 8001
"""

import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Project-root imports (run from project root so app/ and config/ are on path)
# ---------------------------------------------------------------------------
from app.core.inference_engine import PipelineSelector
from app.core.model_registry import (
    get_model_config,
    get_model_name,
    get_model_path,
    list_available_models,
)
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
        logging.FileHandler(os.path.join(log_dir, f"inference_server_{MODEL_VERSION}_{_ts}.log"),
                            mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model cache — one PipelineSelector per version, loaded on first use
# ---------------------------------------------------------------------------
_model_cache_lock = threading.Lock()
_model_cache: Dict[str, PipelineSelector] = {}   # version → engine

# Default version (from .env) — pre-loaded at startup
_default_model_version = MODEL_VERSION


def _get_engine(version: str) -> tuple:
    """
    Return (engine, model_config, model_info) for `version`.
    Loads and caches on first call; subsequent calls return cached object.
    Thread-safe.
    """
    with _model_cache_lock:
        if version not in _model_cache:
            available = list_available_models()
            if version not in available:
                raise ValueError(f"Unknown model version '{version}'. Available: {list(available)}")
            path = get_model_path(get_model_name(version))
            engine = PipelineSelector(version, path)
            _model_cache[version] = engine
            logger.info(f"Model '{version}' loaded and cached.")
        engine = _model_cache[version]
    mc = get_model_config(get_model_name(version))
    mi = engine.get_model_info()
    return engine, mc, mi


# Pre-warm cache at startup — load every model that has a .pkl file on disk.
# This eliminates the cold-load penalty (100-500ms) on first client request.
# Set PRELOAD_ALL_MODELS=false in .env to skip and only load on demand.
_preload_all = os.getenv("PRELOAD_ALL_MODELS", "true").lower() != "false"
if _preload_all:
    for _v in list_available_models():
        try:
            _get_engine(_v)
        except Exception as _e:
            logger.warning(f"Could not preload model '{_v}': {_e}")
else:
    _get_engine(_default_model_version)

# Runtime processing config (updated by POST /config)
_window_size_seconds:  float = WINDOW_SIZE_SECONDS
_hardware_sample_rate: int   = HARDWARE_ACC_SAMPLE_RATE
_resampling_method:    str   = RESAMPLING_METHOD
_acc_sensor_type:      str   = ACC_SENSOR_TYPE

_default_info = _model_cache[_default_model_version].get_model_info()
logger.info("=" * 60)
logger.info("Fall Detection Inference Server  [integration build]")
logger.info(f"  Default model: {_default_model_version.upper()}  ({_default_info['name']})")
logger.info(f"  All models loaded on first request (lazy cache)")
logger.info(f"  Fixed pipeline:  {WINDOW_SIZE_SECONDS}s window @ {ACC_SAMPLE_RATE}Hz")
logger.info("=" * 60)

# ---------------------------------------------------------------------------
# API key auth + rate limiting
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
    title="Fall Detection Inference Server",
    description="Minimal inference server — POST /predict → {fall_detected, confidence}",
    version="1.0.0",
)

origins = ["*"] if CORS_ALLOWED_ORIGINS == "*" else [o.strip() for o in CORS_ALLOWED_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    acc_x:                   List[float]          = Field(..., description="Accelerometer X-axis values (LSB or g)")
    acc_y:                   List[float]           = Field(..., description="Accelerometer Y-axis values (LSB or g)")
    acc_z:                   List[float]           = Field(..., description="Accelerometer Z-axis values (LSB or g)")
    timestamps_ms:           List[float]           = Field(..., description="Timestamps in milliseconds")
    pressure:                Optional[List[float]] = Field(None, description="Barometer pressure values (Pa)")
    pressure_timestamps_ms:  Optional[List[float]] = Field(None, description="Barometer timestamps in ms")
    sample_rate:             Optional[float]       = Field(None, description="Hardware ACC rate Hz (overrides server .env)")
    acc_unit:                Optional[str]         = Field(None, description="'lsb' (default) or 'g'")
    participant:             Optional[str]         = Field(None, description="Subject/device ID (logged only)")
    model_version:           Optional[str]         = Field(None, description="Model to use for this request (e.g. 'v3'). Defaults to server default.")


class PredictResponse(BaseModel):
    fall_detected:  bool
    confidence:     float
    threshold:      float
    result:         str
    model_version:  str
    model_name:     str
    num_features:   int
    window_size:    int
    features:       Dict[str, float]




class ConfigRequest(BaseModel):
    window_seconds:       Optional[float] = Field(None, ge=1.0, le=60.0,
                                                  description="Detection window in seconds")
    hardware_sample_rate: Optional[int]   = Field(None,
                                                  description="Hardware ACC sample rate Hz (25, 50, or 100)")
    resampling_method:    Optional[str]   = Field(None,
                                                  description="'linear', 'decimate', or 'average'")
    acc_sensor_type:      Optional[str]   = Field(None,
                                                  description="'bosch' or 'non_bosch'")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_config_response() -> dict:
    return {
        "window_seconds":        _window_size_seconds,
        "hardware_sample_rate":  _hardware_sample_rate,
        "resampling_method":     _resampling_method,
        "acc_sensor_type":       _acc_sensor_type,
        "model_sample_rate_hz":  ACC_SAMPLE_RATE,
        "window_samples":        int(_window_size_seconds * ACC_SAMPLE_RATE),
    }

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    uptime = (datetime.now() - _server_start_time).total_seconds()
    return {"status": "ok", "model_version": _current_model_version,
            "uptime_seconds": round(uptime, 1)}


@app.get("/model/info")
async def get_model_info(version: Optional[str] = None):
    """
    Return metadata for a model version.
    Pass ?version=v3 to query a specific version without loading it as default.
    Omit to get info on the server's default model.
    The key field for clients is `uses_barometer` — it tells the trigger client
    whether barometer data needs to be fetched from InfluxDB for this model.
    """
    v = version or _default_model_version
    try:
        _, _, mi = _get_engine(v)
        return mi
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/model/list")
async def list_models():
    """List all model versions available on disk."""
    available = list_available_models()
    cached = list(_model_cache.keys())
    return {"available_versions": list(available), "cached_versions": cached,
            "default_version": _default_model_version}


@app.get("/config")
async def get_config():
    return _build_config_response()


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
            raise HTTPException(status_code=422, detail="resampling_method must be 'linear', 'decimate', or 'average'")
        _resampling_method = req.resampling_method
    if req.acc_sensor_type is not None:
        if req.acc_sensor_type not in ("bosch", "non_bosch"):
            raise HTTPException(status_code=422, detail="acc_sensor_type must be 'bosch' or 'non_bosch'")
        _acc_sensor_type = req.acc_sensor_type

    logger.info(f"Config updated: {_build_config_response()}")
    return _build_config_response()


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)])
async def predict(req: PredictRequest):
    """
    Run fall detection inference on a window of sensor data.

    Pipeline: resample → LSB-to-g → DataFrame → window → features → XGBoost
    """
    n = len(req.acc_x)
    if len(req.acc_y) != n or len(req.acc_z) != n or len(req.timestamps_ms) != n:
        raise HTTPException(status_code=422,
                            detail="acc_x, acc_y, acc_z, timestamps_ms must all have the same length.")
    if n == 0:
        raise HTTPException(status_code=422, detail="Sensor data arrays cannot be empty.")

    # Resolve which model to use for this request
    requested_version = req.model_version or _default_model_version
    logger.info(f"Predict: {n} samples  model={requested_version}  participant={req.participant}")
    t_start = time.monotonic()

    try:
        try:
            engine, mc, mi = _get_engine(requested_version)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        mv = requested_version

        # 1. Build numpy arrays  (shape: 3 × N)
        acc_data = np.array([req.acc_x, req.acc_y, req.acc_z])
        acc_time = np.array(req.timestamps_ms)

        # 2. Resampling  (hardware rate → 50 Hz model rate)
        hw_rate = req.sample_rate or _hardware_sample_rate
        if hw_rate != ACC_SAMPLE_RATE:
            resampler = AccelerometerResampler(
                source_rate=hw_rate, target_rate=ACC_SAMPLE_RATE,
                method=_resampling_method)
            acc_data, acc_time = resampler.process(acc_data, acc_time)
            logger.info(f"  Resampled {hw_rate}Hz → {ACC_SAMPLE_RATE}Hz ({acc_data.shape[1]} samples)")

        # 3. LSB → g conversion
        data_is_lsb = (req.acc_unit or "lsb") == "lsb"
        if not mc.acc_in_lsb and data_is_lsb:
            acc_data = convert_lsb_to_g(acc_data)

        # 4. DataFrame
        df = convert_acc_nparray_to_df(acc_data, acc_time)

        # 5. Detection window  (last N samples)
        required_samples = int(_window_size_seconds * ACC_SAMPLE_RATE)
        pressure      = np.array(req.pressure)               if req.pressure               else None
        pressure_time = np.array(req.pressure_timestamps_ms) if req.pressure_timestamps_ms else None
        window_df, window_pressure, window_pressure_time = compose_detection_window(
            df, required_samples, pressure, pressure_time)

        # 6. XGBoost inference
        result     = engine.predict(window_df, pressure=window_pressure,
                                    pressure_timestamps=window_pressure_time)
        is_fall    = result["is_fall"]
        confidence = result["confidence"]
        features   = result["features"]
        threshold  = mc.threshold
        latency_ms = int((time.monotonic() - t_start) * 1000)

        logger.info(f"  → fall={is_fall}  confidence={confidence:.3f}  latency={latency_ms}ms")

        if is_fall:
            msg = ("High confidence"   if confidence > 0.75 else
                   "Moderate confidence" if confidence > 0.60 else "Low confidence")
            result_str = f"{msg} fall detection"
        else:
            msg = ("Near threshold — review recommended" if confidence > 0.40 else
                   "Borderline"  if confidence > 0.25   else "Clear negative")
            result_str = msg

        return PredictResponse(
            fall_detected=is_fall,
            confidence=float(confidence),
            threshold=float(threshold),
            result=result_str,
            model_version=mv,
            model_name=mi["name"],
            num_features=mi["num_features"],
            window_size=len(window_df),
            features=features,
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("SERVER_PORT", 8001))
    uvicorn.run("_EcoSystem_Integration.inference_server.server:app",
                host="0.0.0.0", port=port, workers=1, reload=False)
