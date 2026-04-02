"""
Fall Detection REST API Server

FastAPI server for ML inference. Accepts sensor data via HTTP,
runs the full preprocessing + XGBoost inference pipeline, and
returns fall detection predictions.

Usage:
    python server.py

Or with uvicorn directly:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.inference_engine import PipelineSelector
from app.core.model_registry import get_model_name, get_model_config
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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f"server_{MODEL_VERSION}_{timestamp}.log")

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
# Model Initialization (runs once at startup)
# ---------------------------------------------------------------------------

model_type = get_model_name(MODEL_VERSION)
model_config = get_model_config(model_type)
inference_engine = PipelineSelector(MODEL_VERSION, MODEL_PATH)
model_info = inference_engine.get_model_info()

logger.info("=" * 60)
logger.info("Fall Detection API Server")
logger.info(f"  Model Version:   {MODEL_VERSION.upper()}")
logger.info(f"  Model Path:      {MODEL_PATH}")
logger.info(f"  Model Name:      {model_info['name']}")
logger.info(f"  Uses Barometer:  {model_info['uses_barometer']}")
logger.info(f"  Features:        {model_info['num_features']}")
logger.info(f"  Window:          {WINDOW_SIZE_SECONDS}s @ {ACC_SAMPLE_RATE}Hz")
logger.info("=" * 60)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for the /predict endpoint."""

    acc_x: List[float] = Field(..., description="Accelerometer X-axis values")
    acc_y: List[float] = Field(..., description="Accelerometer Y-axis values")
    acc_z: List[float] = Field(..., description="Accelerometer Z-axis values")
    timestamps_ms: List[float] = Field(
        ..., description="Timestamps in milliseconds"
    )
    pressure: Optional[List[float]] = Field(
        None, description="Barometer pressure values (Pa)"
    )
    pressure_timestamps_ms: Optional[List[float]] = Field(
        None, description="Barometer timestamps in milliseconds"
    )
    sample_rate: Optional[float] = Field(
        None,
        description=(
            "Hardware sampling rate in Hz. "
            "If not provided, uses HARDWARE_ACC_SAMPLE_RATE from server .env."
        ),
    )
    acc_unit: Optional[str] = Field(
        None,
        description=(
            "Unit of accelerometer values: 'lsb' or 'g'. "
            "If not provided, server decides based on model config."
        ),
    )


class PredictResponse(BaseModel):
    """Response body for the /predict endpoint."""

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
    """Response body for the /model/info endpoint."""

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
    """Response body for the /health endpoint."""

    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


# ---------------------------------------------------------------------------
# API Key Authentication (reused logic from Flask middleware)
# ---------------------------------------------------------------------------

_server_start_time = datetime.now()

# Simple in-memory rate limiting
_rate_limit_storage: Dict[str, List[float]] = {}


async def verify_api_key(request: Request):
    """Dependency that checks API key when PUBLIC_ENDPOINT_ENABLED=true."""
    if not PUBLIC_ENDPOINT_ENABLED:
        return

    # Rate limiting
    import time

    client_ip = request.headers.get("X-Forwarded-For", request.client.host)
    now = time.time()
    window_start = now - 60

    if client_ip not in _rate_limit_storage:
        _rate_limit_storage[client_ip] = []
    _rate_limit_storage[client_ip] = [
        t for t in _rate_limit_storage[client_ip] if t > window_start
    ]
    if len(_rate_limit_storage[client_ip]) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_PER_MINUTE} requests per minute.",
        )
    _rate_limit_storage[client_ip].append(now)

    # API key check
    if not API_KEYS:
        logger.warning("PUBLIC_ENDPOINT_ENABLED but no API_KEYS configured!")
        return

    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide API key via X-API-Key header.",
        )
    if api_key not in API_KEYS:
        logger.warning(f"Invalid API key attempt from {client_ip}")
        raise HTTPException(status_code=403, detail="Invalid API key.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fall Detection API",
    description="ML inference API for fall detection using XGBoost models.",
    version="1.0.0",
)

# CORS
if PUBLIC_ENDPOINT_ENABLED:
    origins = (
        ["*"]
        if CORS_ALLOWED_ORIGINS == "*"
        else [o.strip() for o in CORS_ALLOWED_ORIGINS.split(",")]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    uptime = (datetime.now() - _server_start_time).total_seconds()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=MODEL_VERSION,
        uptime_seconds=round(uptime, 1),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Return metadata about the currently loaded model."""
    return ModelInfoResponse(**model_info)


@app.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Depends(verify_api_key)],
)
async def predict(req: PredictRequest):
    """
    Run fall detection inference on a window of sensor data.

    The server handles the full preprocessing pipeline:
    1. Resampling (if sample_rate != 50Hz)
    2. LSB-to-g conversion (if needed by model)
    3. Window extraction
    4. Feature extraction
    5. XGBoost prediction
    """
    # Validate input lengths match
    n = len(req.acc_x)
    if len(req.acc_y) != n or len(req.acc_z) != n or len(req.timestamps_ms) != n:
        raise HTTPException(
            status_code=422,
            detail="acc_x, acc_y, acc_z, and timestamps_ms must all have the same length.",
        )
    if n == 0:
        raise HTTPException(status_code=422, detail="Sensor data arrays cannot be empty.")

    logger.info(f"Predict request: {n} samples")

    try:
        # --- 1. Convert to numpy arrays (shape: 3 x N) ---
        acc_data = np.array([req.acc_x, req.acc_y, req.acc_z])
        acc_time = np.array(req.timestamps_ms)

        # --- 2. Resampling ---
        hw_rate = req.sample_rate or HARDWARE_ACC_SAMPLE_RATE
        if hw_rate != ACC_SAMPLE_RATE:
            resampler = AccelerometerResampler(
                source_rate=hw_rate,
                target_rate=ACC_SAMPLE_RATE,
                method=RESAMPLING_METHOD,
            )
            acc_data, acc_time = resampler.process(acc_data, acc_time)
            logger.info(
                f"  Resampled {hw_rate}Hz -> {ACC_SAMPLE_RATE}Hz ({acc_data.shape[1]} samples)"
            )

        # --- 3. LSB-to-g conversion ---
        # Determine if we need to convert:
        #   - If model expects LSB (acc_in_lsb=True) -> keep raw
        #   - If model expects g (acc_in_lsb=False) AND data is in LSB -> convert
        data_is_lsb = (req.acc_unit or "lsb") == "lsb"
        if not model_config.acc_in_lsb and data_is_lsb:
            acc_data = convert_lsb_to_g(acc_data)
            logger.info(f"  Converted LSB -> g (sensitivity={ACC_SENSOR_SENSITIVITY})")
        elif model_config.acc_in_lsb and not data_is_lsb:
            logger.warning(
                "Model expects LSB but received g values. Results may be inaccurate."
            )

        # --- 4. Convert to DataFrame ---
        df = convert_acc_nparray_to_df(acc_data, acc_time)

        # --- 5. Window extraction ---
        required_samples = int(WINDOW_SIZE_SECONDS * ACC_SAMPLE_RATE)

        pressure = None
        pressure_time = None
        if req.pressure and req.pressure_timestamps_ms:
            pressure = np.array(req.pressure)
            pressure_time = np.array(req.pressure_timestamps_ms)

        window_df, window_pressure, window_pressure_time = compose_detection_window(
            df, required_samples, pressure, pressure_time
        )
        logger.info(f"  Window: {len(window_df)} ACC samples")

        # --- 6. Inference ---
        result = inference_engine.predict(
            window_df,
            pressure=window_pressure,
            pressure_timestamps=window_pressure_time,
        )

        is_fall = result["is_fall"]
        confidence = result["confidence"]
        features_dict = result["features"]
        threshold = model_config.threshold

        # Result message
        if is_fall:
            if confidence > 0.75:
                result_message = "High confidence fall detection"
            elif confidence > 0.60:
                result_message = "Moderate confidence fall detection"
            else:
                result_message = "Low confidence fall detection"
        else:
            if confidence > 0.40:
                result_message = "Close to threshold - consider manual review"
            elif confidence > 0.25:
                result_message = "Borderline case"
            else:
                result_message = "Clear negative"

        logger.info(f"  Result: fall={is_fall}, confidence={confidence:.3f}")

        return PredictResponse(
            fall_detected=is_fall,
            confidence=float(confidence),
            threshold=float(threshold),
            result=result_message,
            model_version=MODEL_VERSION,
            model_name=model_info["name"],
            num_features=model_info["num_features"],
            acc_features=model_info["acc_features"],
            baro_features=model_info["baro_features"],
            window_size=len(window_df),
            features=features_dict,
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("SERVER_PORT", "8000"))
    logger.info(f"Starting server on 0.0.0.0:{port}")
    if PUBLIC_ENDPOINT_ENABLED:
        logger.info(f"  API key auth: enabled ({len(API_KEYS)} keys)")
        logger.info(f"  Rate limit:   {RATE_LIMIT_PER_MINUTE} req/min")
    uvicorn.run(app, host="0.0.0.0", port=port)
