# Plan: Add FastAPI Server for ML Inference (REST API)

## Context

The project is a fall detection system that currently runs as a Flask app on a single machine, tightly coupled with InfluxDB for sensor data. The goal is to enable a **client-server architecture** where:

- **Server** (another laptop): Hosts the ML model via FastAPI, accepts sensor data, runs inference, returns predictions
- **Client** (this laptop): Queries InfluxDB, sends sensor data to the server via HTTP

This allows cloning the repo on the server laptop, running `pip install` + `python server.py`, and having a working inference API.

---

## What Changes

### 1. Create `server.py` (new file - project root)

FastAPI entry point for the ML inference server. Handles:

- **`POST /predict`** - Accept a window of sensor data as JSON, run full preprocessing + inference, return prediction
    - Request body: `acc_x[]`, `acc_y[]`, `acc_z[]`, `timestamps_ms[]`, optional `pressure[]`, `pressure_timestamps_ms[]`, optional `sample_rate` (default 50), optional `acc_unit` ("lsb"/"g")
    - Server does: numpy conversion → resampling (if sample_rate != 50Hz) → LSB-to-g conversion (if needed) → DataFrame creation → window extraction → feature extraction → XGBoost inference
    - Response: `fall_detected`, `confidence`, `threshold`, `model_version`, `model_name`, `features`, etc.
- **`GET /model/info`** - Return loaded model metadata
- **`GET /health`** - Health check

Uses Pydantic models for request/response validation. Loads `PipelineSelector` at startup. Runs via uvicorn on `0.0.0.0:8000`.

Key imports from existing code (no duplication):

- `app.core.inference_engine.PipelineSelector`
- `app.core.model_registry.get_model_name, get_model_config`
- `app.data_input.data_converter.convert_lsb_to_g, convert_acc_nparray_to_df, compose_detection_window`
- `app.data_input.accelerometer_processor.acc_resampler.AccelerometerResampler`
- `config.settings` (MODEL_VERSION, MODEL_PATH, ACC_SAMPLE_RATE, etc.)
- `config.hardware_config.ACC_SENSOR_SENSITIVITY`

### 2. Create `requirements.server.txt` (new file)

Lightweight dependency list for the server (no InfluxDB, Flask, matplotlib, etc.):

`numpy
pandas
scikit-learn
xgboost
joblib
scipy
python-dotenv
fastapi
uvicorn`

### 3. Create `.env.server.example` (new file)

Server-specific env template with only the settings the server needs:

- `MODEL_VERSION`, `HARDWARE_ACC_SAMPLE_RATE`, `ACC_SENSOR_TYPE`, `RESAMPLING_METHOD`
- `PUBLIC_ENDPOINT_ENABLED`, `API_KEYS`, `RATE_LIMIT_PER_MINUTE`
- No InfluxDB settings, no monitoring, no export paths

### 4. Modify `app/data_input/data_converter.py`

- Make `influxdb_client` import conditional with try/except (server won't have influxdb-client installed)
- Remove unused imports: `MODEL_VERSION`, `COLLECT_ADDITIONAL_SENSORS`, `FALL_DATA_EXPORT_DIR`, `TIMEZONE_OFFSET_HOURS` from config.settings (these are leftover from a refactor and not used in any function)

### 5. Add `fastapi` and `uvicorn` to `requirements.txt`

So both client and server environments support the new stack.

---

## What Stays Unchanged

- All existing Flask code (`main.py`, `app/routes/`, etc.) - the client keeps using Flask
- `app/core/` - inference engine, model config, model registry (reused as-is)
- `app/data_input/` preprocessors - accelerometer, barometer processors (reused as-is)
- `model/` directory - model files (reused as-is)
- `config/settings.py` - loads all settings, server ignores InfluxDB ones (they default to empty strings)

---

## Verification

1. On server laptop: clone repo → `pip install -r requirements.server.txt` → copy `.env.server.example` to `.env` → configure MODEL_VERSION → `python server.py`
2. Test health: `curl http://SERVER_IP:8000/health`
3. Test model info: `curl http://SERVER_IP:8000/model/info`
4. Test prediction: `curl -X POST http://SERVER_IP:8000/predict -H "Content-Type: application/json" -d '{"acc_x":[...], "acc_y":[...], "acc_z":[...], "timestamps_ms":[...]}'`
5. Expose publicly: use ngrok (`ngrok http 8000`) or Cloudflare Tunnel per the fast_API.md guide

## Changes Made

### New Files

1. **server.py** - FastAPI server entry point with 3 endpoints:
    - `POST /predict` - Accepts sensor data (acc_x/y/z + timestamps), runs full preprocessing pipeline (resampling, LSB-to-g conversion, windowing, feature extraction) + XGBoost inference, returns prediction
    - `GET /model/info` - Returns loaded model metadata
    - `GET /health` - Health check with uptime
    - Includes API key auth + rate limiting (reused from existing Flask middleware logic)
    - Uses Pydantic for request/response validation
    - Auto-generates OpenAPI docs at `/docs`
2. **requirements.server.txt** - Lightweight deps (no InfluxDB, Flask, matplotlib, etc.)
3. **.env.server.example** - Server-specific env template with only relevant settings

### Modified Files

1. **data_converter.py** - Made `influxdb_client` import conditional with try/except, removed unused config imports
2. **requirements.txt** - Added `fastapi` and `uvicorn`

## How to Use on the Server Laptop

`# 1. Clone the repo
git clone <your-repo-url>
cd FallDetection_new

# 2. Install server dependencies only
pip install -r requirements.server.txt

# 3. Configure
cp .env.server.example .env
# Edit .env: set MODEL_VERSION, HARDWARE_ACC_SAMPLE_RATE, etc.

# 4. Start the server
python server.py
# Server starts on 0.0.0.0:8000

# 5. Expose publicly (pick one)
ngrok http 8000                    # ngrok tunnel
cloudflared tunnel --url http://localhost:8000  # Cloudflare tunnel`

The client on this laptop sends sensor data via HTTP POST to `/predict` with JSON containing the accelerometer arrays, and the server handles all ML preprocessing + inference.