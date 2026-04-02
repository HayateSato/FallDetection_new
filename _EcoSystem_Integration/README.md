# Fall Detection — Ecosystem Integration Package

Stripped-down build containing only what is needed to integrate the fall
detection ML pipeline into an existing production system.

```
_EcoSystem_Integration/        ← self-contained, move this folder anywhere
├── README.md                  ← you are here
├── DATA_FLOW.md               ← full data transformation diagram
├── .env                       ← create from inference_server/.env.example
├── app/                       ← shared ML pipeline (do not edit)
│   ├── core/                  ← inference engine, model registry
│   └── data_input/            ← InfluxDB fetcher, resampler, barometer, converter
├── config/                    ← settings.py reads from .env
├── model/                     ← XGBoost .pkl files (v0, v3, v0_lsb_int, v5_lsb)
├── inference_server/
│   ├── server.py              ← FastAPI inference server
│   ├── .env.example           ← environment variable template
│   └── requirements.txt       ← server dependencies
└── trigger_client/
    ├── client.py              ← InfluxDB polling + POST /predict
    ├── .env.example           ← environment variable template
    └── requirements.txt       ← client dependencies
```

**This folder is fully self-contained.**  Copy it anywhere and run from
inside it — no other files from the parent project are needed.

---

## What was removed vs. the full system

| Full system component | Status in this build |
|----------------------|---------------------|
| Redis pub/sub (patient alerts, fall events) | Removed |
| PostgreSQL inference logging | Removed |
| Prometheus metrics + Grafana dashboards | Removed |
| Patient feedback loop (popup, timer, emergency) | Removed |
| Caregiver / emergency dashboards | Removed |
| Live Monitor (subprocess management) | Removed |
| MinIO datalake + CSV replay | Removed |
| Model comparison page | Removed |
| Flask web UI in main.py | Removed |
| CSV export / recording state | Removed |
| Manual truth markers | Removed |
| **`POST /predict`** | **Kept** |
| **`GET/POST /config`** | **Kept** |
| **`GET /health`, `/model/info`, `/model/list`** | **Kept** |
| **`POST /model/switch`** | **Kept** |
| **InfluxDB polling loop** | **Kept** |
| **API key auth + rate limiting** | **Kept** |

---

## Quick start

### 1. Install dependencies

```bash
# From project root
pip install -r _EcoSystem_Integration/inference_server/requirements.txt
pip install -r _EcoSystem_Integration/trigger_client/requirements.txt
```

### 2. Configure environment

```bash
# Copy example files and fill in your values
cp _EcoSystem_Integration/inference_server/.env.example .env
# Edit .env — at minimum set:  MODEL_VERSION, API_KEYS
```

```bash
# For the client (append to .env or export separately):
# REMOTE_SERVER_URL, REMOTE_API_KEY
# INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET
# HARDWARE_ACC_SAMPLE_RATE, ACC_SENSOR_TYPE
```

### 3. Start the inference server

```bash
# From project root
uvicorn _EcoSystem_Integration.inference_server.server:app --host 0.0.0.0 --port 8001
```

Verify:
```bash
curl http://localhost:8001/health
curl http://localhost:8001/model/info
curl http://localhost:8001/docs          # OpenAPI / Swagger UI
```

### 4. Start the trigger client

As a standalone script:
```bash
python -m _EcoSystem_Integration.trigger_client.client
```

As a library inside your existing system:
```python
from _EcoSystem_Integration.trigger_client.client import FallDetectionClient

def handle_result(result: dict) -> None:
    if result["fall_detected"]:
        # Push to your alerting system, write to your DB, etc.
        print(f"FALL  confidence={result['confidence']:.2%}  ts={result['timestamp']}")

client = FallDetectionClient(on_result=handle_result)
client.start_async()   # non-blocking — runs in a daemon thread
```

---

## API Reference

### `POST /predict`

```json
Request:
{
  "acc_x":               [float, ...],     // required — accelerometer X (LSB or g)
  "acc_y":               [float, ...],     // required
  "acc_z":               [float, ...],     // required
  "timestamps_ms":       [float, ...],     // required — millisecond timestamps
  "pressure":            [float, ...],     // optional — barometer Pa (needed for v3 model)
  "pressure_timestamps_ms": [float, ...],  // optional
  "sample_rate":         25,               // optional — hardware Hz (overrides .env)
  "acc_unit":            "lsb",           // optional — "lsb" (default) or "g"
  "participant":         "device_01"       // optional — logged in server output
}

Response:
{
  "fall_detected":  true,
  "confidence":     0.87,
  "threshold":      0.5,
  "result":         "High confidence fall detection",
  "model_version":  "v3",
  "model_name":     "...",
  "num_features":   20,
  "window_size":    450,
  "features":       { "acc_x_min": ..., ... }
}
```

Header required when `PUBLIC_ENDPOINT_ENABLED=true`:
```
X-API-Key: your_api_key
```

### `GET /config` / `POST /config`

```bash
# Get current config
curl http://localhost:8001/config

# Update at runtime (no restart needed)
curl -X POST http://localhost:8001/config \
  -H "Content-Type: application/json" \
  -d '{"hardware_sample_rate": 25, "window_seconds": 9, "acc_sensor_type": "bosch"}'
```

### `POST /model/switch`

```bash
curl -X POST http://localhost:8001/model/switch \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{"version": "v3"}'
```

---

## Integrating the result into your company system

The `on_result` callback in `trigger_client/client.py` is the single
integration point.  Replace `_default_result_handler` with whatever
your company system needs:

```python
def on_result(result: dict) -> None:
    if result["fall_detected"]:
        # Examples:
        # influxdb_client.write_marker(result["timestamp"], "fall")
        # your_alert_api.send(patient_id=result["participant"], ...)
        # mqtt_client.publish("fall_events", json.dumps(result))
        pass
```

Result dict keys:

| Key | Type | Description |
|-----|------|-------------|
| `fall_detected` | bool | True if model predicted a fall |
| `confidence` | float 0–1 | XGBoost fall probability |
| `threshold` | float | Decision threshold (default 0.5) |
| `result` | str | Human-readable label |
| `model_version` | str | e.g. `"v3"` |
| `window_size` | int | Number of ACC samples used (typically 450) |
| `features` | dict | All 16–22 feature values used for this prediction |
| `timestamp` | str | ISO-8601 UTC timestamp of detection cycle |
| `participant` | str \| None | Value from `PARTICIPANT_ID` env or constructor |

---

## Data flow summary

See [DATA_FLOW.md](DATA_FLOW.md) for the full step-by-step diagram.

Short version:

```
InfluxDB (cloud)
  │  influxdb-client query  (lookback ~15 s)
  ▼
trigger_client/client.py
  │  raw LSB arrays + timestamps  →  POST /predict  +  X-API-Key
  ▼
inference_server/server.py  (FastAPI :8001)
  │
  │  1. AccelerometerResampler      25 Hz → 50 Hz  (linear interpolation)
  │  2. convert_lsb_to_g            raw_LSB ÷ 16384 → g
  │  3. convert_acc_nparray_to_df   numpy → DataFrame
  │  4. compose_detection_window    last 9 s = 450 samples
  │  5. PipelineSelector            extract 16–22 features
  │  6. XGBoost.predict_proba       → confidence score
  │  7. threshold (0.5)             → { fall_detected, confidence }
  ▼
JSON response  →  on_result() callback  →  your system
```
