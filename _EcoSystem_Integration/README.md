# Fall Detection — Ecosystem Integration Package

Stripped-down, self-contained build for integrating the fall detection
ML pipeline into an existing production system.

```
_EcoSystem_Integration/        ← copy this entire folder anywhere
├── .env                       ← single config file for both components
├── .env.example               ← template with all available variables
├── README.md                  ← you are here
├── DATA_FLOW.md               ← full data transformation diagram
├── app/                       ← shared ML pipeline (do not edit)
│   ├── core/                  ← inference engine, model registry
│   └── data_input/            ← InfluxDB fetcher, resampler, barometer, converter
├── config/                    ← settings.py — loads .env at import time
├── model/                     ← XGBoost .pkl files (v0, v3, v0_lsb_int, v5_lsb)
├── inference_server/
│   └── server.py              ← FastAPI :8001 — inference only, no DB/Redis/Prometheus
└── trigger_client/
    └── client.py              ← InfluxDB polling loop → POST /predict → callback
```

**Fully self-contained** — no other files from the parent project needed.
Always run both components from inside `_EcoSystem_Integration/` as the working directory.

---

## What was removed vs. the full system

| Full system component | Status |
|---|---|
| Redis pub/sub, patient feedback loop, emergency alerts | Removed |
| PostgreSQL inference logging | Removed |
| Prometheus metrics + Grafana | Removed |
| Caregiver / patient / emergency dashboards | Removed |
| MinIO datalake + CSV replay | Removed |
| Flask web UI, Live Monitor, recording state | Removed |
| **`POST /predict`** — inference endpoint | **Kept** |
| **`GET/POST /config`** — runtime config | **Kept** |
| **`GET /health`, `/model/info`, `/model/list`** | **Kept** |
| **InfluxDB polling + barometer auto-detect** | **Kept** |
| **API key auth + rate limiting** | **Kept** |

---

## Quick start

```bash
# 1. Enter the folder — all commands run from here
cd _EcoSystem_Integration

# 2. Install dependencies
pip install -r inference_server/requirements.txt
pip install -r trigger_client/requirements.txt

# 3. Configure — edit .env (already contains working defaults)
#    Minimum required changes: INFLUXDB_* values and API key pair

# 4. Terminal 1 — start inference server
uvicorn inference_server.server:app --host 0.0.0.0 --port 8001

# 5. Terminal 2 — start trigger client
python -m trigger_client.client
```

Verify the server is running:
```bash
curl http://localhost:8001/health
curl http://localhost:8001/model/list
curl http://localhost:8001/docs      # interactive Swagger UI
```

---

## Environment variables — who reads what

Every variable is in a single `.env` at the `_EcoSystem_Integration/` root.
Both components load it automatically via `load_dotenv()` on startup.

| Variable | Read by | Purpose |
|---|---|---|
| `PUBLIC_ENDPOINT_ENABLED` | **Server** | Enable X-API-Key enforcement on /predict |
| `API_KEYS` | **Server** | Accepted API keys (comma-separated) |
| `REMOTE_API_KEY` | **Client** | Key sent as X-API-Key header — must match `API_KEYS` |
| `SERVER_PORT` | **Server** | Port uvicorn binds to |
| `REMOTE_SERVER_URL` | **Client** | Base URL of the inference server |
| `MODEL_VERSION` | **Server + Client** | Default model; client sends this in every /predict request |
| `PRELOAD_ALL_MODELS` | **Server** | Load all .pkl files at startup (eliminates cold-load delay) |
| `HARDWARE_ACC_SAMPLE_RATE` | **Server + Client** | Physical Hz of sensor; server uses for resampling default, client includes in payload |
| `ACC_SENSOR_TYPE` | **Server + Client** | `bosch` → `bosch_acc_x/y/z`; `non_bosch` → `acc_x/y/z` in InfluxDB |
| `RESAMPLING_METHOD` | **Server** | `linear` (upsample) / `decimate` / `average` (downsample) to 50 Hz |
| `INFLUXDB_URL` | **Client** | InfluxDB cloud URL |
| `INFLUXDB_TOKEN` | **Client** | Auth token |
| `INFLUXDB_ORG` | **Client** | Organisation name |
| `INFLUXDB_BUCKET` | **Client** | Bucket to query |
| `MONITORING_INTERVAL_SECONDS` | **Client** | How often to query InfluxDB (default 5s) |
| `MONITORING_LOOKBACK_SECONDS` | **Client** | How far back each query fetches (default 15s) |
| `RATE_LIMIT_PER_MINUTE` | **Server** | Max requests per IP per minute |
| `CORS_ALLOWED_ORIGINS` | **Server** | CORS origin whitelist (`*` for any) |
| `PARTICIPANT_ID` | **Client** | Optional device/subject ID forwarded in each request |

---

## Selecting a model version

The server caches every model it loads.  Clients select which model to use
**per request** — no server restart, no global model switch.

### Option 1 — via `.env` (simplest, affects entire client process)

```env
MODEL_VERSION=v3
```

On startup the client queries `GET /model/info?version=v3`, discovers
`uses_barometer=true`, and automatically starts fetching barometer data
from InfluxDB.  Every `/predict` request includes `"model_version": "v3"`.

### Option 2 — via constructor argument (useful when embedding the client)

```python
from trigger_client.client import FallDetectionClient

# v0: ACC only — barometer NOT fetched from InfluxDB
client_v0 = FallDetectionClient(model_version="v0")

# v3: ACC + barometer — barometer IS fetched automatically
client_v3 = FallDetectionClient(model_version="v3")
```

### Option 3 — multiple clients in parallel (different models simultaneously)

```python
import threading
from trigger_client.client import FallDetectionClient

def on_v0(result):
    print(f"v0  fall={result['fall_detected']}  conf={result['confidence']:.2%}")

def on_v3(result):
    print(f"v3  fall={result['fall_detected']}  conf={result['confidence']:.2%}")

c0 = FallDetectionClient(model_version="v0", on_result=on_v0)
c3 = FallDetectionClient(model_version="v3", on_result=on_v3)

c0.start_async()   # background thread
c3.start()         # blocks — Ctrl+C to stop both
```

Each client independently discovers whether it needs barometer data.
The server handles both request streams with no configuration change —
v0 and v3 are cached after the first call to each.

### Available model versions

| Version | Features | Barometer | Notes |
|---|---|---|---|
| `v0` | 16 (ACC statistical) | No | Lightest, no baro hardware needed |
| `v0_lsb_int` | 16 (ACC, raw LSB) | No | Raw integer input variant |
| `v3` | 20 (ACC + baro slope-limit) | **Yes** | Best accuracy — recommended |
| `v5_lsb` | 22 (raw features, LSB) | Yes | Raw feature variant |

All models: XGBoost, threshold = 0.5, window = 9s @ 50Hz (450 samples).

---

## API reference

### `POST /predict`

```
Header: X-API-Key: <your key>
```

```json
{
  "acc_x":                  [float, ...],
  "acc_y":                  [float, ...],
  "acc_z":                  [float, ...],
  "timestamps_ms":          [float, ...],
  "model_version":          "v3",            // optional — overrides server default
  "pressure":               [float, ...],    // required when model uses barometer
  "pressure_timestamps_ms": [float, ...],
  "sample_rate":            25,              // optional — hardware Hz
  "acc_unit":               "lsb",          // "lsb" (default) or "g"
  "participant":            "device_01"      // optional — logged in server output
}
```

```json
{
  "fall_detected": true,
  "confidence":    0.87,
  "threshold":     0.5,
  "result":        "High confidence fall detection",
  "model_version": "v3",
  "model_name":    "V3",
  "num_features":  20,
  "window_size":   450,
  "features":      { "acc_x_min": -0.12, ... }
}
```

### `GET /model/info?version=v3`

Returns metadata for any version — including `uses_barometer`.
Used by the client on startup to decide whether to fetch baro data from InfluxDB.

```json
{
  "version": "v3",
  "name": "V3",
  "uses_barometer": true,
  "acc_preprocessing": "v1_features",
  "baro_preprocessing": "v2_paper",
  "num_features": 20
}
```

### `GET /model/list`

```json
{
  "available_versions": ["v0", "v0_lsb_int", "v3", "v5_lsb"],
  "cached_versions":    ["v0", "v3"],
  "default_version":    "v0"
}
```

### `GET /config` / `POST /config`

Update preprocessing config at runtime without restarting the server:

```bash
curl -X POST http://localhost:8001/config \
  -H "Content-Type: application/json" \
  -d '{"hardware_sample_rate": 25, "window_seconds": 9, "acc_sensor_type": "bosch"}'
```

### `GET /health`

```json
{ "status": "ok", "model_version": "v0", "uptime_seconds": 142.3 }
```

---

## Integration point — the `on_result` callback

Replace `_default_result_handler` with whatever your system needs:

```python
from trigger_client.client import FallDetectionClient

def on_result(result: dict) -> None:
    if result["fall_detected"]:
        # write InfluxDB annotation, call alert API, publish MQTT, etc.
        pass

client = FallDetectionClient(
    model_version="v3",
    on_result=on_result,
    participant="ward_b_bed_12",
)
client.start_async()
```

`result` dict keys:

| Key | Type | Description |
|---|---|---|
| `fall_detected` | bool | True if model predicted a fall |
| `confidence` | float 0–1 | XGBoost fall probability |
| `threshold` | float | Decision threshold (0.5) |
| `result` | str | Human-readable label |
| `model_version` | str | Model that produced this result |
| `window_size` | int | ACC samples used (typically 450) |
| `features` | dict | All feature values used for this prediction |
| `timestamp` | ISO-8601 str | UTC time of this detection cycle |
| `participant` | str \| None | Device/subject ID |

---

## Data flow

See [DATA_FLOW.md](DATA_FLOW.md) for the full step-by-step diagram.

```
InfluxDB (cloud)
  │  query last ~15s of bosch_acc_x/y/z  [+ bmp_pressure if model needs it]
  ▼
trigger_client/client.py
  │  { acc_x[], acc_y[], acc_z[], timestamps_ms[], model_version, ... }
  │  POST /predict  +  X-API-Key
  ▼
inference_server/server.py  (FastAPI :8001)
  │  1. Load model from cache (lazy, thread-safe)
  │  2. Resample  hardware_rate → 50 Hz
  │  3. LSB → g
  │  4. Extract last 9s window (450 samples)
  │  5. Feature extraction (16–22 features)
  │  6. XGBoost.predict_proba → confidence
  │  7. threshold 0.5 → { fall_detected, confidence }
  ▼
JSON response  →  on_result() callback  →  your system
```
