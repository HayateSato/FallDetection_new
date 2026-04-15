# Fall Detection — 6G / Charite Integration (MQTT)

Three-component fall detection stack designed for the FOCUS / Charite monitoring
ecosystem. The inference server returns FHIR R4 Observation resources over HTTP.
The mobile app (mock or real) handles patient confirmation and publishes a confirmed
alert over MQTT. The caregiver client subscribes to those alerts and shows a
real-time dashboard.

```
_6G_Integration_v2_mqtt/
├── .env                        ← single shared config
├── README.md                   ← you are here
├── fhir_converter.py           ← builds FHIR R4 Observation
├── app/                        ← shared ML pipeline (do not edit)
├── config/                     ← settings.py reads .env
├── model/                      ← XGBoost .pkl files
│
├── inference_server/           ← HTTP-only ML server
│   ├── server.py               ← FastAPI :8001
│   ├── services/
│   │   └── metrics_collector.py  ← Prometheus metrics
│   └── requirements.txt
│
├── mock_app/                   ← simulates the mobile app
│   ├── client.py               ← entry point: python -m mock_app.client
│   ├── poller.py               ← fetch → infer → confirm → publish
│   ├── influx_fetcher.py       ← queries InfluxDB for ACC windows
│   ├── api_caller.py           ← HTTP client to inference server
│   └── requirements.txt
│
└── caregiver_client/           ← dashboard (subscribes to MQTT alerts)
    ├── client.py               ← entry point: python -m caregiver_client.client
    ├── mqtt_listener.py        ← FallEventBroker: MQTT → SSE fan-out
    ├── web.py                  ← FastAPI :8002 (JSON API + SSE + dashboard)
    ├── db.py                   ← SQLAlchemy: participant_session + fall_history
    ├── dashboard/
    │   ├── index.html
    │   ├── app.js
    │   └── style.css
    └── requirements.txt
```

---

## Architecture

### Message flow

```
[InfluxDB]
    │  fetch ACC window
    ▼
[mock_app]  ──── HTTP POST /predict ────►  [inference_server :8001]
    ◄─────────── HTTP response ──────────  fall_detected, confidence, FHIR
    │
    │  10s patient confirmation window
    │  (real app: popup on patient's phone)
    │  (mock: waits MOCK_PATIENT_RESPONSE_TIMEOUT seconds)
    │
    │  MQTT PUBLISH  fall/alert/<patient_id>
    ▼
[MQTT broker :1883]
    │  route to subscribers of fall/alert/#
    ▼
[caregiver_client :8002]
    │  DB write (SQLite)
    │  SSE fan-out
    ▼
[caregiver dashboard browser]
    alert: "patient-001 has fallen at 10:00:10 and needs help"
```

### MQTT clients

| Component | Role | Topic |
|-----------|------|-------|
| `mock_app` | publisher | `fall/alert/<patient_id>` |
| `caregiver_client` | subscriber | `fall/alert/#` |

The inference server has **no MQTT client**. It receives requests and returns
responses over HTTP. The mobile app already has the fall result in the HTTP
response — no second channel is needed.

### Component responsibilities

| Concern | Component |
|---------|-----------|
| Sensor data fetching | mock_app (real app: BLE wearable) |
| ML inference | inference_server |
| FHIR conversion | inference_server (returned in HTTP response) |
| Patient confirmation popup | mock_app (real app: phone UI) |
| Confirmed alert delivery | mock_app publishes MQTT |
| Fall history storage | caregiver_client (own SQLite/Postgres DB) |
| Live dashboard push | caregiver_client MQTT → SSE → browser |
| FHIR auto-push to partner server | inference_server (optional) |

---

## Quick start

```powershell
cd _6G_Integration_v2_mqtt

# 1 — Install dependencies
pip install -r inference_server/requirements.txt
pip install -r mock_app/requirements.txt
pip install -r caregiver_client/requirements.txt

# 2 — Start MQTT broker (first time only)
docker run -d --name mqtt-local -p 1883:1883 eclipse-mosquitto

# 3 — Edit .env  (see Configuration section below)

# 4 — Terminal 1: inference server
uvicorn inference_server.server:app --host 0.0.0.0 --port 8001

# 5 — Terminal 2: caregiver dashboard
python -m caregiver_client.client
# Dashboard: http://localhost:8002/

# 6 — Terminal 3: mock mobile app
python -m mock_app.client
```

Verify:
```powershell
curl.exe http://localhost:8001/health
curl.exe http://localhost:8001/model/info      # check uses_barometer
curl.exe http://localhost:8002/api/patients    # caregiver API
```

---

## Inference Server API (port 8001)

### `POST /predict` — header: `X-API-Key: <your key>`

```json
{
  "patient_id":              "charite-patient-007",
  "device_id":               "smarko-wearable-42",
  "acc_x":                   [-512, -498, "..."],
  "acc_y":                   [128, 134, "..."],
  "acc_z":                   [16300, 16280, "..."],
  "timestamps_ms":           [1712345678000, 1712345678040, "..."],
  "pressure":                [101325.0, 101322.5, "..."],
  "pressure_timestamps_ms":  [1712345678000, 1712345678040, "..."]
}
```

Input ACC values must be **raw LSB integers** (as recorded by the SmarKo app).
The server converts LSB → g and resamples from hardware rate to 50 Hz internally.

The response includes a complete FHIR R4 Observation in `fhir_observation`.
If `FHIR_SERVER_URL` is set, the server also POSTs the observation there automatically.

### `GET /model/info`

Returns the loaded model metadata including the `uses_barometer` flag — the mock_app
reads this to decide whether to fetch barometer data from InfluxDB.

### `GET /model/list`

Returns all model versions available on disk.

### `POST /model/switch` — header: `X-API-Key: <your key>`

Hot-swaps the loaded model without restarting the server.

```json
{ "version": "v3" }
```

### `GET /health`

Liveness check. Returns model version, uptime, sensor config.

### `GET /metrics`

Prometheus metrics endpoint (requires `prometheus-fastapi-instrumentator` installed).
Exposes: `fall_detections_total`, `inference_latency_seconds`, `model_confidence`.

---

## Caregiver Client API (port 8002)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Dashboard HTML/JS |
| GET | `/api/patients` | Patient list with fall counts |
| GET | `/api/falls?patient_id=&limit=` | Fall history rows |
| GET | `/api/stream` | Server-Sent Events feed of live confirmed alerts |

The dashboard:
- **Patients tab** — one card per registered patient, fall count badge, red border on patients who have fallen.
- **Fall History tab** — table of confirmed alerts with patient_confirmed status.
- **Alert banner** — flashes the instant a confirmed alert arrives over SSE.

---

## `fall_history` table — schema

Created automatically on first run by `caregiver_client/db.py`.

| Column | Type | Notes |
|--------|------|-------|
| `id` | int PK | |
| `patient_id` | string, indexed | FHIR Patient identifier |
| `fall_detection` | boolean | True if the model flagged a fall |
| `patient_confirmed` | string | `yes` / `no` / `not_answered` — set by mobile app |
| `detection_time` | datetime | Indexed; defaults to row insertion time |

`patient_confirmed` is written from the MQTT alert payload — the value is determined
by the mobile app during the patient confirmation step, not by the caregiver backend.

---

## Configuration

All three components share a single `.env` in this folder.

| Variable | SERVER | MOCK APP | CAREGIVER | Notes |
|----------|:------:|:--------:|:---------:|-------|
| `MODEL_VERSION` | X | | | Fixed model on server (restart to change) |
| `ACC_SENSOR_TYPE` | X | X | | Both sides must agree on hardware |
| `HARDWARE_ACC_SAMPLE_RATE` | X | X | | Same |
| `RESAMPLING_METHOD` | X | | | Server resamples to 50 Hz |
| `INFLUXDB_*` | | X | | mock_app fetches sensor data |
| `PATIENT_IDS` | | X | X | Patients to poll / register |
| `MAC_IDS` | | X | X | Comma-separated, 1:1 with PATIENT_IDS |
| `POLL_INTERVAL_SECONDS` | | X | | How often mock_app polls InfluxDB |
| `POLL_LOOKBACK_SECONDS` | | X | | Seconds of history to fetch each cycle |
| `INFERENCE_SERVER_URL` | | X | | Where mock_app POSTs /predict |
| `INFERENCE_API_KEY` | | X | | Must match `API_KEYS` on server |
| `API_KEYS` | X | | | Accepted X-API-Key header values |
| `DATABASE_URL` | | | X | SQLite by default; Postgres in production |
| `MQTT_BROKER_HOST` | | X | X | Broker hostname or IP |
| `MQTT_BROKER_PORT` | | X | X | Default 1883 (8883 for TLS) |
| `MQTT_ALERT_TOPIC` | | X | X | Default `fall/alert` |
| `MQTT_USERNAME` | | X | X | Leave empty if broker has no auth |
| `MQTT_PASSWORD` | | X | X | Leave empty if broker has no auth |
| `MOCK_PATIENT_RESPONSE_TIMEOUT` | | X | | Seconds to wait before treating as no-answer |
| `FHIR_SERVER_URL` | X | | | Optional — server auto-pushes observations |
| `CAREGIVER_PORT` | | | X | Default 8002 |
| `SERVER_PORT` | X | | | Default 8001 |

---

## What is NOT included

Compared to the full system on the `complete_system` branch:
- No Grafana dashboards (Prometheus metrics endpoint exists but no dashboard config)
- No patient feedback popup served from this backend (popup is the mobile app's job)
- No Docker Compose (run with three terminals + one Docker container for the broker)
- No JWT auth / login screen on the dashboard
- No MinIO datalake / CSV replay

---

## Development notes

- The inference server runs `--workers 1`. If you need multiple workers, Prometheus
  counters become per-process — use a pushgateway or run a single worker.
- SQLite is the default for zero-setup local testing. Switch to Postgres in production
  by changing `DATABASE_URL` only — no code change needed.
- If `MQTT_BROKER_HOST` is empty: mock_app logs a warning and skips publishing;
  caregiver SSE sends keepalives only; fall history is never written (no alerts arrive).
- `MAC_IDS` uses positional mapping to `PATIENT_IDS` (comma-separated lists, same order).
  Do not use `key:value` format — MAC addresses contain `:` which breaks parsing.
- `python-dotenv` does not strip inline `#` comments. Put comments on their own line.
