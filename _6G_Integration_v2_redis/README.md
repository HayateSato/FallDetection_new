# Fall Detection — 6G / Charite Integration

Two-component fall detection stack designed for the FOCUS / Charite monitoring
ecosystem. The inference server returns FHIR R4 Observation resources; the
caregiver client polls InfluxDB, calls the server, stores its own fall history,
and shows a minimal real-time dashboard.

```
_6G_Integration/
├── .env                        ← single shared config (annotated SERVER/CLIENT/BOTH)
├── README.md                   ← you are here
├── fhir_converter.py           ← builds FHIR R4 Observation
├── app/                        ← shared ML pipeline (do not edit)
├── config/                     ← settings.py reads .env
├── model/                      ← XGBoost .pkl files
│
├── inference_server/           ← SERVER — pure ML
│   ├── server.py               ← FastAPI :8001
│   └── requirements.txt
│
└── caregiver_client/           ← CLIENT — polls Influx, stores history, dashboard
    ├── client.py               ← entry point (poller + web UI)
    ├── influx_poller.py        ← background thread
    ├── inference_client.py     ← HTTP client to inference server
    ├── redis_listener.py       ← subscribes to fall_events
    ├── web.py                  ← FastAPI :8002 (JSON API + SSE)
    ├── db.py                   ← SQLAlchemy: participant_session + fall_history
    ├── dashboard/              ← minimal HTML/JS/CSS
    │   ├── index.html
    │   ├── app.js
    │   └── style.css
    └── requirements.txt
```

---

## Architecture

```
            ┌──────────────────────────────────────────────────────────┐
            │              CAREGIVER CLIENT  (port 8002)               │
            │                                                          │
   InfluxDB │  ┌─────────────┐    POST /predict     ┌────────────────┐ │
   ────────►│  │ poller      │─────────────────────►│ inference_     │ │
            │  │ (thread)    │◄─────────────────────│ client (HTTP)  │ │
            │  └──────┬──────┘  FHIR Observation    └────────────────┘ │
            │         │                                                │
            │         ▼ writes fall_history row                        │
            │  ┌─────────────┐                                         │
            │  │  SQLite /   │                                         │
            │  │  Postgres   │                                         │
            │  └─────────────┘                                         │
            │         ▲                                                │
            │         │ /api/falls, /api/patients                      │
            │  ┌──────┴──────┐    /api/stream (SSE)   ┌─────────────┐  │
            │  │ web (FastAPI│◄──────────────────────►│  Browser    │  │
            │  │ + dashboard)│                        │ dashboard   │  │
            │  └──────▲──────┘                        └─────────────┘  │
            │         │                                                │
            │  ┌──────┴──────┐                                         │
            │  │ redis       │                                         │
            │  │ subscriber  │                                         │
            │  └──────▲──────┘                                         │
            └─────────┼────────────────────────────────────────────────┘
                      │ SUBSCRIBE fall_events
                      │
            ┌─────────┴────────────────────────────────────────────────┐
            │              INFERENCE SERVER  (port 8001)               │
            │                                                          │
            │   POST /predict                                          │
            │     1. Resample 25Hz → 50Hz                              │
            │     2. LSB → g                                           │
            │     3. XGBoost (model fixed via MODEL_VERSION)           │
            │     4. Build FHIR R4 Observation                         │
            │     5. PUBLISH fall_events  ──────────► (only if fall)   │
            │     6. Optional: POST to FHIR_SERVER_URL                 │
            │     7. Return JSON {inference, fhir_observation}         │
            │                                                          │
            │   GET /model/info  → uses_barometer flag                 │
            │   GET /health                                            │
            └──────────────────────────────────────────────────────────┘
```

**Key separation**

| Concern              | Lives in            |
|----------------------|---------------------|
| Sensor fetching      | client (poller)     |
| ML inference         | server              |
| FHIR conversion      | server (returned to client in `/predict` response) |
| Fall history storage | client (own DB)     |
| Live dashboard push  | server publishes Redis → client subscribes → SSE → browser |
| FHIR auto-push       | server (optional)   |

The poller writes every fall to the client DB the moment `/predict` returns
(durable history). Redis is used purely as the **live notification channel**
so the dashboard can flash a banner the instant a fall is detected.

---

## What is fixed vs. configurable

| Parameter                       | Where                             | Runtime change? |
|---------------------------------|-----------------------------------|-----------------|
| Model version                   | `.env` → `MODEL_VERSION`          | No — restart    |
| Sensor type / hardware rate     | `.env` → `ACC_SENSOR_TYPE` etc.   | No              |
| ACC unit (always raw LSB)       | hardcoded                          | No              |
| Detection window (9 s × 50 Hz)  | hardcoded                          | No              |
| FHIR server URL                 | `.env` → `FHIR_SERVER_URL`         | No              |
| API key                         | `.env` → `API_KEYS`                | No              |
| Patients to poll                | `.env` → `PATIENT_IDS`             | No              |
| Poll interval / lookback        | `.env` → `POLL_*`                  | No              |
| Database                        | `.env` → `DATABASE_URL`            | No              |

---

## Quick start

```bash
cd _6G_Integration

# 1. Install dependencies
pip install -r inference_server/requirements.txt
pip install -r caregiver_client/requirements.txt

# 2. Edit .env — at minimum:
#      MODEL_VERSION         (server)
#      API_KEYS              (server)
#      INFERENCE_API_KEY     (client — must match API_KEYS)
#      INFLUXDB_*            (client)
#      PATIENT_IDS           (client)
#      REDIS_URL             (optional — enables live dashboard banner)
#      FHIR_SERVER_URL       (optional — server auto-pushes)

# 3. Start the inference server
uvicorn inference_server.server:app --host 0.0.0.0 --port 8001

# 4. In another terminal, start the caregiver client
python -m caregiver_client.client
```

Verify:
```bash
curl http://localhost:8001/health
curl http://localhost:8001/model/info        # check uses_barometer
curl http://localhost:8002/api/patients      # caregiver API
# Open the dashboard:
#   http://localhost:8002/
```

---

## Inference Server API

**`POST /predict`** — header: `X-API-Key: <your key>`

```json
{
  "patient_id":              "charite-patient-007",
  "device_id":               "smarko-wearable-42",
  "acc_x":                   [-512, -498, ...],
  "acc_y":                   [128, 134, ...],
  "acc_z":                   [16300, 16280, ...],
  "timestamps_ms":           [1712345678000, 1712345678040, ...],
  "pressure":                [101325.0, 101322.5, ...],
  "pressure_timestamps_ms":  [1712345678000, 1712345678040, ...]
}
```

The response includes a complete FHIR R4 Observation in `fhir_observation`
and a `fhir_pushed` boolean indicating whether the server forwarded it to
`FHIR_SERVER_URL`. When `fall_detected=true` the server **also publishes**
the same payload to the Redis channel `fall_events`. Caregiver clients
subscribed to this channel receive the event in real time.

---

## Caregiver Client API (port 8002)

| Method | Path                              | Purpose                                  |
|--------|-----------------------------------|------------------------------------------|
| GET    | `/`                               | Dashboard HTML/JS                        |
| GET    | `/api/patients`                   | Patient list with fall counts            |
| GET    | `/api/falls?patient_id=&limit=`   | Fall history rows                        |
| POST   | `/api/falls/{id}/confirm?confirmed=yes\|no\|not_answered` | Manually set patient_confirmed |
| GET    | `/api/stream`                     | Server-Sent Events feed of live falls    |

The dashboard:
- **Patients tab** — one card per registered patient, fall count badge,
  red border on patients who have fallen.
- **Fall History tab** — sortable/filterable table with confirm-yes/no buttons.
- **Alert banner** — red flashing banner the instant a fall arrives over SSE
  (driven by the inference server's Redis publish).
- **Live status chip** — shows whether the SSE channel is connected.

---

## `fall_history` table — schema

Created automatically on first run by `caregiver_client/db.py`.

| Column              | Type           | Notes                                       |
|---------------------|----------------|---------------------------------------------|
| `id`                | int PK         |                                             |
| `patient_id`        | string, indexed| FHIR Patient identifier                     |
| `fall_detection`    | boolean        | True if the model flagged a fall            |
| `patient_confirmed` | string         | `yes` / `no` / `not_answered` (default)     |
| `detection_time`    | datetime       | Indexed; defaults to row insertion time     |

The legacy `participant_session` table from the full system is also reused —
it tracks the active recording session per patient and increments `fall_count`.

---

## Configuration cheatsheet (which side reads what)

| Variable                  | SERVER | CLIENT | Notes |
|---------------------------|:------:|:------:|-------|
| `MODEL_VERSION`           |   X    |        | Fixed model on the server |
| `ACC_SENSOR_TYPE`         |   X    |   X    | Both sides need to agree on the hardware |
| `HARDWARE_ACC_SAMPLE_RATE`|   X    |   X    | Same here |
| `RESAMPLING_METHOD`       |   X    |        | Server-only — server resamples |
| `INFLUXDB_*`              |        |   X    | Only the client queries Influx |
| `PATIENT_IDS`             |        |   X    | Which patients to poll |
| `INFLUX_PATIENT_TAG`      |        |   X    | Tag used to filter Influx by patient |
| `INFERENCE_SERVER_URL`    |        |   X    | Where the client POSTs |
| `INFERENCE_API_KEY`       |        |   X    | Must match `API_KEYS` |
| `API_KEYS`                |   X    |        | Accepted X-API-Key values |
| `DATABASE_URL`            |        |   X    | Caregiver DB (SQLite by default) |
| `REDIS_URL`               |   X    |   X    | Server publishes, client subscribes |
| `REDIS_FALL_CHANNEL`      |   X    |   X    | Default `fall_events` |
| `FHIR_SERVER_URL`         |   X    |        | Optional — server auto-pushes |
| `CAREGIVER_PORT`          |        |   X    | Default 8002 |
| `SERVER_PORT`             |   X    |        | Default 8001 |

---

## What is NOT included

Compared to the full system on the `complete_system` branch:
- No model switching at runtime (restart required)
- No Prometheus / Grafana
- No patient feedback popup / 12-second emergency timer
- No emergency tablet UI
- No Docker Compose (intentionally skipped — run with two terminals)
- No JWT auth / login screen on the dashboard

---

## Development notes

- The poller and web UI run **in the same Python process** so the SSE broker
  can directly receive events from the poller as well as from Redis. This is
  intentional simplicity for the integration build — split into two services
  later if you scale beyond one caregiver machine.
- SQLite is the default DB for zero-setup local testing. Switch to Postgres
  in production by changing `DATABASE_URL` only — no code change needed.
- The Redis path is **optional**. If `REDIS_URL` is empty:
  - Server skips the publish call (no error)
  - Client's SSE channel sends keepalives only
  - Fall history is still written when `/predict` returns
  - The dashboard's table still updates every 15 s
