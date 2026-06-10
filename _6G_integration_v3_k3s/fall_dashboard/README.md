# fall_dashboard — caregiver-facing fall events backend

Backend service that converts MQTT fall alerts into a live SSE feed and REST API.
Drives the **Fall Dashboard** view inside the caregiver web app.

| | |
|---|---|
| Port | `8002` |
| Run | `python -m fall_dashboard.main` |
| Audience | caregivers (read-only via `/api/*`) |
| Production target | runs in FOCUS's environment (k3s / Docker) |

**No local database.** Patient list comes from `PATIENT_IDS` env var.
Fall history and fall counts are queried from InfluxDB.

---

## What it does

```
[MQTT broker] ──fall/possible/<patient_id>──► fall_dashboard ──► SSE fan-out ("Possible fall" badge on patient card)
[MQTT broker] ──fall/alert/<patient_id>─────► fall_dashboard ──► SSE fan-out (red alert banner, if action needed)
fall_dashboard ─────────────────────────────► InfluxDB        ──► /api/falls, /api/patients fall counts
```

1. Subscribes to `fall/possible/#` — published immediately on fall detection (before patient confirms)
2. Subscribes to `fall/alert/#` — published after patient confirmation or 10s timeout
3. **Conditionally** fans the confirmed alert via SSE only when:
   - `patient_confirmed == -1` ("not_answered") — patient could not respond, treat as serious
   - `patient_confirmed == 1` ("yes") AND `needs_help == True` — confirmed + asked for help
4. False positives (`0` = "no") and "patient okay" (`1` + needs_help=false) are **not alerted**
   — visible only in `/api/falls` history

### patient_confirmed int encoding (InfluxDB + SSE events)
```
 1  = patient confirmed it was a fall  ("yes")
 0  = patient denied (false positive)  ("no")
-1  = no response within timeout       ("not_answered")
```
Postgres / MQTT payloads carry the string form. The conversion to int happens in
`main.py::_on_fall_mqtt` (single conversion point for the caregiver layer).

---

## REST + SSE API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/patients` | list of patients from `PATIENT_IDS` env var + fall counts from InfluxDB |
| GET | `/api/falls?patient_id=&only_falls=&limit=` | fall history from InfluxDB (default `only_falls=true`, `limit=200`) |
| GET | `/api/stream` | Server-Sent Events — pending and confirmed fall events |
| GET | `/` | local-test HTML dashboard (replaced by FOCUS Flutter app in production) |

CORS is wide-open (`*`) so the caregiver web app can call this cross-origin.
Auth is NOT enforced here — defer to ingress / reverse proxy.

---

## Files

```
fall_dashboard/
├── main.py             <- uvicorn entry + MQTT subscriber wiring + patient_confirmed normalisation
├── web.py              <- FastAPI app: /api/patients, /api/falls, /api/stream
├── mqtt_listener.py    <- FallEventBroker — paho thread -> asyncio bridge; subscribes to both topics
├── db.py               <- list_patients (PATIENT_IDS + InfluxDB counts), list_falls (InfluxDB)
├── dashboard/          <- local-test HTML/JS (index.html, app.js, style.css)
├── Dockerfile
└── requirements.txt
```

---

## Configuration (read from `.env` via `dotenv` in `main.py`)

| Var | Default | Purpose |
|-----|---------|---------|
| `CAREGIVER_HOST` | `0.0.0.0` | bind address |
| `CAREGIVER_PORT` | `8002` | HTTP port |
| `MQTT_BROKER_HOST` | (required) | broker hostname |
| `MQTT_BROKER_PORT` | `1883` | broker port |
| `MQTT_ALERT_TOPIC` | `fall/alert` | confirmed alerts; subscribed as `<topic>/#` |
| `MQTT_POSSIBLE_TOPIC` | `fall/possible` | pre-confirmation alerts; subscribed as `<topic>/#` |
| `INFLUXDB_URL` | (required) | InfluxDB v2 endpoint |
| `INFLUXDB_TOKEN` | (required) | InfluxDB auth token |
| `INFLUXDB_ORG` | (required) | InfluxDB organisation |
| `INFLUXDB_FALL_EVENTS_BUCKET` | (or `INFLUXDB_BUCKET`) | bucket for fall_events measurement |
| `PATIENT_IDS` | (required) | comma-separated patient IDs — defines the patient list |
| `MAC_IDS` | optional | positional 1:1 with `PATIENT_IDS` — added to API responses |

---

## Production notes

- Single replica only — SSE fan-out state is in-process. Multiple replicas would
  silently drop events to subscribers on other instances. If scaling is needed,
  use a Redis pub/sub bridge instead.
- No Alembic, no database migrations. No Postgres dependency at all.
