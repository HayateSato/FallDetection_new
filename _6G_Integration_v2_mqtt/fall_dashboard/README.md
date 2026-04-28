# fall_dashboard — caregiver-facing fall events backend

Backend service that converts MQTT fall alerts into a durable record + live SSE
feed. Drives the **Fall Dashboard** view inside the caregiver web app.

| | |
|---|---|
| Port | `8002` |
| Run | `python -m fall_dashboard.main` |
| Audience | caregivers (read-only via `/api/*`) |
| Production target | runs in `mcs-fall-detection` namespace (StatefulSet behind Service) |

---

## What it does

```
[MQTT broker] ──fall/alert/<patient_id>──► fall_dashboard ──► Postgres fall_history
                                                          │
                                                          └──► SSE fan-out to caregiver browsers
                                                               (only when caregiver action needed)
```

1. Subscribes to `fall/alert/#` on the MQTT broker
2. On each alert: writes one row to `fall_history` in Postgres
3. **Conditionally** fans out via SSE only when:
   - `patient_confirmed == "not_answered"` (patient could not respond — assume serious)
   - `patient_confirmed == "yes"` AND `needs_help == True` (confirmed + asked for help)
4. False positives (`"no"`) and "patient confirmed but okay" (`"yes" + needs_help=false`)
   are **stored but NOT alerted** — kept for retraining, hidden from the live banner

This conditional fan-out matches the server-side filter in `main.py::_on_fall_mqtt`.

---

## REST + SSE API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/patients` | list of registered patients with lifetime fall counts |
| GET | `/api/falls?patient_id=&only_falls=&limit=` | fall history rows (default `only_falls=true`, `limit=200`, max 2000) |
| GET | `/api/stream` | Server-Sent Events — one event per *actionable* confirmed fall |
| GET | `/` | local-test HTML dashboard (replaced by Isa's web app in production) |

CORS is wide-open (`*`) so the caregiver web app in the FOCUS namespace can call
this cross-namespace without setup. Auth is NOT enforced — defer to ingress.

See `handover_docs/ISA_web_app_dashboards.md` for the integration contract Isa
uses to consume these endpoints from the Patient Dashboard.

---

## Files

```
fall_dashboard/
├── main.py             ← uvicorn entry + MQTT subscriber wiring
├── web.py              ← FastAPI app: /api/patients, /api/falls, /api/stream
├── mqtt_listener.py    ← FallEventBroker — paho thread → asyncio bridge
├── db.py               ← record_fall, list_patients, list_falls helpers
├── dashboard/          ← local-test HTML/JS (index.html, app.js, style.css)
├── inference_client.py ← LEGACY (Redis-era) — unused in MQTT flow
├── influx_poller.py    ← LEGACY (Redis-era) — unused in MQTT flow
├── Dockerfile
└── requirements.txt
```

The two `LEGACY` files are kept for git history; safe to delete in a future
cleanup pass once nothing references them externally.

---

## Configuration (read from `.env` via `dotenv` in `main.py`)

| Var | Default | Purpose |
|-----|---------|---------|
| `CAREGIVER_HOST` | `0.0.0.0` | bind address |
| `CAREGIVER_PORT` | `8002` | HTTP port |
| `DATABASE_URL` | `postgresql+psycopg2://fall_user:fall_pass@localhost:5432/fall_detection` | shared Postgres |
| `MQTT_BROKER_HOST` | (required) | broker hostname |
| `MQTT_BROKER_PORT` | `1883` | broker port |
| `MQTT_ALERT_TOPIC` | `fall/alert` | subscribed as `<topic>/#` |
| `PATIENT_IDS` | (required) | comma-separated; pre-creates participant_session rows |
| `MAC_IDS` | optional | positional 1:1 with `PATIENT_IDS` |

---

## Production notes

- Single replica only — SSE fan-out state is in-process, multiple replicas would
  silently drop events to subscribers connected to other instances. If scaling
  is needed, use a Redis pub/sub (or NATS) bridge instead.
- Schema is managed by Alembic — run migrations before first deploy:
  `alembic upgrade head` (from `_6G_Integration_v2_mqtt/` as cwd).
- `auto_confirm_timer` code was removed (2026-04-27) — patient confirmation now
  lives in the mobile app / mock_app, not here.
