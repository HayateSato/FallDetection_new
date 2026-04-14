# Q&A — Architecture & Concepts
*Companion file: `_6G_Integration_v2/_docs/Q&Amd` covers operational topics (Redis setup, DB config, how to run, patient IDs, patient feedback flow)*

---

## Topic 1 — Terminology: Poll vs Query vs GET vs Push vs Subscribe

**Poll** means repeatedly calling something on a timer — "check every 10 seconds, check again, check again, forever."

It is not a different protocol or verb. A poll *is* usually a GET request or a query under the hood. The distinction is about **who initiates it and how often**:

| Term | What it means | Who initiates | How often |
|------|--------------|---------------|-----------|
| **Query** | A single request for data | Your code | Once, on demand |
| **GET request** | An HTTP verb — how the request is made | Your code | Once, on demand |
| **Poll** | Repeatedly querying on a timer | Your code | Every N seconds, forever |
| **Push / webhook** | The server contacts you when something happens | The other side | When the event occurs |
| **Subscribe (SSE/WebSocket)** | You open one connection and the server streams events to you | Your code (once) | Connection stays open |

In our pipeline:

```
Our influx_poller.py
  → every 10 seconds: runs a Flux query against InfluxDB  ← this is polling
  → sends result to inference server via POST /predict      ← this is a one-off HTTP request
```

**Why polling vs push?**
InfluxDB doesn't push data to you when new sensor readings arrive — it just stores them. So the only way to get new data is to keep asking. That's why we poll. In contrast, our Redis → dashboard alert is "push": we publish once when a fall happens, and the dashboard receives it immediately without having to keep asking.

---

## Topic 2 — Browser Update Methods: WebSocket vs SSE vs Polling, and Where Redis Fits

These are all ways to get real-time updates to a browser, but they work differently:

|  | **Polling** | **SSE (Server-Sent Events)** | **WebSocket** |
|--|-------------|------------------------------|---------------|
| Who initiates | Browser asks every N seconds | Browser opens once, server pushes | Both sides can send any time |
| Direction | One way (browser → server → browser) | One way (server → browser only) | Two way |
| Connection | Opens and closes each time | Stays open | Stays open |
| Complexity | Simple | Simple | More complex |
| Good for | Basic dashboards | Live feeds (alerts, logs) | Chat, games |

**Redis is not a browser technology.** It lives entirely on the server side and is used to move events *between backend services*. The flow is:

```
Inference server ──Redis publish──► Caregiver backend ──SSE──► Browser
```

Redis is never exposed to the browser. The browser only speaks HTTP/SSE/WebSocket to the caregiver backend. Redis is the internal pipe between two Python processes.

Redis is used because the inference server and caregiver client are **separate processes** with no shared memory — Redis is how they talk. Smaller systems skip it and just have the dashboard poll the DB directly every few seconds.

---

## Topic 3 — Where the Redis Publisher Lives (and Why It Has Nothing to Do with nginx)

The Redis publisher lives **directly inside `inference_server/server.py`** as a Python client object (`_redis_client`). On startup, the server creates an `aioredis` connection to Redis. Inside the `/predict` handler, it calls:

```python
await _redis_client.publish("fall_events", json.dumps(payload))
```

This is a **direct TCP connection** from the inference server process to the Redis process. No HTTP, no nginx, no intermediary. Think of it like a database connection — your Python code opens a socket to Redis and writes to it directly.

```
inference server process
    │
    └── _redis_client (aioredis connection pool)
              │  TCP :6379 (Redis protocol, NOT HTTP)
              ▼
           Redis process
```

Nginx only proxies HTTP traffic. Redis runs on its own TCP protocol (port 6379). They never interact. Nginx would only be relevant if you were trying to expose Redis to the internet — which you would never do, since Redis has no authentication by default.

In development all three run on the same machine. In production they could be on separate machines. The caregiver client (poller + web backend + Redis subscriber) all run as one process — `web.py` serves the dashboard HTML/JS AND handles the SSE stream. The browser is just the frontend that renders it.

```
[Inference Server]          [Redis]         [Caregiver Client + Dashboard Backend]
  server.py  ──publish──►  Redis  ◄──subscribe──  redis_listener.py
                                                           │
                                                      web.py (FastAPI)
                                                           │
                                                     browser (dashboard)
```

---

## Topic 4 — Why Not Give the Partner Direct Access to Our Redis

Four reasons:

1. **Security** — Redis has no authentication by default. Exposing it externally means anyone on the network can read all events.
2. **Coupling** — If they connect directly to our Redis, they depend on our exact channel names, message format, and Redis version. If we change anything, their code breaks.
3. **Their stack may already have a message broker** — If they use RabbitMQ, Kafka, or MQTT, they don't want another one.
4. **SSE is HTTP** — It works through proxies, firewalls, and load balancers with no special configuration. Redis connections don't.

The conventional pattern is: internal systems talk Redis, external consumers get HTTP/SSE. SSE is the "safe, public face" of the same events.

---

## Topic 5 — FHIR: "Push to Server" vs "Return in Response"

These are two different destinations, not two ways of doing the same thing.

- **Return in API response**: The FHIR Observation is in the JSON body of the `POST /predict` response. The caller (our caregiver client) receives it. It goes nowhere else unless the caller does something with it.
- **Push to FHIR server**: Our inference server makes an *additional* HTTP POST to `https://their-fhir-server.de/fhir/Observation`. This is a second outbound call that happens inside the server, invisible to the original caller.

```
Caregiver client ──POST /predict──► Inference Server
                ◄──response (FHIR)─                │
                                                    └──POST Observation──► FHIR Server (their DB)
```

Both can be active at the same time. Controlled by whether `FHIR_SERVER_URL` is set in `.env`.

---

## Topic 6 — The Actual FHIR Output (what it looks like)

The FHIR Observation is returned in the `fhir_observation` field of the `/predict` response. Source code: `_6G_Integration_v2/fhir_converter.py`. Example for a fall with confidence 0.994:

```json
{
  "resourceType": "Observation",
  "id": "some-uuid-here",
  "status": "final",
  "category": [{ "coding": [{ "system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "activity", "display": "Activity" }] }],
  "code": {
    "coding": [{ "system": "http://snomed.info/sct", "code": "217082002", "display": "Fall (event)" }],
    "text": "Fall detection"
  },
  "subject": { "reference": "Patient/test_patient-001", "display": "test_patient-001" },
  "effectiveDateTime": "2026-04-10T10:00:00+00:00",
  "issued": "2026-04-10T10:00:00+00:00",
  "valueBoolean": true,
  "component": [
    {
      "code": { "coding": [{ "system": "http://loinc.org", "code": "72514-3", "display": "Fall risk assessment score" }], "text": "Fall confidence score" },
      "valueQuantity": { "value": 0.994, "unit": "probability", "system": "http://unitsofmeasure.org", "code": "1" }
    },
    {
      "code": { "coding": [{ "system": "http://snomed.info/sct", "code": "246075003", "display": "Causative agent" }], "text": "Detection algorithm version" },
      "valueString": "SmarKo-FallDetection-v0"
    }
  ],
  "device": { "reference": "Device/6c:1d:eb:04:a9:e6", "display": "6c:1d:eb:04:a9:e6" }
}
```

**Caveat:** `72514-3` is officially "Fall risk assessment score" — a score measured over time, not a per-event confidence. This is technically a code reuse. If their FHIR validator is strict, they might flag it. Worth checking with their FHIR team.

---

## Topic 7 — Inference Server Internal Components

The entire inference server is **one file**: `inference_server/server.py`. There is no separate "inference API component" and "inference engine component" as separate files. `PipelineSelector` (the engine) is just a Python object that lives inside the same process, called like any other function.

```
inference_server/server.py
│
├── LOADED AT STARTUP (module-level, before any request)
│     ├── PipelineSelector (_engine)     ← loads XGBoost .pkl from disk
│     ├── _redis_client                  ← aioredis connection (if REDIS_URL set)
│     └── config values from .env        ← model version, sensor type, etc.
│
├── ON EVERY POST /predict REQUEST
│     ├── verify_api_key()               ← auth + rate limiting
│     ├── AccelerometerResampler         ← 25Hz → 50Hz
│     ├── convert_lsb_to_g()            ← LSB integers → g units
│     ├── convert_acc_nparray_to_df()   ← numpy → pandas DataFrame
│     ├── compose_detection_window()    ← take last 9s = 450 samples
│     ├── _engine.predict()             ← XGBoost inference → {is_fall, confidence}
│     ├── build_fhir_observation()      ← wrap result in FHIR R4 JSON
│     ├── _push_to_fhir_server()        ← optional HTTP POST to partner FHIR server
│     ├── _publish_fall_event()         ← optional Redis PUBLISH if fall
│     └── return PredictResponse        ← FHIR + inference result to caller
│
└── OTHER ENDPOINTS
      ├── GET /health                   ← uptime, model version
      ├── GET /model/info               ← uses_barometer, feature count
      └── GET /docs                     ← Swagger UI (auto-generated by FastAPI)
```

**No Postgres in this codebase.** The tables `inference_log`, `feature_snapshot`, `api_request_log` exist only in the full system (`system_operator/ml_server/` on the `complete_system` branch). In `_6G_Integration_v2`, the only database is in `caregiver_client/db.py` — SQLite, two tables: `FallHistory` + `ParticipantSession`, written by the caregiver client (the poller), not the inference server.

---

## Topic 8 — How the Caregiver Dashboard Displays Fall History

The browser never talks to the database directly. It only talks to `web.py` via HTTP. `web.py` is the only thing that reads the database. This is standard web architecture — the database is always behind the backend, never exposed to the browser.

Full chain:

```
Browser (app.js)
    │
    │  every 15s: fetch('/api/falls?only_falls=true&limit=500')
    │  (plain HTTP GET — browser's fetch() API)
    ▼
web.py  GET /api/falls endpoint
    │
    │  calls cdb.list_falls(...)
    ▼
db.py  list_falls()
    │
    │  SQLAlchemy SELECT from fall_history table
    ▼
SQLite file (caregiver.db on disk)
    │
    └──► returns list of dicts
              │
              ▼
         web.py adds mac_id lookup from mac_map
              │
              ▼
         returns JSON: {"falls": [...]}
              │
              ▼
         app.js receives JSON → builds HTML table rows → inserts into DOM
```

---

## Topic 9 — Deployment: Docker, OS, Networking, Adding Patients

### Does OS matter when providing a docker-compose.yml?

**OS:** Mostly no. Docker abstracts the OS — containers run the same on Ubuntu, Debian, or RHEL. The only exception is **CPU architecture**: if they run ARM servers (AWS Graviton, Apple Silicon), the Docker image must be built for ARM or as a multi-arch image. Worth asking "x86 or ARM?" but it's not a showstopper.

**Networking:** You still need to care — but differently. Docker Compose creates an internal network between your containers automatically. The question is whether your containers can *reach their InfluxDB*, which lives outside your Compose network:

- Same machine as our containers → use `host.docker.internal` (Linux quirk: needs `--add-host`)
- Different server on the same LAN → usually works fine
- Behind a VPN or firewall → their IT needs to whitelist outbound traffic from our container

You don't configure this in docker-compose — it's their infrastructure. But you need to know the answer to set `INFLUXDB_URL` correctly. So yes, it's still your question to ask, even if it's their problem to solve.

### Adding more patients — do you need duplicate folders or containers?

No. Just add more entries to `PATIENT_IDS` and `MAC_IDS` in the same `.env`:

```env
PATIENT_IDS=Patient/001,Patient/002,Patient/003
MAC_IDS=aa:bb:cc:dd:ee:ff,11:22:33:44:55:66,77:88:99:aa:bb:cc
```

The poller loops over the list. Each polling cycle processes every patient sequentially — patient 10 waits for patients 1–9 to finish. For a small medical trial (~20 patients at 10-second polling) this is fine. For 100+ patients you would want parallel workers, but that's a future problem.

No duplicate folders, no duplicate Docker containers.
