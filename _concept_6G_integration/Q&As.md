**Poll** means repeatedly calling something on a timer — "check every 10 seconds, check again, check again, forever."

It's not a different protocol or verb. A poll *is* usually a GET request or a query under the hood. The distinction is about **who initiates it and how often**:

| Term | What it means | Who initiates | How often |
| --- | --- | --- | --- |
| **Query** | A single request for data | Your code | Once, on demand |
| **GET request** | An HTTP verb — how the request is made | Your code | Once, on demand |
| **Poll** | Repeatedly querying on a timer | Your code | Every N seconds, forever |
| **Push / webhook** | The server contacts you when something happens | The other side | When the event occurs |
| **Subscribe (SSE/WebSocket)** | You open one connection and the server streams events to you | Your code (once) | Connection stays open |

So in the pipeline:

`Our influx_poller.py
  → every 10 seconds: runs a Flux query against InfluxDB  ← this is polling
  → sends result to inference server via POST /predict      ← this is a one-off HTTP request`

The word "poll" just emphasizes the repetitive timer aspect — that the code wakes up, asks "any new data?", processes it, sleeps, wakes up again.

**Why polling vs push?**

InfluxDB doesn't push data to you when new sensor readings arrive — it just stores them. So the only way to get new data is to keep asking. That's why we poll. In contrast, our Redis → dashboard alert is "push": we publish once when a fall happens, and the dashboard receives it immediately without having to keep asking.

---

Where does the Redis publisher live?**

The Redis publisher lives in the **inference server**. When a fall is detected, `inference_server/server.py` calls `redis.publish(...)` before sending the HTTP response back. Redis itself is a separate process (a Docker container, or `docker run redis`), but the *act of publishing* happens in the inference server code.

The caregiver client has the **subscriber** — it listens to Redis and, when a message arrives, sends it to connected browser tabs via SSE.

So the machines are:

```
[Inference Server machine]          [Redis machine]         [Caregiver Client machine]
  inference_server/server.py  ──publish──►  Redis  ◄──subscribe──  caregiver_client/redis_listener.py
                                                                              │
                                                                     caregiver_client/web.py
                                                                              │
                                                                    browser (dashboard)`
```

In development (your laptop) all three run on the same machine. In production they could be separate. The caregiver client and caregiver dashboard are **the same machine** — `web.py` is the backend that serves the dashboard HTML/JS AND handles the SSE stream. The browser is just the frontend that renders it.

---

**"Push to FHIR server" vs "return in API response" — what's the difference?**

These are two different destinations, not two ways of doing the same thing.

- **Return in API response**: The FHIR Observation is included in the JSON body of the `POST /predict` response. The *caller* (our caregiver client) receives it. It never goes anywhere else unless the caller does something with it.
- **Push to FHIR server**: Separately, our inference server makes an additional HTTP POST to `https://their-fhir-server.de/fhir/Observation`. This is a second outbound call that happens inside the server, invisible to the original caller.
```
Caregiver client ──POST /predict──► Inference Server
                ◄──response (FHIR)─                │
                                                    └──POST Observation──► FHIR Server (their DB)
```

If they only want to receive the FHIR in the response and store it themselves — Option 1. If they want our server to auto-register every fall into their FHIR server directly — Option 2. Both can be active at once. This is controlled by whether `FHIR_SERVER_URL` is set in `.env`.

---

**WebSocket vs SSE vs Polling — and where does Redis fit?**

These are all ways to get real-time updates to a browser, but they work differently:

|  | **Polling** | **SSE (Server-Sent Events)** | **WebSocket** |
| --- | --- | --- | --- |
| Who initiates | Browser asks every N seconds | Browser opens once, server pushes | Both sides can send any time |
| Direction | One way (browser → server → browser) | One way (server → browser only) | Two way |
| Connection | Opens and closes each time | Stays open | Stays open |
| Complexity | Simple | Simple | More complex |
| Good for | Basic dashboards | Live feeds (alerts, logs) | Chat, games |

Redis is **not** a browser technology — it lives entirely on the server side. It is used to move events *between backend services*. The flow is:

`Inference server ──Redis publish──► Caregiver backend ──SSE──► Browser`

Redis is never exposed to the browser. The browser only speaks HTTP/SSE/WebSocket to the caregiver backend. Redis is the internal pipe between two Python processes.

Is Redis common in business? Yes, but mostly in larger systems. Smaller systems skip it and just have the dashboard poll the DB directly every few seconds. We use Redis because the inference server and caregiver client are **separate processes** with no shared memory — Redis is how they talk.

---

**Why not give FOCUS direct access to our Redis?**

A few reasons:

1. **Security** — Redis has no authentication by default. Exposing it externally means anyone on the network can read all events.
2. **Coupling** — If they connect directly to our Redis, they depend on our exact channel names, message format, and Redis version. If we change anything, their code breaks.
3. **Their stack may already have a message broker** — If they use RabbitMQ or Kafka, they don't want another one.
4. **SSE is HTTP** — It works through proxies, firewalls, and load balancers with no special configuration. Redis connections don't.

The conventional pattern is: internal systems talk Redis, external consumers get HTTP/SSE. SSE is the "safe, public face" of the same events.

---

**If I give them a docker-compose.yml, do I still need to care about OS and networking?**

**OS:** Mostly no. Docker abstracts the OS — your containers run the same on Ubuntu, Debian, or RHEL. The only exception is CPU architecture: if they run ARM servers (AWS Graviton, Apple Silicon), your Docker image must be built for ARM or as a multi-arch image. Worth asking "x86 or ARM?" but it's not a showstopper.

**Networking:** Yes, you still need to care — but in a different way. Docker Compose creates an internal network between your containers automatically. The question is whether your containers can *reach their InfluxDB*, which lives outside your Compose network. This depends on:

- Is their InfluxDB on the same machine? → Use `host.docker.internal` (Linux quirk: needs `-add-host`)
- Is it on a different server on the same LAN? → Usually works fine
- Is it behind a VPN or firewall? → Their IT needs to whitelist your container's outbound traffic

You don't configure this in your docker-compose — it's their infrastructure. But you need to *know* the answer to configure `INFLUXDB_URL` correctly. So yes, it's still your question to ask, even if it's their problem to solve.

---

**Hosting on their machine/cloud — does this answer some questions?**

Yes. I'll update the document now to remove the ngrok/Cloudflare question and reframe the deployment section.

---

**Adding more patients — do you need duplicate folders?**

No. Just add more entries to `PATIENT_IDS` and `MAC_IDS` in the same `.env`:

`PATIENT_IDS=Patient/001,Patient/002,Patient/003
MAC_IDS=aa:bb:cc:dd:ee:ff,11:22:33:44:55:66,77:88:99:aa:bb:cc`

The poller loops over the list. Each polling cycle it processes every patient sequentially. The bottleneck at scale is that patients are processed **one after another** in the same thread — patient 10 waits for patients 1–9 to finish. For a small medical trial (say, 10–20 patients with 10-second polling) this is fine. For 100+ patients you would want parallel workers, but that's a future problem — the single `.env` list is the right design for now.

No duplicate folders, no duplicate Docker containers.

---

**Where can you see the actual FHIR output?**

The easiest way: run the inference server and call it with `curl.exe`. The FHIR Observation is returned in the `fhir_observation` field of the response. You can also look directly at the code — the exact output structure is in _6G_Integration_v2/fhir_converter.py. Here is exactly what gets returned for a fall with confidence 0.994:

```
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

Note: `72514-3` is officially "Fall risk assessment score" — which is a score measured over time, not a per-event confidence. This is technically a code reuse. If their FHIR validator is strict, they might flag it. A more accurate code would be something custom or a `valueDecimal` under a locally defined extension. Worth checking with their FHIR team.

--- 

**Redis publisher location:** Lives in the inference server. Redis itself is a separate process. The caregiver client (poller + web backend + dashboard) all run together on one machine. The browser is just the frontend — it never talks to Redis directly.

**FHIR "push" vs "return in response":** Two different destinations. "In response" = caregiver client gets it. "Push" = inference server makes a second HTTP call directly to their FHIR database, independent of the caregiver client. Like the difference between handing someone a document vs mailing a copy directly to the filing office yourself.

**WebSocket vs SSE vs Polling:** All browser-update mechanisms, not competing with Redis — Redis is only server-to-server. SSE is what we expose to browsers. Polling is the simplest fallback. WebSocket is bidirectional (more complexity, not needed here).

**Why not expose Redis directly:** Security, coupling, firewall unfriendliness. SSE is just HTTP — it works everywhere without extra ports or configuration on their side.

**OS / networking with docker-compose:** OS doesn't matter (Docker abstracts it), but CPU architecture (x86 vs ARM) still matters for the Docker image. Networking is their infrastructure problem, but you need to ask because the answer determines what goes in `INFLUXDB_URL`.

**Hosting on their machine:** Removed the ngrok/Cloudflare question from the document — that was only relevant for external hosting. Updated to focus on internal network layout instead.

**Adding patients:** One `.env` entry per patient, comma-separated. No duplicate folders or containers. The same loop handles all of them.

**FHIR output:** Added the actual JSON output to the document so you can show it directly in the meeting. Also flagged the LOINC code reuse as a potential validator issue to raise proactively.

--- 

**Question 1 — How does the inference server talk to Redis? Does it go through nginx?**

No nginx involved at all for Redis. Redis is not HTTP — it uses its own TCP protocol on port 6379. Nginx only proxies HTTP traffic.

The Redis publisher is **directly inside `server.py`** as a Python client object (`_redis_client`). Look at lines 112 and 295–310: on startup, the server creates an `aioredis` connection to Redis. Then inside the `/predict` handler at line 498, it just calls:

`await _redis_client.publish("fall_events", json.dumps(payload))`

This is a direct TCP connection from the inference server process to the Redis process. No HTTP, no nginx, no intermediary. Think of it like a database connection — your Python code opens a socket to Redis and writes to it directly.

`inference server process
    │
    └── _redis_client (aioredis connection pool)
              │  TCP :6379 (Redis protocol, NOT HTTP)
              ▼
           Redis process`

Nginx would only be relevant if you needed to expose Redis to the internet through a reverse proxy — which you would never do (Redis has no authentication by default and should never be exposed publicly).

---

**Question 2 — Who writes to Postgres in this inference server? What about inference_log, feature_snapshot, api_request_log?**

Those tables **do not exist in this codebase** (`_6G_Integration_v2`). Look at the server.py you just read — there is zero Postgres code in it. No `db_writer`, no SQLAlchemy imports, no `DATABASE_URL`.

Those three tables are from the **full system** (`system_operator/ml_server/server.py` on the `complete_system` branch), which has a `db_writer.py` service that writes to Postgres as a FastAPI `BackgroundTask` after every `/predict` call.

In `_6G_Integration_v2`, the **only database** is:

- `caregiver_client/db.py` — SQLite, two tables: `FallHistory` + `ParticipantSession`
- Written by the **caregiver client** (the poller), not the inference server

The inference server in this integration has no database at all.

---

**Question 3 — Your mental model of the inference server components**

Based on the actual code, here is the accurate picture:

`inference_server/server.py  (the entire inference server is one file)
│
├── LOADED AT STARTUP (module-level, before any request arrives)
│     ├── PipelineSelector(_engine)       ← loads XGBoost .pkl from disk
│     ├── _redis_client                   ← aioredis connection (if REDIS_URL set)
│     └── config values from .env         ← model version, sensor type, etc.
│
├── ON EVERY POST /predict REQUEST
│     ├── verify_api_key()                ← auth + rate limiting
│     ├── AccelerometerResampler          ← 25Hz → 50Hz
│     ├── convert_lsb_to_g()             ← LSB integers → g units
│     ├── convert_acc_nparray_to_df()    ← numpy → pandas DataFrame
│     ├── compose_detection_window()     ← take last 9s = 450 samples
│     ├── _engine.predict()              ← XGBoost inference → {is_fall, confidence}
│     ├── build_fhir_observation()       ← wrap result in FHIR R4 JSON
│     ├── _push_to_fhir_server()         ← optional HTTP POST to partner FHIR server
│     ├── _publish_fall_event()          ← optional Redis PUBLISH if fall
│     └── return PredictResponse         ← FHIR + inference result to caller
│
└── OTHER ENDPOINTS
      ├── GET /health                    ← uptime, model version
      ├── GET /model/info                ← uses_barometer, feature count
      └── GET /docs                      ← Swagger UI (auto-generated by FastAPI)`

There is no separate "inference API component" and "inference engine component" as separate files talking to each other. `PipelineSelector` (the engine) is just a Python object that lives inside the same process. It gets called like any other function.

---

**Question 4 — How does the caregiver dashboard show fall history?**

Yes, your mental model is correct. Here is the exact chain:

`Browser (app.js)
    │
    │  every 15s: fetch('/api/falls?only_falls=true&limit=500')
    │  (plain HTTP GET — browser's fetch() API)
    ▼
web.py  GET /api/falls endpoint  (line 113 of web.py)
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
         app.js receives JSON → builds HTML table rows → inserts into DOM`

The browser never talks to the database directly. It only talks to `web.py` via HTTP. `web.py` is the only thing that reads the database. This is standard web architecture — the database is always behind the backend, never exposed to the browser.

