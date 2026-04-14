# Q&A — 6G Integration Technical Notes

---

## Topic 1 — Redis

### What is REDIS_URL?

It's a connection string, same idea as a database URL:

```
redis://hostname:port/db_number
```

| Scenario | REDIS_URL |
|----------|-----------|
| Redis running locally on your laptop | `redis://localhost:6379/0` |
| Redis inside Docker, port-forwarded | `redis://localhost:6379/0` (same — you talk to the forwarded port) |
| Redis inside Docker, accessed by other containers | `redis://redis:6379/0` (use the Docker service name) |
| Redis on a remote server | `redis://192.168.1.50:6379/0` |

For local setup, run Redis in a one-line Docker container and set:

```
docker run -d --name redis-local -p 6379:6379 redis:7
```

```env
REDIS_URL=redis://localhost:6379/0
```

---

### Why both the inference server and caregiver client need the same REDIS_URL

```
inference_server (PUBLISHER)  ──publishes to──►  Redis instance  ◄──subscribes from──  caregiver_client (SUBSCRIBER)
        │                                              ▲                                        │
        └── needs REDIS_URL                            │                    needs REDIS_URL ─────┘
                                                       │
                                            This is the Redis instance
                                            (just a message broker, like a mailbox)
```

- **Inference server** = publisher. Needs `REDIS_URL` to know where to send messages.
- **Caregiver client** = subscriber. Needs `REDIS_URL` to know where to listen.
- Both must point at the **same** Redis instance, otherwise the subscriber never sees the messages.

Redis is a **separate process** (its own server). Even if the publisher and subscriber run on the same machine, they both need the URL to connect to it. Think of Redis like a shared mailbox:

```
Publisher  ──puts letter in──►  Mailbox  ──subscriber picks up──►  Subscriber
```

If `REDIS_URL` is empty, the system still works — the inference server just skips publishing, and fall history still gets written via the HTTP response path. Redis is purely for **instant** live dashboard updates.

---

### If everything is in Docker Compose

```yaml
services:
  redis:
    image: redis:7
    ports:
      - "6379:6379"    # only needed if you want to access from outside Docker

  inference_server:
    environment:
      - REDIS_URL=redis://redis:6379/0    # "redis" = Docker service name

  caregiver_client:
    environment:
      - REDIS_URL=redis://redis:6379/0    # same Redis instance
```

---

### Is Redis pub/sub bidirectional?

No. Redis pub/sub is **one-directional**:

```
Publisher  →  Channel  →  Subscriber(s)
```

A subscriber cannot reply back through the same channel to the publisher. The relationship is fixed.

For patient feedback, Redis is not used at all for the return message. The patient sends feedback via a normal **HTTP POST** to the caregiver client. This is the standard pattern:

- Redis delivers the notification (one-way push from server to patient device)
- HTTP POST delivers the response (patient → server)

Two different transport mechanisms for two different directions. WebSocket would be the alternative if you wanted bidirectional on one connection, but HTTP POST is simpler and sufficient here since feedback is a single click.

---

### Can fall markers be injected into InfluxDB via Redis?

Yes, but Redis itself doesn't write to InfluxDB — you add a **subscriber that listens to `fall_events` and writes to InfluxDB**. Two options:

**Option A:** Add logic to the caregiver client's poller — after storing in SQLite, also write a point to InfluxDB (e.g. measurement `fall_event`, fields: `fall_detected=true`, `confidence=0.99`, tag: `macAddress=6c:1d:...`). The `influxdb-client` package is already installed.

**Option B (implemented):** A separate standalone script (`influx_marker_writer.py`) that subscribes to Redis `fall_events` and writes to InfluxDB. This is cleaner — it decouples "who detects falls" from "who records them."

Data flow:

```
inference_server → Redis fall_events → influx_marker_writer.py → InfluxDB write
```

---

## Topic 2 — Three Redis Subscribers and Full Fall Flow

When a fall is detected, the inference server publishes to Redis `fall_events`. Three separate subscribers react:

```
                    inference_server (:8001)
                          │
                    PUBLISH fall_events
                          │
                          ▼
                       Redis (:6379)
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
  Subscriber 1      Subscriber 2      Subscriber 3
  caregiver         influx_marker_    patient popup
  dashboard         writer.py         /patient/
  (alert banner)    (writes           (shows "Did you fall?"
                     fall_marker=1     → "Need help?")
                     to InfluxDB)           │
                                       Patient clicks YES/NO
                                            │
                                            ▼
                                  POST /api/falls/{id}/confirm
                                  (HTTP back to caregiver API)
                                            │
                                            ▼
                              fall_history.patient_confirmed = 'yes'/'no'
```

---

## Topic 3 — Patient Feedback Mechanism

### Full flow

```
Fall detected
  │
  ├──► Redis fall_events ──► Patient popup (/patient/)
  │                                │
  │                          "Did you fall?" (10s countdown)
  │                                │
  │                    ┌───────────┴───────────┐
  │                    ▼                       ▼
  │                YES clicked             NO clicked
  │                    │                       │
  │             "Need help?" (10s)        POST confirmed='no'
  │             ┌───────┴────────┐
  │             ▼                ▼
  │         YES clicked     NO/timeout
  │             │                │
  │     POST confirmed='yes' POST confirmed='yes'
  │
  └── POST /api/falls/{id}/confirm  ──►  caregiver API  ──►  DB update
```

### 10-second auto-confirm timer

If the patient doesn't respond within 10 seconds, something must write `patient_confirmed = 'not_answered'`. Two options:

- **Client-side timer:** The patient popup JS starts a 10s countdown. If it expires, the JS itself sends `POST /api/falls/{id}/confirm?confirmed=not_answered`. Simple, but fails if the browser crashes.
- **Server-side timer (implemented):** The caregiver server starts an asyncio timer when the fall is stored. If no `/confirm` arrives within 10s, it auto-writes `not_answered`. More reliable.

The server-side timer is implemented in `caregiver_client/web.py` as `start_auto_confirm_timer(fall_id)` / `cancel_auto_confirm_timer(fall_id)`.

---

## Topic 4 — Local Database

### What is actually stored in the local DB (vs FHIR)

The inference server returns a full response including a FHIR Observation:

```json
{
  "patient_id": "...",
  "inference": { "fall_detected": true, "confidence": 0.994, "model_version": "v0" },
  "fhir_observation": { "resourceType": "Observation", ... },
  "fhir_pushed": false
}
```

The poller extracts only these fields from that response:

```python
inference = result.get("inference", {})
is_fall   = bool(inference.get("fall_detected", False))
det_time  = result.get("timestamp")
```

What gets written to the local DB (`db.record_fall()`):

```python
FallHistory(
    patient_id        = patient_id,    # string
    fall_detection    = is_fall,       # boolean
    detection_time    = det_time_dt,   # datetime
    patient_confirmed = "not_answered"
)
```

The `fhir_observation` from the response is **completely ignored** when saving to DB. The local `FallHistory` table only stores four fields: who, yes/no fall, when, and patient confirmation status.

The FHIR Observation does travel further — it gets passed into the `on_fall` callback and ends up in the Redis publish event so the browser can receive it via SSE. But it is **never persisted to disk** in this codebase.

**Practical implication for the partner meeting:** If FOCUS wants the FHIR Observation stored somewhere permanently, the options are:
- (a) Save it in the local DB as a JSON column
- (b) Push it to their FHIR server via `FHIR_SERVER_URL`
- (c) Let them store it themselves after receiving it in the API response

Currently none of those happen automatically unless `FHIR_SERVER_URL` is set.

---

### How to switch from SQLite to Postgres

Two things to change:

**Step 1 — Install the driver (one-time):**

```
pip install psycopg2-binary
```

**Step 2 — Change one line in `.env`:**

```env
# Before
DATABASE_URL=sqlite:///./caregiver.db

# After
DATABASE_URL=postgresql+psycopg2://user:password@host:5432/caregiver
```

That's it. No code changes. SQLAlchemy creates the `fall_history` and `participant_session` tables automatically on startup (`init_db()` calls `create_all`).

---

## Topic 5 — Dashboard

### How the dashboard currently receives fall events

Two paths exist:

1. **In-process path (working without Redis):** Poller thread detects fall → calls `_on_fall_sync` → `run_coroutine_threadsafe` pushes to the SSE broker → browser gets the event via `/api/stream` EventSource.
2. **Redis path:** `redis_listener.py` subscribes to `fall_events` → pushes to the same SSE broker → browser. Useful when inference server and caregiver client are on separate machines.

Both paths feed the same SSE stream that the browser listens to.

---

### Does it matter what frontend technology their dashboard uses? (React / Vue / Angular / plain JS)

No. It does not matter.

`EventSource` (SSE) is a built-in browser API, identical in every framework:

```javascript
// Same line whether they use React, Vue, Angular, or plain JS
const es = new EventSource('/api/stream');
es.onmessage = (e) => console.log(JSON.parse(e.data));
```

There is nothing framework-specific about it. Their developer will know how to plug one line into their existing codebase regardless of what it is built with.

The questions that actually matter are:

- **Can they modify the dashboard?** (Is there a developer who has access and time to add one line?)
- **Do they want to modify it, or do they want us to provide a separate dashboard?**

---

### Where does the Redis question belong — dashboard section or infrastructure section?

It belongs in the **dashboard section**. The reason you care about Redis is entirely a dashboard concern — "how do we get live fall alerts to your browser?" Redis is just the internal mechanism on our side to achieve that.

When talking about their dashboard and real-time updates, the natural follow-up is:

> "On our side, we use Redis internally to pass events between our inference server and our backend. You never interact with Redis directly — your dashboard just connects to our SSE endpoint. But if you already have Redis in your stack, we can reuse it instead of running a separate container."

The infrastructure section is for things like CPU architecture, network layout, SSL. Redis belongs in the dashboard conversation because it only comes up as a consequence of wanting real-time alerts.

---

## Topic 6 — Patient Identifiers

### Why we need to agree on patient identifiers

**Reason 1 — Dashboard correlation:** The identifier shown on their dashboard must match what appears in the FHIR Observation and our fall history records. If their dashboard shows `"Participant_042"` but our FHIR says `"Patient/test-001"`, a caregiver cannot correlate the two.

**Reason 2 (more critical) — InfluxDB query:** The identifier is also what we use to query the right data from InfluxDB. In our system, `MAC_IDS` in `.env` maps 1:1 to `PATIENT_IDS`. The MAC address is the InfluxDB tag we filter by to get that patient's sensor data. The patient ID is what goes into the FHIR Observation.

Full chain:

```
MAC address (InfluxDB tag)  →  used to query the right sensor data from InfluxDB
      +
Patient ID (.env)           →  goes into FHIR Observation + local DB + dashboard display
```

Both need to be agreed. The MAC address comes from the SmarKo hardware automatically. The patient ID is whatever identifier Charite or FOCUS uses in their system — we need to match it exactly, not invent our own.

**The two concrete questions to ask:**
- What patient/participant IDs are registered in their system? (We put these in `PATIENT_IDS`)
- Is the SmarKo MAC address already linked to each patient in their records, or do we need to establish that mapping?

---

## Topic 7 — How to Run the Full System (End-to-End)

### Start commands

**Terminal 0 — Redis:**
```
docker run -d --name redis-local -p 6379:6379 redis:7
```

**Terminal 1 — Inference server:**
```powershell
cd C:\Users\hayat\Documents\6G\FallDetection_new\_6G_Integration_v2
.\.venv\Scripts\Activate.ps1
uvicorn inference_server.server:app --host 0.0.0.0 --port 8001
```
Expected: `Connected to Redis at redis://localhost:6379/0`

**Terminal 2 — Caregiver client:**
```powershell
cd C:\Users\hayat\Documents\6G\FallDetection_new\_6G_Integration_v2
.\.venv\Scripts\Activate.ps1
python -m caregiver_client.client
```
Expected: `FallEventBroker subscribing to redis://localhost:6379/0`

**Terminal 3 — InfluxDB marker writer:**
```powershell
cd C:\Users\hayat\Documents\6G\FallDetection_new\_6G_Integration_v2
.\.venv\Scripts\Activate.ps1
python influx_marker_writer.py
```
Expected: `Subscribed to 'fall_events' — waiting for fall events...`

**Browser:**
- Caregiver dashboard: http://localhost:8002/
- Patient popup: http://localhost:8002/patient/

---

### What happens when a fall is detected

```
Inference server detects fall
       │
       ├──► Returns FHIR to caregiver client (HTTP response)
       │         │
       │         ├──► Writes fall_history row (SQLite)
       │         ├──► Pushes to SSE broker (in-process)
       │         └──► Starts 10s auto-confirm timer
       │
       └──► Publishes to Redis fall_events
                 │
                 ├──► Subscriber 1 (caregiver dashboard) → alert banner flashes
                 ├──► Subscriber 2 (influx_marker_writer) → writes fall_marker=1 to InfluxDB
                 └──► Subscriber 3 (patient popup) → shows "Did you fall?" popup
                           │
                           ├──► Patient clicks YES → "Need help?" → POST /api/falls/{id}/confirm
                           ├──► Patient clicks NO → POST /api/falls/{id}/confirm?confirmed=no
                           └──► 10s timeout → treated as fall (server timer fires as backup)
```
