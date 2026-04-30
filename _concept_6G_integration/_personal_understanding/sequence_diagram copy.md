# UML Sequence Diagram — Fall Detection System (with hosting groups)
Render with: VS Code + "Markdown Preview Mermaid Support" extension (v1.26+), or paste into https://mermaid.live

## Color key

| Color | Hosting group |
|-------|--------------|
| 🔵 Blue | **Caregiver Client process** — single Python process on one machine (port 8002) |
| 🟠 Orange | **Inference Server process** — separate Python process (port 8001) |
| 🟣 Purple | **Shared infrastructure** — Docker containers (Redis) |
| 🟢 Green | **External / partner systems** — InfluxDB, FHIR server (not our machines) |
| ⬜ White | **End-user devices** — browsers, wearable |

---

## Diagram 1 — Normal Polling Cycle (no fall detected)

```mermaid
sequenceDiagram
    box White End-user devices
        participant W as Wearable + App
    end
    box rgb(200,230,200) External / partner systems
        participant IDB as InfluxDB
    end
    box rgb(173,213,255) Caregiver Client process (:8002)
        participant POL as Poller Thread<br/>(influx_poller.py)
        participant DB  as Local DB<br/>(SQLite/Postgres)
    end
    box rgb(255,200,160) Inference Server process (:8001)
        participant INF as Inference Server<br/>(server.py)
    end

    W->>IDB: Continuous sensor upload (Bluetooth → App → HTTPS)

    loop Every 10 seconds, per patient
        POL->>IDB: Flux query — last 15s, filter macAddress
        IDB-->>POL: Raw ACC arrays + timestamps
        POL->>INF: POST /predict<br/>{acc_x, acc_y, acc_z, timestamps_ms, patient_id}
        INF-->>POL: FHIR Observation<br/>{valueBoolean: false, confidence: 0.12}
        POL->>DB: record_fall(fall_detected=False)
        Note over POL,DB: No alert — stored silently
    end
```

---

## Diagram 2 — Fall Detected → Real-Time Alert

```mermaid
sequenceDiagram
    box rgb(200,230,200) External / partner systems
        participant IDB as InfluxDB
    end
    box rgb(255,200,160) Inference Server process (:8001)
        participant INF as Inference Server<br/>(server.py)
    end
    box rgb(220,200,255) Shared infrastructure
        participant RED as Redis<br/>(:6379)
    end
    box rgb(173,213,255) Caregiver Client process (:8002)
        participant POL as Poller Thread<br/>(influx_poller.py)
        participant RLS as Redis Listener<br/>(redis_listener.py)
        participant WEB as FastAPI web.py
        participant DB  as Local DB
    end
    box White End-user devices
        participant BWS as Caregiver Browser<br/>(dashboard)
    end

    POL->>IDB: Flux query (every 10s)
    IDB-->>POL: ACC data

    POL->>INF: POST /predict
    INF->>INF: Resample 25→50Hz → Window 9s<br/>→ 16 features → XGBoost
    Note over INF: confidence=0.994 > 0.5 → FALL

    INF->>RED: PUBLISH fall_events<br/>{patient_id, mac_id, confidence, fhir_observation}
    INF-->>POL: FHIR Observation {valueBoolean: true, confidence: 0.994}

    Note over POL,INF: These two happen concurrently —<br/>publish and response are independent

    POL->>DB: record_fall(fall_detected=True) → returns fall_id

    RED-->>RLS: Message on fall_events channel
    RLS->>WEB: Put event into asyncio Queue<br/>(in-process handoff — no network)
    WEB-->>BWS: SSE stream: {patient_id, mac_id, confidence, fall_id}
    Note over BWS: Alert banner: "FALL DETECTED —<br/>6c:1d:eb:... (confidence 0.994)"
```

---

## Diagram 3 — Patient Feedback Flow

```mermaid
sequenceDiagram
    box rgb(220,200,255) Shared infrastructure
        participant RED as Redis<br/>(:6379)
    end
    box rgb(173,213,255) Caregiver Client process (:8002)
        participant RLS as Redis Listener<br/>(redis_listener.py)
        participant WEB as FastAPI web.py<br/>+ auto-confirm timer
        participant DB  as Local DB
    end
    box White End-user devices
        participant PBWS as Patient Browser<br/>(/patient/)
        participant CBWS as Caregiver Browser<br/>(dashboard)
    end

    Note over RED,PBWS: Fall event has arrived (see Diagram 2)

    RED-->>RLS: fall_events message
    RLS-->>PBWS: SSE → patient popup:<br/>"Strong impact — did you fall?"
    WEB->>WEB: start_auto_confirm_timer(fall_id)<br/>10s countdown

    alt Patient answers within 10s — YES
        PBWS->>WEB: POST /api/falls/{id}/confirm?confirmed=yes
        WEB->>WEB: cancel_auto_confirm_timer(fall_id)
        WEB->>DB: patient_confirmed = "yes"
        WEB-->>PBWS: {ok: true}
    else Patient answers within 10s — NO
        PBWS->>WEB: POST /api/falls/{id}/confirm?confirmed=no
        WEB->>WEB: cancel_auto_confirm_timer(fall_id)
        WEB->>DB: patient_confirmed = "no"
        WEB-->>PBWS: {ok: true}
    else 10s timeout — no response
        WEB->>DB: patient_confirmed stays "not_answered"
        Note over WEB: Treated as unconfirmed fall
    end

    loop Every 15s (auto-refresh)
        CBWS->>WEB: GET /api/falls
        WEB->>DB: list_falls()
        DB-->>WEB: rows with patient_confirmed values
        WEB-->>CBWS: {falls: [...]}
        Note over CBWS: History table shows confirmation status
    end
```

---

## Diagram 4 — Optional FHIR Server Push

```mermaid
sequenceDiagram
    box rgb(173,213,255) Caregiver Client process (:8002)
        participant POL as Poller Thread<br/>(influx_poller.py)
    end
    box rgb(255,200,160) Inference Server process (:8001)
        participant INF as Inference Server<br/>(server.py)
    end
    box rgb(200,230,200) External / partner systems
        participant FSV as Their FHIR Server<br/>(external HTTP)
    end

    POL->>INF: POST /predict

    alt FHIR_SERVER_URL is set in .env
        INF->>FSV: POST /Observation<br/>FHIR R4 Observation JSON<br/>Authorization: Bearer <token>
        FSV-->>INF: 201 Created
        Note over INF,FSV: Fall registered in partner's FHIR DB
    else FHIR_SERVER_URL is empty
        Note over INF: FHIR push disabled — skip
    end

    INF-->>POL: HTTP response always includes FHIR:<br/>{fhir_observation: {...}, is_fall: true, confidence: 0.994}
    Note over POL: FHIR is in the response regardless<br/>of whether push happened
```

---

## Diagram 5 — InfluxDB Fall Marker Writer (standalone script)

```mermaid
sequenceDiagram
    box rgb(220,200,255) Shared infrastructure
        participant RED as Redis<br/>(:6379)
    end
    box rgb(255,240,180) Standalone script<br/>(separate terminal)
        participant IMW as influx_marker_writer.py
    end
    box rgb(200,230,200) External / partner systems
        participant IDB as InfluxDB
    end

    Note over IMW: Started separately:<br/>python influx_marker_writer.py

    IMW->>RED: SUBSCRIBE fall_events
    RED-->>IMW: Subscribed OK

    loop On every fall event
        RED-->>IMW: fall_events message<br/>{patient_id, mac_id, ...}
        IMW->>IDB: Write Point("SMART_DATA")<br/>.tag("macAddress", mac_id)<br/>.field("fall_marker", 1)
        Note over IMW,IDB: Fall timestamped in InfluxDB<br/>alongside raw sensor data
    end
```

---

## Diagram 6 — Startup Sequence

```mermaid
sequenceDiagram
    box White Configuration
        participant ENV as .env file
    end
    box rgb(173,213,255) Caregiver Client process (:8002)
        participant CLI as client.py<br/>(entry point)
        participant APP as FastAPI app<br/>(web.py)
        participant DB  as Local DB
        participant RLS as Redis Listener
        participant POL as Poller Thread
    end
    box rgb(220,200,255) Shared infrastructure
        participant RED as Redis<br/>(:6379)
    end
    box rgb(255,200,160) Inference Server process (:8001)
        participant INF as Inference Server
    end

    CLI->>ENV: load_dotenv()
    CLI->>APP: uvicorn.run(app)

    APP->>DB: init_db() — create tables if missing
    APP->>RLS: broker.start() — connect to Redis
    RLS->>RED: SUBSCRIBE fall_events
    RED-->>RLS: OK

    Note over APP: FastAPI startup hook complete

    APP->>DB: ensure_session(patient_id)<br/>for each PATIENT_ID in .env
    APP->>POL: InfluxPoller.start() — background thread

    loop Every 10s (polling loop)
        POL->>INF: POST /predict
        INF-->>POL: result
    end
```

---

## Summary — What lives where

```
┌─────────────────────────────────────────────────────────┐
│  🟠 INFERENCE SERVER process  (port 8001)               │
│     inference_server/server.py                          │
│     Start: uvicorn inference_server.server:app          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  🔵 CAREGIVER CLIENT process  (port 8002)               │
│     client.py           ← entry point                   │
│       ├── influx_poller.py   [background thread]        │
│       ├── redis_listener.py  [async task in FastAPI]    │
│       ├── web.py             [FastAPI event loop]       │
│       └── db.py              [SQLite file on disk]      │
│     Start: python -m fall_dashboard.client            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  🟣 REDIS  (Docker container, port 6379)                │
│     Start: docker run -d -p 6379:6379 redis:7           │
│     Not our code — third-party service                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  🟡 INFLUX MARKER WRITER  (standalone script)           │
│     influx_marker_writer.py                             │
│     Start: python influx_marker_writer.py               │
│     Subscribes Redis → writes to InfluxDB               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  🟢 EXTERNAL (not our machines)                         │
│     InfluxDB  — cloud-hosted by partner                 │
│     FHIR Server — partner's server (optional push)      │
└─────────────────────────────────────────────────────────┘

⬜ BROWSERS — end-user devices (caregiver PC, patient phone)
```
