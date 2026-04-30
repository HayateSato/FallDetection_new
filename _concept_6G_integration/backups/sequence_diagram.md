# UML Sequence Diagram — Fall Detection System
Render with: VS Code + "Markdown Preview Mermaid Support" extension, or paste into https://mermaid.live

---

## Diagram 1 — Normal Polling Cycle (no fall detected)

```mermaid
sequenceDiagram
    participant W  as Wearable + App
    participant IDB as InfluxDB
    participant POL as Poller Thread<br/>(influx_poller.py)
    participant INF as Inference Server<br/>(:8001)
    participant DB  as Local DB<br/>(SQLite/Postgres)

    loop Every 10 seconds, per patient
        POL->>IDB: Flux query<br/>last 15s, filter macAddress
        IDB-->>POL: Raw ACC arrays + timestamps
        POL->>INF: POST /predict<br/>{acc_x, acc_y, acc_z, timestamps, patient_id}
        INF-->>POL: FHIR Observation<br/>{valueBoolean: false, confidence: 0.12}
        POL->>DB: record_fall(fall_detected=False)
        Note over POL,DB: No alert — result stored silently
    end

    W->>IDB: Continuous sensor upload (independent of polling)
```

---

## Diagram 2 — Fall Detected → Real-Time Alert

```mermaid
sequenceDiagram
    participant IDB as InfluxDB
    participant POL as Poller Thread<br/>(influx_poller.py)
    participant INF as Inference Server<br/>(:8001)
    participant RED as Redis<br/>(:6379)
    participant RLS as Redis Listener<br/>(redis_listener.py)
    participant DB  as Local DB
    participant BWS as Caregiver Browser<br/>(dashboard)

    POL->>IDB: Flux query (every 10s)
    IDB-->>POL: ACC data
    POL->>INF: POST /predict
    
    INF->>INF: Resample → Window → Features → XGBoost
    Note over INF: confidence=0.994 > 0.5 → FALL

    INF->>RED: PUBLISH fall_events<br/>{patient_id, mac_id, confidence, fhir_observation}
    INF-->>POL: FHIR Observation<br/>{valueBoolean: true, confidence: 0.994}

    POL->>DB: record_fall(fall_detected=True)<br/>returns fall_id

    RED-->>RLS: Message received on fall_events
    RLS->>RLS: Put event into asyncio Queue<br/>for each connected SSE client

    RLS-->>BWS: SSE data: {patient_id, mac_id,<br/>confidence, fall_id}
    Note over BWS: Alert banner appears<br/>"FALL DETECTED — 6c:1d:eb:... (0.994)"
```

---

## Diagram 3 — Patient Feedback Flow

```mermaid
sequenceDiagram
    participant RED  as Redis<br/>(:6379)
    participant RLS  as Redis Listener<br/>(caregiver backend)
    participant WEB  as web.py<br/>(FastAPI)
    participant DB   as Local DB
    participant PBWS as Patient Browser<br/>(/patient/)
    participant CBWS as Caregiver Browser<br/>(dashboard)

    Note over RED,PBWS: Fall event arrives (from Diagram 2)

    RED-->>RLS: fall_events message
    RLS-->>PBWS: SSE → patient popup appears<br/>"Strong impact — did you fall?"
    
    WEB->>WEB: start_auto_confirm_timer(fall_id)<br/>10s countdown starts

    alt Patient answers YES within 10s
        PBWS->>WEB: POST /api/falls/{id}/confirm<br/>?confirmed=yes
        WEB->>WEB: cancel_auto_confirm_timer(fall_id)
        WEB->>DB: update patient_confirmed = "yes"
        WEB-->>PBWS: {ok: true}
        Note over WEB,DB: Fall confirmed by patient
    else Patient answers NO within 10s
        PBWS->>WEB: POST /api/falls/{id}/confirm<br/>?confirmed=no
        WEB->>WEB: cancel_auto_confirm_timer(fall_id)
        WEB->>DB: update patient_confirmed = "no"
        WEB-->>PBWS: {ok: true}
    else 10s timeout — no response
        WEB->>DB: patient_confirmed stays "not_answered"
        Note over WEB,DB: Treated as unconfirmed fall
    end

    CBWS->>WEB: GET /api/falls (every 15s auto-refresh)
    WEB->>DB: list_falls()
    DB-->>WEB: rows including patient_confirmed
    WEB-->>CBWS: {falls: [..., {patient_confirmed: "yes"}]}
    Note over CBWS: History table updates with confirmation status
```

---

## Diagram 4 — Optional FHIR Server Push

```mermaid
sequenceDiagram
    participant POL as Poller Thread
    participant INF as Inference Server<br/>(:8001)
    participant FSV as Their FHIR Server<br/>(external)
    participant POL2 as Poller Thread<br/>(response handler)

    POL->>INF: POST /predict

    alt FHIR_SERVER_URL is set in .env
        INF->>FSV: POST /Observation<br/>FHIR R4 Observation JSON<br/>Authorization: Bearer <token>
        FSV-->>INF: 201 Created
        Note over INF,FSV: Fall registered in their FHIR database
    else FHIR_SERVER_URL is empty
        Note over INF: Skip — FHIR push disabled
    end

    INF-->>POL2: HTTP response<br/>{fhir_observation: {...}, is_fall: true, confidence: 0.994}
    Note over POL2: FHIR is ALSO in the response<br/>regardless of whether push happened
```

---

## Diagram 5 — Full System Startup Sequence

```mermaid
sequenceDiagram
    participant ENV as .env file
    participant CLI as client.py<br/>(entry point)
    participant APP as FastAPI app<br/>(web.py)
    participant DB  as Local DB
    participant RLS as Redis Listener
    participant RED as Redis
    participant POL as Poller Thread
    participant INF as Inference Server

    CLI->>ENV: load_dotenv()
    CLI->>APP: uvicorn.run(app)
    
    APP->>DB: init_db() — create tables if not exist
    APP->>RLS: broker.start() — connect to Redis
    RLS->>RED: SUBSCRIBE fall_events
    RED-->>RLS: Subscribed OK

    Note over APP: FastAPI startup hook complete

    APP->>DB: ensure_session(patient_id)<br/>for each patient in PATIENT_IDS
    
    APP->>POL: InfluxPoller.start()<br/>(background thread)
    Note over POL: Polling loop begins

    loop Every 10s
        POL->>INF: POST /predict
        INF-->>POL: result
    end
```

---

## Plain-text summary (actors and their communication protocols)

```
Actor                     Communicates with        Protocol
─────────────────────────────────────────────────────────────
Wearable App         →   InfluxDB                  HTTPS
Poller Thread        →   InfluxDB                  HTTP (Flux query)
Poller Thread        →   Inference Server          HTTP POST /predict
Inference Server     →   Redis                     Redis PUBLISH
Inference Server     →   Their FHIR Server         HTTP POST (optional)
Redis Listener       ←   Redis                     Redis SUBSCRIBE
Redis Listener       →   SSE Queue                 asyncio (in-process)
Web.py (FastAPI)     →   Browser (caregiver)       SSE + HTTP responses
Web.py (FastAPI)     →   Browser (patient)         SSE + HTTP responses
Browser (patient)    →   Web.py (FastAPI)          HTTP POST (feedback)
Browser (caregiver)  →   Web.py (FastAPI)          HTTP GET (polling)
influx_marker_writer ←   Redis                     Redis SUBSCRIBE
influx_marker_writer →   InfluxDB                  HTTP (write point)
```
