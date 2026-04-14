# To-Do List — 6G / Charite Integration

**Owners:** Hayate (H), Isa (I)  
**Status key:** [ ] not started · [~] in progress · [x] done · [!] blocked

---

## 0 — Blockers: Ask These First (H)

Nothing in Steps 2–7 can start until these are answered.

| # | Question | Ask | Blocking what |
|---|----------|-----|---------------|
| 0.1 | [ ] InfluxDB bucket name, measurement name, ACC field names (`bosch_acc_x/y/z`?) | Isa | Step 3, 4 |
| 0.2 | [ ] SmarKo sample rate in this trial — 25Hz or 100Hz? | Isa / Charite | Step 3, 4 |
| 0.3 | [ ] MQTT broker: host, port, topic naming convention (e.g. `fall/{patient_id}`?), auth | FOCUS DevOps | Step 1, 2 |
| 0.4 | [ ] Patient ID format in their system (what goes in FHIR + local DB) | FOCUS | Step 3, 4, 5 |
| 0.5 | [ ] Do they have a FHIR server, and is FHIR format required? | FOCUS | Step 5 |
| 0.6 | [ ] Where should the detection result land — FHIR DB / dashboard DB / MQTT only? | FOCUS | Step 5 |
| 0.7 | [ ] Kubernetes namespace + naming conventions | FOCUS DevOps | Step 7 |
| 0.8 | [ ] Container registry — do they have one, or should images be on Docker Hub? | FOCUS DevOps | Step 7 |
| 0.9 | [ ] Is Isa the dashboard developer who will integrate our event subscriber? | FOCUS | Step 2 |

---

## Step 1 — Replace Redis with MQTT in Inference Server (H)

**File:** `_6G_Integration_v2/inference_server/server.py`

- [ ] 1.1 Add `paho-mqtt` to `requirements.txt`
- [ ] 1.2 Replace `aioredis` import + `_redis_client` with an MQTT client (`paho.mqtt.client`)
- [ ] 1.3 Replace `await _redis_client.publish(...)` with `mqtt_client.publish(topic, payload)` — topic TBD from 0.3
- [ ] 1.4 Update `.env` / `.env.example`: remove `REDIS_URL`, add `MQTT_BROKER_HOST`, `MQTT_BROKER_PORT`, `MQTT_TOPIC_PREFIX`, `MQTT_USERNAME`, `MQTT_PASSWORD`
- [ ] 1.5 Handle MQTT connection failure gracefully (same pattern as Redis — skip publish, don't crash)

**Reuse from `_OLD`:** none needed — MQTT is a new transport, simpler than aioredis.

---

## Step 2 — Replace Redis Subscriber with MQTT in Caregiver Client (H)

**File:** `_6G_Integration_v2/caregiver_client/redis_listener.py`

- [ ] 2.1 Rewrite `FallEventBroker` to subscribe to MQTT instead of Redis
- [ ] 2.2 Confirm with Isa (0.9): does the dashboard consume MQTT directly (browser MQTT client), or does it still need SSE from our backend?
  - If SSE still needed → keep the SSE fan-out logic, just change the event source from Redis to MQTT
  - If direct MQTT → we can remove SSE entirely and just publish to MQTT
- [ ] 2.3 Also update `influx_marker_writer.py` (currently Redis subscriber → writes to InfluxDB) to subscribe to MQTT instead

---

## Step 3 — Separate the Data Fetcher (H)

**Current state:** `influx_poller.py` is coupled inside `caregiver_client/` — fetches data, calls inference, writes DB all in one.

- [ ] 3.1 Create `data_fetcher/` as a standalone module
- [ ] 3.2 Move InfluxDB query logic from `influx_poller.py` into `data_fetcher/influx_fetcher.py` — only responsibility: fetch sensor data per patient and pass to API Caller
- [ ] 3.3 Create `data_fetcher/api_caller.py` that takes sensor data → calls `POST /predict` on inference server
- [ ] 3.4 Results flow to MQTT Event Publisher (Step 1), not back through the fetcher
- [ ] 3.5 Update Dockerfile/compose so each component can be an independent Kubernetes pod

---

## Step 4 — Update InfluxDB Config with Confirmed Field Names (H)

**Blocked on: 0.1, 0.2**

- [ ] 4.1 Update `.env`:
  ```
  INFLUXDB_BUCKET=<confirmed>
  ACC_FIELD_X=<confirmed>
  ACC_FIELD_Y=<confirmed>
  ACC_FIELD_Z=<confirmed>
  HARDWARE_ACC_SAMPLE_RATE=<25 or 100>
  ```
- [ ] 4.2 Check `_build_query()` in `influx_poller.py` — confirm `macAddress` tag name is correct for their setup
- [ ] 4.3 Confirm `r["_measurement"] == "SMART_DATA"` — ask Isa what their measurement name is

---

## Step 5 — FHIR / Output Format (H)

**Blocked on: 0.5, 0.6**

- [ ] 5.1 If FHIR server required → confirm `FHIR_SERVER_URL` is set in `.env` (already implemented in `server.py`)
- [ ] 5.2 If they want FHIR stored in local DB → add JSON column to `FallHistory` in `db.py`
- [ ] 5.3 If no FHIR required → confirm plain JSON output from `/predict` is sufficient; update `fhir_converter.py` if format changes needed
- [ ] 5.4 Check with FOCUS whether LOINC `72514-3` (fall risk score, reused for confidence) will pass their FHIR validator

---

## Step 6 — MLOps: Prometheus + Grafana (H)

Reuse directly from `_OLD/system_operator/ml_server/services/`.

### 6a — Prometheus metrics

- [ ] 6.1 Copy `_OLD/system_operator/ml_server/services/metrics_collector.py` → `_6G_Integration_v2/inference_server/services/metrics_collector.py` (no changes needed)
  - Provides: `fall_detections_total`, `inference_latency_seconds`, `model_confidence`
- [ ] 6.2 Add to `requirements.txt`:
  ```
  prometheus-client
  prometheus-fastapi-instrumentator
  ```
- [ ] 6.3 Add 3 lines to `server.py`:
  ```python
  from prometheus_fastapi_instrumentator import Instrumentator
  Instrumentator().instrument(app).expose(app)
  from inference_server.services.metrics_collector import record_prediction
  ```
- [ ] 6.4 Call `record_prediction(model_version, fall_detected, confidence, latency)` inside `/predict` handler (after inference, before return)

### 6b — Postgres inference log (optional — confirm if needed for this partner)

- [ ] 6.5 Copy `_OLD/system_operator/ml_server/services/db_writer.py` → `_6G_Integration_v2/inference_server/services/db_writer.py`
- [ ] 6.6 Copy `_OLD/shared/db/models.py` + `session.py` → `_6G_Integration_v2/shared/db/`
- [ ] 6.7 Add `BackgroundTasks` DB write call in `/predict` (same pattern as `_OLD` server.py)
- [ ] 6.8 Add `DATABASE_URL` to `.env`

### 6c — Grafana dashboards

- [ ] 6.9 Add Prometheus + Grafana to Helm chart (Step 7) — can reuse `_OLD/infrastructure/prometheus/` and `_OLD/infrastructure/grafana/dashboards/`
- [ ] 6.10 Wire 3 dashboards:
  - `ml_server_overview` — request rate, error rate, p95 latency (alert if > 2s)
  - `model_performance` — confidence distribution, fall rate per hour (alert if median confidence < 0.6)
  - `fall_events_timeline` — falls today, scatter over 24h (Postgres datasource — only if 6b done)

---

## Step 7 — Model Hot-Swap Endpoint (H)

Currently `server.py` has `GET /model/info` but no `POST /model/switch`.  
The `_OLD` server has the full implementation.

- [ ] 7.1 Port `POST /model/switch` from `_OLD/system_operator/ml_server/server.py` into `_6G_Integration_v2/inference_server/server.py`
- [ ] 7.2 Port `GET /model/list` (returns available model versions from disk)
- [ ] 7.3 Confirm model files (`v0`, `v3`, etc.) will be packaged in the Helm chart image, or mounted as a volume

---

## Step 8 — Two-Role Dashboard (I — with API from H)

**Owner: Isa** for UI. H provides the backend API endpoints.

### H provides:
- [ ] 8.1 Document `GET /health` and `GET /model/info` endpoints (already exist) — for Admin view
- [ ] 8.2 Confirm `GET /api/patients` and `GET /api/falls` return correct fields for Caregiver view
- [ ] 8.3 Add role-based auth if needed:
  - Copy `_OLD/shared/auth/jwt_utils.py` → `_6G_Integration_v2/shared/auth/jwt_utils.py`
  - Add login endpoint to `web.py`

### Isa builds:
- [ ] 8.4 **Admin view:** service health status (inference server up/down), model version loaded, last prediction time — no patient names or IDs
- [ ] 8.5 **Caregiver view:** patient list scoped to their group/floor, fall history, real-time alerts via SSE or MQTT
- [ ] 8.6 Integrate our SSE endpoint (`/api/stream`) or MQTT subscriber (depending on decision from Step 2.2)

---

## Step 9 — Helm Chart (H + FOCUS DevOps)

**Blocked on: 0.7, 0.8**

- [ ] 9.1 Write `Dockerfile` for each component (inference server, data fetcher, API caller, event publisher/subscriber, model updater)
- [ ] 9.2 Create `helm/fall-detection/` chart with one `values.yaml` for all config (InfluxDB URL, MQTT broker, patient IDs, etc.)
- [ ] 9.3 Each component = one Kubernetes `Deployment` + `Service`
  ```
  fall-detection/
  ├── inference-api/
  ├── data-fetcher/
  ├── api-caller/
  ├── event-publisher/
  ├── event-subscriber/
  └── model-updater/
  ```
- [ ] 9.4 Confirm resource limits per pod with FOCUS DevOps (12 cores, 32GB total, shared namespace)
- [ ] 9.5 Confirm ingress controller type (nginx? traefik?) for exposing inference API
- [ ] 9.6 Consider raising with FOCUS DevOps: they may prefer to write the Helm chart themselves given the chart format — we provide a `docker-compose.yml` equivalent as reference

---

## Step 10 — End-to-End Integration Test (H + I)

- [ ] 10.1 Point Data Fetcher at their InfluxDB (service name in same namespace)
- [ ] 10.2 Trigger a test fall (manually inject data into InfluxDB or use CSV replay)
- [ ] 10.3 Verify: Inference API returns FHIR Observation with correct patient ID format
- [ ] 10.4 Verify: MQTT event published with correct topic + payload
- [ ] 10.5 Verify: Dashboard (Isa) receives real-time alert
- [ ] 10.6 Verify: Fall history retrievable via `GET /api/falls`
- [ ] 10.7 Verify: Admin sees service health; Caregiver sees only their patients
- [ ] 10.8 Verify: Prometheus `/metrics` endpoint scraped; Grafana shows inference latency + fall rate

---

## Summary by Owner

### Hayate (H)
Steps 0–7, Step 9, backend API for Step 8, Step 10 verification.  
**First action:** Send questions from Step 0 to Isa and FOCUS DevOps — everything else is blocked until those are answered.

### Isa (I)
Step 8 (dashboard UI) once H has confirmed API endpoints.  
Step 10.5 — dashboard alert test.

---

## What Can Start Now (no blockers)

| Task | Notes |
|------|-------|
| Step 6a (Prometheus metrics) | Self-contained — copy 1 file + 3 lines in server.py |
| Step 7 (model hot-swap endpoint) | Copy from `_OLD` — no external dependencies |
| Step 6b (Postgres log) — optional | Only if confirmed needed; requires deciding on DB |
| Step 8.1–8.3 (document API for Isa) | Write up current endpoints |
