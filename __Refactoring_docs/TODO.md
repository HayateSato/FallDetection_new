# To-Do List — Architecture Refactoring (FOCUS Hosting Change)

## Background Summary
FOCUS is shifting from hosting everything on their premise to hosting **only InfluxDB + caregiver/patient dashboard**
in their namespace. The inference server and all ML/post-training components move to **MCS network/cloud**.

---

## 1. Architecture Decision

- [x] **VPN (WireGuard) NOT required for production**
  - Mobile app -> inference server: HTTPS over the internet (no VPN needed)
  - Mobile app -> InfluxDB (fall timestamp injection): internal FOCUS network (no VPN needed)
  - MCS inference server does NOT write to FOCUS InfluxDB directly (mobile app handles injection)
  - WireGuard conf is kept securely as a fallback for testing/troubleshooting only -- not used in deployment

---

## 2. Infrastructure -- MCS Side (Inference & Post-training Layer)

- [x] Decide hosting environment: **Hetzner** (company standard)
- [x] Public domain: **reuse existing MCS company domain** (no new domain purchase needed)
- [ ] **[Mohammed]** TLS certificate for the inference server subdomain
  - Ask MCS domain admin to create a DNS A-record pointing the subdomain (e.g. `fall-api.mcs-labs.de`) to the Hetzner server IP
  - On Hetzner: install Nginx as reverse proxy + Certbot for free Let's Encrypt certificate
    ```
    apt install certbot python3-certbot-nginx
    certbot --nginx -d fall-api.mcs-labs.de
    ```
  - Certbot auto-renews every 90 days; certificate is free via Let's Encrypt
  - Nginx forwards HTTPS :443 → inference-server :8001 and fall-dashboard :8002 internally
- [x] **Remove `mock_focus_fhir`** from local dev compose + codebase (FHIR opted out; `local_dev/mock_focus/` deleted)
- [x] Prepare **Docker Compose** deployment — split into two layers:
  - `inference_posttraining_layer/` (MCS/Hetzner): 8 services — inference-server, ml-dashboard, server-health, postgres, mlflow, minio, prometheus, grafana
  - `caregiver_layer/` (FOCUS mock / second laptop): 4 services — mock-app, mqtt, postgres, fall-dashboard
  - InfluxDB not containerised — uses external instance (MCS cloud for testing, FOCUS k3s in production)
- [x] **Two-laptop cross-machine test PASSED** (2026-06-03)
  - Laptop 2 (caregiver_layer) → Laptop 1 inference-server :8001: OK
  - Laptop 1 (inference_posttraining_layer) → Laptop 2 fall-dashboard :8002: OK
  - All services healthy on both machines; mock-app polling and patient popup at :8005
- [x] Document required .env variables for Mohammed's deployment (in `inference_posttraining_layer/.env.example`)
- [ ] **[Mohammed]** Deploy to Hetzner and verify all services are reachable

---

## 3. Code Changes -- Inference Server (MCS)

- [x] **Remove fall_history table** from MCS Postgres (IMPLEMENTED)
  - `patient_confirmed` and `needs_help` moved into `inference_log` table
  - Mobile app calls new `POST /inference/{observation_id}/confirm` endpoint after patient confirmation popup
  - Retraining pipeline updated to read labels from `inference_log` directly (no JOIN needed)
  - Alembic migration `0003` created: drops `fall_history`, adds the two new columns
- [x] Verify inference server does NOT connect to FOCUS InfluxDB (confirmed -- mobile app handles injection)
- [x] Confirm observation_id is returned in HTTP response (unchanged -- already in PredictResponse)
- [ ] Update .env / config for MCS deployment (DB host, MLflow URI, MinIO, Hetzner specifics)
- [x] Run Alembic migrations `0003` + `0004` on the Hetzner Postgres after deployment
  - Handled automatically by the `db-migrate` service in `docker-compose.yml` — runs `alembic upgrade head` on every `docker compose up` before inference-server starts

---

## 4. Caregiver Dashboard -- Instruction Document for FOCUS DevOps

FOCUS's caregiver dashboard is built in **Flutter** (not Python). We cannot hand them code.
We need to provide a specification / instruction document so their Flutter dev can implement
the same features that our Python `fall_dashboard` component provides.

**Agreed InfluxDB schema (decided by MCS, communicated to FOCUS):**
- Measurement: `fall_events`
- Tags:  `patient_id`, `device_id`
- Fields: `fall_detected` (bool), `patient_confirmed` (str: yes/no/not_answered),
          `needs_help` (bool), `observation_id` (str UUID), `confidence` (float),
          `model_version` (str)
- Timestamp: detection time of the fall event

**What the mobile app writes to InfluxDB (trigger: after patient confirmation popup):**
  One `fall_events` point per detected fall, with all fields above.

**MQTT broker — action required for FOCUS:**
> Currently the MQTT broker runs only on our (MCS) side for local development.
> In the final architecture the broker must be hosted in the FOCUS network so that
> MQTT Client A (mobile app) and MQTT Client B (caregiver dashboard) can reach it
> on the internal FOCUS network without going through the internet.
> **FOCUS DevOps does not know how to set this up — we need to include setup
> instructions in the document.**

**Instruction document needs to cover:**
- [ ] **MQTT broker setup**: how to deploy and configure a broker (e.g. Mosquitto) in the
      FOCUS network; recommended port (1883 / 8883 TLS); auth credentials format
- [ ] **MQTT Client B** (Flutter side): how to subscribe to `fall/alert/#`; payload format
      (JSON fields: patient_id, observation_id, fall_detected, confidence, patient_confirmed,
      needs_help, timestamp); how to display the live alert to the caregiver
- [ ] Live fall alert (SSE / MQTT): mobile app publishes to MQTT broker (FOCUS network);
      Flutter dashboard subscribes to `fall/alert/#` and shows the live alert
- [ ] Fall history view: query InfluxDB `fall_events` measurement; filter by patient_id, date range,
      patient_confirmed, needs_help; display count + table
- [ ] Patient list: query `fall_events` GROUP BY patient_id to get per-patient fall counts
- [ ] The `observation_id` field links a fall event back to MCS inference logs (future cross-reference)
- [ ] Exact Flux query examples for all three views above

- [ ] **Prepare the instruction document** (see `handover_docs_2/` for format)
- [ ] Send instruction document to FOCUS DevOps for Flutter implementation

---

## 5. Integration Testing with Isa (Mobile App)

- [ ] Test **mobile app -> inference server HTTPS** communication (end-to-end, new MCS endpoint)
- [ ] Test **InfluxDB marker injection** from mobile app (fall_events point written to FOCUS InfluxDB)
- [ ] Test **MQTT flow**: mobile app -> MQTT broker (FOCUS) -> caregiver dashboard SSE/alert
- [ ] Validate patient confirmation popup and the three response paths:
  - Patient confirms fall + requests help -> rescue MQTT message sent
  - Patient confirms fall, no help needed -> no rescue message
  - No response within 10s -> rescue message sent automatically
- [ ] Confirm MQTT payload includes observation_id so it can be cross-referenced with inference_log
- [ ] Test **POST /inference/{observation_id}/confirm** call from mobile app after popup
      (Isa needs to add this call after the confirmation popup)

> **Two-laptop local test PASSED (2026-06-03):** `caregiver_layer/` runs mock-app, mqtt, fall-dashboard
> on the second laptop. Cross-machine communication verified. The mock-app is now containerised
> in Docker inside the caregiver layer, correctly simulating the mobile app on the FOCUS network.

---

## 6. End-to-End Testing

- [ ] Full sequence: SmarKo -> mobile app -> inference server -> response -> InfluxDB injection -> MQTT -> caregiver dashboard alert
- [ ] Fall-history dashboard: verify it reads correctly from InfluxDB (filter by patient, period, help-requested)
- [ ] Retraining pipeline: feature_snapshot (MCS Postgres) -> retrain script -> MLflow -> hot-swap from ml-dashboard
- [ ] Grafana dashboards loading correctly for MCS-hosted components

---

## 7. Credentials & Configuration

- [ ] Store FOCUS InfluxDB credentials securely in MCS environment (if any MCS component needs them -- verify first)
- [ ] Store WireGuard conf securely even if unused (keep for contingency)
- [ ] Confirm the HTTPS public endpoint URL with Isa so mobile app config can be updated
- [ ] Document final .env variable list for MCS deployment

---

## Open Questions

| # | Question | Owner |
|---|----------|-------|
| 1 | ~~VPN needed?~~ **Resolved: NOT needed in production** (WireGuard conf kept for testing only) | DONE |
| 2 | ~~Where exactly is MCS hosting?~~ **Resolved: Hetzner** | DONE |
| 3 | ~~What InfluxDB schema?~~ **Resolved: MCS decides.** measurement=`fall_events`, fields: `fall_detected`, `patient_confirmed`, `needs_help`, `observation_id`, `confidence`, `model_version`; tags: `patient_id`, `device_id` | DONE |
| 4 | ~~Timestamp only or full fields?~~ **Resolved: full fields needed** (`patient_confirmed` + `needs_help` required in InfluxDB) | DONE |
| 5 | ~~Who updates caregiver dashboard?~~ **Resolved: FOCUS DevOps implements in Flutter** based on instruction document MCS prepares | DONE |
