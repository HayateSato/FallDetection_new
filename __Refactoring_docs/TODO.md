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
- [ ] Obtain TLS certificate for the inference server subdomain (e.g. fall-api.mcs-domain.de)
- [ ] Prepare **Docker Compose** deployment package (NOT Kubernetes -- MCS will not use K8s on their side)
  - Services to package: inference server, Postgres, MLflow, MinIO, Prometheus, Grafana,
    ml-dashboard, server-health dashboard
  - Goal: Mohammed can clone the repo and run `docker compose up` on Hetzner with minimal setup
- [ ] Test the Docker Compose stack locally before handing over to Mohammed
- [ ] Document required .env variables for Mohammed's deployment
- [ ] Mohammed deploys to Hetzner and verifies all services are reachable

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
- [ ] Run Alembic migration `0003` on the Hetzner Postgres after deployment

---

## 4. Code Changes -- Caregiver Dashboard (FOCUS side)

> Coordinate with FOCUS DevOps team -- this is their component.

- [ ] Update caregiver webapp server to fetch fall history **from InfluxDB** instead of Postgres
  - Old: queries Postgres fall_history table (was going to be in MCS namespace, now removed)
  - New: queries FOCUS InfluxDB using fall timestamps written by the mobile app
- [ ] Update fall-history dashboard query / API endpoint accordingly
- [ ] Confirm FOCUS has the right InfluxDB schema/fields to support the fall dashboard queries
  (patient_id, fall timestamp, confirmed, help_requested)

---

## 5. Integration Testing with Isa (Mobile App)

- [ ] Test **mobile app -> inference server HTTPS** communication (end-to-end, new MCS endpoint)
- [ ] Test **InfluxDB marker injection** from mobile app (fall timestamp + observation_id written to FOCUS InfluxDB)
- [ ] Test **MQTT flow**: mobile app -> MQTT broker (FOCUS) -> caregiver dashboard SSE/alert
- [ ] Validate patient confirmation popup and the three response paths:
  - Patient confirms fall + requests help -> rescue MQTT message sent
  - Patient confirms fall, no help needed -> no rescue message
  - No response within 10s -> rescue message sent automatically
- [ ] Confirm MQTT payload includes observation_id so it can be cross-referenced with inference_log

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
| 3 | What InfluxDB fields/measurement names will FOCUS use for fall timestamps? Needed to align caregiver dashboard query | FOCUS DevOps |
| 4 | Does the caregiver dashboard need help_requested and fall_confirmed fields in InfluxDB, or is the timestamp enough? | FOCUS DevOps + Hayate |
| 5 | Who updates the caregiver webapp server (fetch from InfluxDB) -- MCS or FOCUS DevOps? | Clarify ownership |
