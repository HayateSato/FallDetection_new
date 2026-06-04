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
  - `inference_posttraining_layer/` (MCS/Hetzner): 8 services — inference-server, ml-dashboard, server-health, postgres (`mcs_fall_postgres`), mlflow, minio, prometheus, grafana
  - `caregiver_layer/` (FOCUS mock / second laptop): **3 services** — mock-app, mqtt, fall-dashboard (**no Postgres** as of 2026-06-04)
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
- [x] Run Alembic migrations `0003` → `0004` → `0005` on the inference layer Postgres
  - Handled automatically by the `db-migrate` service in `inference_posttraining_layer/docker-compose.yml`
  - Migration 0005 (2026-06-04): drops `participant_session` from inference layer (was legacy from when both layers shared a schema)

---

## 4. Caregiver Dashboard -- Instruction Document for FOCUS DevOps

FOCUS's caregiver dashboard is built in **Flutter** (not Python). We cannot hand them code.
We need to provide a specification / instruction document so their Flutter dev can implement
the same features that our Python `fall_dashboard` component provides.

**Local two-laptop test PASSED (2026-06-03):**
End-to-end flow verified between two Windows laptops:
- mock-app detects fall → patient responds via popup at :8005 → `patient_confirmed` written
  as integer (1/0/-1) to InfluxDB → fall_dashboard queries InfluxDB → UI correctly
  displays confirmed falls, help requested, and per-patient fall history with timestamps.
- MQTT alert flow: mock-app publishes to broker → fall-dashboard subscribes and fans out via SSE.
- All services run in Docker (caregiver_layer compose) on laptop 2.

**Agreed InfluxDB schema (decided by MCS, communicated to FOCUS):**
- Measurement: `fall_events`
- Tags:  `patient_id`, `device_id`
- Fields: `fall_detected` (bool), `patient_confirmed` (int: 1=confirmed / 0=denied / -1=not_answered),
          `needs_help` (bool), `observation_id` (str UUID), `confidence` (float),
          `model_version` (str)
- Timestamp: detection time of the fall event

**What the mobile app writes to InfluxDB (trigger: after patient confirmation popup):**
  One `fall_events` point per detected fall, with all fields above.
  `patient_confirmed` must be written as an **integer** (not string) — InfluxDB field type
  locks on first write; mixing types causes silent write failures.

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
- [ ] **MQTT topics** (updated 2026-06-04 — two topics now):
      - `fall/possible/<patient_id>` — published immediately on fall detection (pre-confirmation); Flutter shows subtle "Possible fall" notice
      - `fall/alert/<patient_id>` — published after patient confirms or 10s timeout; Flutter shows full alert
- [ ] **MQTT Client B** (Flutter side): how to subscribe to both `fall/possible/#` and `fall/alert/#`; payload format
      (JSON fields: patient_id, observation_id, fall_detected, confidence, patient_confirmed,
      needs_help, timestamp, status); how to display the live alert to the caregiver
- [ ] Live fall alert (SSE / MQTT): mobile app publishes to MQTT broker (FOCUS network);
      Flutter dashboard subscribes to `fall/alert/#` and shows the live alert
- [ ] Fall history view: query InfluxDB `fall_events` measurement; filter by patient_id, date range,
      patient_confirmed (int), needs_help; display count + table
- [ ] Patient list: query `fall_events` GROUP BY patient_id to get per-patient fall counts
- [ ] The `observation_id` field links a fall event back to MCS inference logs (future cross-reference)
- [ ] Exact Flux query examples for all three views above (including integer filter: `r["patient_confirmed"] == 1`)

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

---

## 8. Caregiver Layer — K3s Integration (FOCUS Production)

FOCUS runs k3s in production. The caregiver layer (currently Docker Compose on a second laptop)
needs to be adapted for their cluster. Two open architectural questions before implementation.

### Open question: should the MQTT broker be inside or outside k3s?

**Option A — MQTT broker inside k3s** (wrapped as a Deployment/StatefulSet):

- Pros: lifecycle managed by k3s, single place to configure TLS + auth, easy scaling
- Cons: the mobile app (Isa) publishes from outside the cluster → needs a NodePort or LoadBalancer + Traefik IngressRouteTCP to expose port 1883/8883 externally. This is solvable but requires Traefik TCP routing config (different from HTTP ingress).
- `fall_dashboard` reaches the broker via its k8s internal DNS name — simple.

**Option B — MQTT broker outside k3s** (standalone Mosquitto on the FOCUS server, or existing broker):

- Pros: broker is reachable by any device on the network without cluster config; mobile app connects directly; no Traefik TCP setup needed.
- Cons: `fall_dashboard` (inside k3s) must reach the broker via the host IP or a k8s ExternalName Service — requires `MQTT_BROKER_HOST` set to the host machine IP (not a cluster DNS name).

**Recommended: Option A** — broker inside k3s, exposed via Traefik TCP IngressRoute on port 1883.
This keeps everything managed by one platform. The mobile app connects to the FOCUS server's external IP on :1883, same as it would with a standalone broker.

### Caregiver layer — what to wrap

Current 3 services (`caregiver_layer/docker-compose.yml`):

| Service | Wrap in k3s? | Notes |
|---------|-------------|-------|
| mqtt (Mosquitto) | Yes | Deployment + PVC for persistence + Traefik TCP route on :1883 |
| fall-dashboard | Yes | Deployment, no PVC (stateless), HTTP Ingress on caregiver path |
| mock-app | No — replace with real mobile app in production | Mock only; real Isa app runs on a phone, not in k3s |

### Tasks

- [ ] **Decide** with FOCUS DevOps: broker inside or outside k3s (see above)
- [ ] **If inside k3s**: write Helm chart or k8s manifests for mqtt + fall-dashboard; add Traefik TCP IngressRoute for MQTT port
- [ ] **If outside k3s**: configure `MQTT_BROKER_HOST` as ExternalName Service or plain IP in fall-dashboard Deployment env
- [ ] **Verify** MQTT_POSSIBLE_TOPIC (`fall/possible/#`) is documented in the FOCUS instruction doc — both topics must be subscribed by the Flutter client
- [ ] **Confirm** fall-dashboard `PATIENT_IDS` env var is populated from FOCUS's patient management system (or hardcoded in their k3s ConfigMap)
- [ ] **Remove mock-app** from any production manifests — it is a dev-only simulation of the mobile app

---

## Open Questions

| # | Question | Owner |
|---|----------|-------|
| 1 | ~~VPN needed?~~ **Resolved: NOT needed in production** (WireGuard conf kept for testing only) | DONE |
| 2 | ~~Where exactly is MCS hosting?~~ **Resolved: Hetzner** | DONE |
| 3 | ~~What InfluxDB schema?~~ **Resolved: MCS decides.** measurement=`fall_events`, fields: `fall_detected`, `patient_confirmed`, `needs_help`, `observation_id`, `confidence`, `model_version`; tags: `patient_id`, `device_id` | DONE |
| 4 | ~~Timestamp only or full fields?~~ **Resolved: full fields needed** (`patient_confirmed` + `needs_help` required in InfluxDB) | DONE |
| 5 | ~~Who updates caregiver dashboard?~~ **Resolved: FOCUS DevOps implements in Flutter** based on instruction document MCS prepares | DONE |
| 6 | MQTT broker — inside or outside k3s for FOCUS production? See Section 8 above. | Open |
