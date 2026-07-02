# Fall Detection System — Architecture Handover

**Audience:** anyone joining the project who needs the big picture before diving into a specific component (FOCUS DevOps, Charite tech contact, new MCS engineer).
**Repo / branch:** `_6G_Integration_v2_mqtt/` on branch `6G-integration_with_MQTT`.
**Scope:**
- One-page architecture diagram
- Sequence diagrams (per-call, per-fall, per-retrain)
- Prerequisites for the whole system
- Pointers to user-flow docs and component-specific handovers

This is the doc to read **first**. Every other handover doc assumes you've already seen the architecture here.

---

## 1. Architecture overview

### 1.1 The picture in one diagram

```
      External                         FOCUS NAMESPACE                  OUR NAMESPACE  (mcs-fall-detection)
   ──────────────                  ───────────────────────             ────────────────────────────────────

  ┌────────────┐                                                       ┌──────────────────────────────────┐
  │  SmarKo    │ BLE                                                   │                                  │
  │  wearable  │ ──── ACC + barometer ────►                            │   inference-server (:8001)       │
  └────────────┘                                                       │   ▲      │       │               │
                                                                       │   │HTTP  │bg     │bg             │
  ┌────────────┐ HTTP POST /predict (over Ingress)                     │   │      ▼       ▼               │
  │  Mobile    │ ─────────────────────────────────────────────────────►│   │   inference  feature_        │
  │  app       │ HTTP response: fall_detected, observation_id, ...     │   │   _log (PG)  snapshot (PG)   │
  │  (Isa)     │ ◄─────────────────────────────────────────────────────│   │                              │
  └────────────┘                                                       │   │   (optional) FHIR push ─────►│ (FOCUS FHIR
       │                                                               │   │                              │  if URL set)
       │    [popup: did you fall? need help?]                          │   │                              │
       │                                                               │   │                              │
       │    MQTT PUBLISH fall/alert/<patient_id>  (over WS :9001)      │   │                              │
       └────────────────────────────────────────────►                  │   │   mqtt-broker  (:1883/9001)  │
                                                                       │   │     │                        │
                                                                       │   │     │ subscribe fall/alert/# │
                                                                       │   │     ▼                        │
                                                                       │   │   fall-dashboard (:8002)     │
                                                                       │   │     │                        │
                                                                       │   │     ├─► fall_history (PG)    │
                                                                       │   │     │                        │
                                                                       │   │     └─► SSE  /api/stream     │
                                                                       │   │                ▲             │
                                                                       │   │                │             │
                              ┌────────────────────────────────┐       │   │                │             │
                              │  FOCUS namespace               │       │   │   postgres (StatefulSet)    │
                              │  ────────────────              │       │   │   minio (StatefulSet)       │
                              │  FHIR server  ◄────────────────┼───────┘   │   mlflow (Deployment)       │
                              │  InfluxDB                       │           │   prometheus, grafana       │
                              │  Patient Dashboard ─────────────┼──────────►│   ▲                          │
                              │  (Flutter web — FOCUS DevOps)   │  REST/SSE │   │                          │
                              └────────────────────────────────┘            └───┴──────────────────────────┘
```

### 1.2 What lives where — quick reference

| Component | Owner | Where it runs | Image type |
|-----------|-------|---------------|-----------|
| SmarKo wearable | SmarKo | patient's body | hardware |
| Mobile app | Isa (SmarKo) | patient's phone | mobile (no K8s) |
| FHIR server | FOCUS | FOCUS namespace | existing FOCUS infra |
| InfluxDB | FOCUS | FOCUS namespace | existing FOCUS infra |
| Patient Dashboard (Flutter web) | FOCUS DevOps | FOCUS namespace | existing FOCUS infra |
| **inference-server** | **MCS (us)** | our namespace | custom image |
| **fall-dashboard** | **MCS (us)** | our namespace | custom image |
| MLflow | MCS | our namespace | custom image (thin wrapper) |
| Postgres | MCS | our namespace | public image |
| MQTT broker (Mosquitto) | MCS | our namespace | public image |
| MinIO | MCS | our namespace | public image |
| Prometheus + Grafana | MCS | our namespace | public images |

For the cross-namespace traffic table see `01_k8s.md` Section 2.

---

## 2. Sequence diagrams

### 2.1 Per-call: a /predict request that does NOT detect a fall

```
mobile app    inference-server     postgres    [BackgroundTask runs after HTTP response]
   │                  │                │
   ├── POST /predict ►│                │
   │  (450 ACC + 225 │                 │
   │   pressure samples)               │
   │                  │                │
   │   [LSB→g, resample, features]     │
   │   [XGBoost.predict_proba]         │
   │                  │                │
   │   confidence=0.12, fall=false     │
   │◄──── 200 OK ─────┤                │
   │                  ├── BG: write inference_log (fall_detected=false) ─►│
   │                  ├── BG: write feature_snapshot (×N feature rows) ──►│
```

Latency on the HTTP path: ~30–60 ms p50. The Postgres writes happen after the response — they never add latency to `/predict`.

### 2.2 Per-fall: a /predict that detects a fall, end-to-end

```
mobile         inference-       MQTT          fall-           postgres        Patient
app            server          broker         dashboard                       Dashboard
                                                                              (browser)

  ├── POST /predict ►│                                                            │
  │      (fall window)                                                            │
  │   [features + XGBoost]                                                        │
  │                  │                                                            │
  │   fall=true      │                                                            │
  │   confidence=0.82│                                                            │
  │   observation_id=UUID                                                         │
  │◄────── 200 ──────┤                                                            │
  │                  ├── BG: write inference_log (fall=true, UUID) ─► postgres    │
  │                  ├── BG: write feature_snapshot ──────────────► postgres      │
  │                                                                               │
  │  [show popup, 10s countdown]                                                  │
  │  patient: "Yes I fell, yes need help"                                         │
  │                                                                               │
  ├── PUBLISH fall/alert/<pid> ──►│                                               │
  │   {observation_id, patient_confirmed='yes', needs_help=true, ...}             │
  │                               │                                               │
  │                               ├── routed to fall/alert/# ──►│                 │
  │                                                             │                 │
  │                                                             ├── write fall_history ──► postgres
  │                                                             │                 │
  │                                                             ├── SSE event ───────────►│  red flag appears
  │                                                                                       │  on patient card
```

Total wall time from POST /predict to red flag in browser: ~1–2 seconds (assuming the patient answers immediately). The 10-second countdown is the dominant factor when the patient takes their full time.

### 2.3 Per-fall, alternative branches

| Patient action | MQTT payload | DB write | SSE fan-out | Caregiver sees |
|----------------|--------------|:--------:|:-----------:|----------------|
| "Yes I fell, yes need help" | confirmed=yes, needs_help=true | yes | **yes** | red flag |
| "Yes I fell, no I'm okay" | confirmed=yes, needs_help=false | yes | no | nothing |
| "No I didn't fall" (false positive) | confirmed=no | yes | no | nothing |
| no response in 10s | confirmed=not_answered | yes | **yes** | amber flag |

All four cases are stored in `fall_history` because they're all useful for retraining. Only the two actionable ones reach the caregiver.

### 2.4 Per-retrain: the full retraining loop

```
admin                ml_dashboard        retrain.py            postgres        MLflow         MinIO
                       (:8004)
  ├─ click "Retrain"►│
  │                   ├─ subprocess: python -m retrain.retrain ─►│
  │                                                              │
  │                                          [data_pipeline.py: SQL JOIN
  │                                          inference_log + feature_snapshot + fall_history
  │                                          ON observation_id]
  │                                                              ├─ SELECT ──► postgres
  │                                                              ◄────────────┤
  │                                                              │
  │                                          [80/20 split, train XGBoost]
  │                                                              │
  │                                                              ├─ log params + metrics ─► MLflow tracking
  │                                                              ├─ log .pkl artifact ──────────────────► MinIO
  │                                                              ├─ register_model("fall_detector") ──► MLflow Registry
  │                                          [Stage = None]
  │                                                              │
  │   "run #123 done, recall +0.02"
  │ ◄─────────────────┤                                          │
  │                                                              │
  ├─ click "Promote v123 to Staging" ──►│
  │                   ├─ MLflow API: transition_model_version_stage("Staging") ──►│
  │                                                                                │
  ├─ smoke test                                                                    │
  │                                                                                │
  ├─ click "Promote to Production" ───►│                                           │
  │                   ├─ MLflow API: transition_model_version_stage("Production") ─►│
  │                                                                                │
  ├─ click "Hot-swap inference to Production" ──►│
  │                   ├─ POST /model/switch {source: registry, version: Production} ─► inference-server
  │                                                                                    │
  │                                                                                    [load .pkl from MinIO via MLflow]
  │                                                                                    [reload feature_names.json]
  │   "active version: v123"                                                            │
  │ ◄─────────────────┤◄───────────────────────────────────────────────────────────────┤
```

The four steps (retrain, promote-staging, promote-production, hot-swap) are deliberately separate — each is a human decision. We do not auto-promote.

---

## 3. Prerequisites

### 3.1 What you need before "the system can run"

This list assumes **production deployment**. For local dev, see `01_k8s.md` Section 9 (the local dry-run path).

| Prerequisite | Owner | Status as of 2026-04-29 |
|--------------|-------|-------------------------|
| K8s cluster with kubectl access | FOCUS DevOps | confirmed (32 GB RAM, Traefik) |
| Container registry URL | FOCUS DevOps | **TBD** — blocks helm install |
| Default StorageClass | FOCUS DevOps | TBD — empty in values.yaml works if cluster has a default |
| Ingress hostname / FQDN | FOCUS DevOps | **TBD** — blocks ingress |
| FOCUS namespace name | FOCUS DevOps | **TBD** |
| Our namespace name (default `mcs-fall-detection`) | MCS | confirmed |
| FHIR server URL | FOCUS | optional — empty = no FHIR push |
| InfluxDB URL | FOCUS | not needed for our pipeline (mobile app talks to InfluxDB directly) |
| MQTT broker (we ship our own) | MCS | confirmed |
| Postgres (we ship our own StatefulSet) | MCS | confirmed |
| MinIO (we ship our own StatefulSet) | MCS | confirmed |
| MLflow (we ship our own) | MCS | confirmed |
| SmarKo wearable hardware | SmarKo | confirmed |
| Mobile app implementing /predict + MQTT publish + popup | Isa (SmarKo) | in progress — see `04_mobile_app_integration.md` |
| Patient Dashboard fall panel integration | FOCUS DevOps | in progress — see `05_web_app_integration.md` |
| API key rotation policy | MCS + FOCUS jointly | TBD — currently dev key in `.env` |
| MQTT broker auth | MCS + FOCUS jointly | TBD — currently anonymous |

### 3.2 What you need to test the system end-to-end

Bare minimum smoke test (5 components, ~10 minutes):

1. inference-server reachable on `/health`
2. fall-dashboard reachable on `/api/falls`
3. MQTT broker accepting connections on 1883 (TCP) and 9001 (WS)
4. Postgres accepting writes (`fall_history` rises after a test publish)
5. SSE stream delivers events to a browser tab on `/api/stream`

Detailed e2e test in `04_mobile_app_integration.md` Section 8 and `05_web_app_integration.md` Section 7.

### 3.3 What you need to retrain the model

- A populated `fall_history` table with at least ~50 confirmed (yes/no) labels per patient. With less than that, retraining mostly relearns the public-dataset baseline.
- `python` 3.11+ + the `retrain/requirements.txt` deps.
- MLflow tracking server reachable (or local file store).
- MinIO reachable (or local file store for artifacts).

See `02_fall_detection_algorithm.md` Section 4 for the full retraining playbook.

### 3.4 What you need to operate ongoing

- An assigned **ML admin** (whoever runs retraining + promotion) — see `08_user_flow_admin.md`
- An assigned **caregiver responder** (per-shift, hospital staff) — see `07_user_flow_caregiver.md`
- A **monitoring rota** — Prometheus alerts → someone's pager. Nothing is wired up to a pager yet — alerts live in Grafana only.

---

## 4. Component reference — what each piece does in one sentence

| Component | One-line description |
|-----------|----------------------|
| `inference_server/` | FastAPI :8001 — runs `/predict`, exposes `/model/switch`, `/model/list`, writes inference_log + feature_snapshot to Postgres in background |
| `fall_dashboard/` | FastAPI :8002 — MQTT subscriber on `fall/alert/#`, writes fall_history, SSE-fans-out actionable events |
| `ml_dashboard/` | Admin UI :8004 — buttons for retrain / promote / hot-swap |
| `server_health/` | Admin UI :8006 — aggregates `/health` of all our pods |
| `ml_pipeline/` | Shared signal-processing + inference-engine code (used by inference_server) |
| `shared_db/` | SQLAlchemy ORM + Alembic migrations — used by inference_server and fall_dashboard |
| `retrain/` | MLflow retraining CLI + data pipeline + synthetic seeder |
| `model/` | XGBoost `.pkl` files (versions v0, v0_lsb_int, v3, v5_lsb) |
| `local_dev/mock_app/` | Local-dev simulation of the SmarKo mobile app |
| `local_dev/mock_focus/` | Local-dev simulation of FOCUS namespace (FHIR + Patient Dashboard mock) |
| `helm/fall-detection/` | The production Helm chart |
| `helm/mock-focus/` | Dry-run-only chart for cross-namespace testing on Docker Desktop K8s |
| `infrastructure/` | Local-dev Docker Compose (Postgres + MQTT + MinIO + MLflow + Prometheus + Grafana) |

---

## 5. Where each user lives in the system

The system has 3 human roles. Each has a dedicated user-flow doc.

| Role | What they do | Their entry point | User-flow doc |
|------|--------------|-------------------|---------------|
| **Patient** | Wears the wearable, responds to fall popup | Mobile app on their phone | [`06_user_flow_patient.md`](06_user_flow_patient.md) |
| **Caregiver** | Watches Patient Dashboard, responds to alerts | FOCUS Patient Dashboard (Flutter web) | [`07_user_flow_caregiver.md`](07_user_flow_caregiver.md) |
| **Admin / tech** | Operates ML pipeline, retrains, deploys | `ml_dashboard` :8004 + `server_health` :8006 + cluster CLI | [`08_user_flow_admin.md`](08_user_flow_admin.md) |

These are mutually exclusive in the production auth model — a caregiver cannot see the admin UI and vice versa. Roles are issued by FOCUS SSO (planned, not yet wired — see todo.md Step 11.5.4).

---

## 6. Data lifecycle — where each piece of data lives

| Data | Origin | Stored in | Read by |
|------|--------|-----------|---------|
| Raw ACC samples | Wearable BLE | not stored long-term — passed in /predict body and discarded | inference-server only |
| Inference result (fall_detected, confidence) | inference-server | inference_log (Postgres) + Prometheus | retrain pipeline + Grafana |
| Feature vector at predict time | inference-server | feature_snapshot (Postgres) | retrain pipeline only |
| Patient confirmation | mobile app popup | fall_history (Postgres) | fall_dashboard SSE + retrain |
| FHIR Observation | inference-server | Optionally pushed to FOCUS FHIR | FOCUS Patient Dashboard (if FOCUS reads its own FHIR) |
| Biosignals (HR etc.) | mobile app | FOCUS InfluxDB (we don't touch it) | FOCUS Patient Dashboard |
| Patient demographics | FOCUS FHIR | FOCUS namespace | FOCUS Patient Dashboard |
| Trained model `.pkl` | retrain.py | MinIO (`mlflow-artifacts/`) | inference-server hot-swap |
| MLflow runs / metrics | retrain.py | Postgres (`mlflow` DB) | MLflow UI |

**Key invariant:** the only data we hold long-term about a patient is their fall events + features. We do **not** store raw ACC samples beyond the predict call. We do **not** store biosignals. We do **not** store demographics. This is intentional for data-protection reasons.

---

## 7. Local-dev reference

Reproduce the whole system on the operator's laptop in 6 terminals:

```powershell
# from _6G_Integration_v2_mqtt/ as cwd

# 1 — Infrastructure (Postgres, MQTT, MinIO, MLflow, Prometheus, Grafana)
docker-compose -f infrastructure/docker-compose.yml up

# 2 — Inference server
uvicorn inference_server.server:app --host 0.0.0.0 --port 8001

# 3 — Fall dashboard (caregiver SSE feed)
python -m fall_dashboard.main          # http://localhost:8002

# 4 — Mock mobile app (InfluxDB → /predict → MQTT alert)
python -m local_dev.mock_app.main      # popup at http://localhost:8005

# 5 — ml_dashboard (admin UI for retrain + hot-swap)
python -m ml_dashboard.main            # http://localhost:8004

# 6 — server_health (admin status dashboard)
python -m server_health.main           # http://localhost:8006
```

Each component has its own README under `_6G_Integration_v2_mqtt/<component>/README.md`.

---

## 8. Cross-references

- [`01_k8s.md`](01_k8s.md) — deployment, helm chart, image build/push
- [`02_fall_detection_algorithm.md`](02_fall_detection_algorithm.md) — model, training, monitoring, versioning
- [`04_mobile_app_integration.md`](04_mobile_app_integration.md) — the mobile-app contract
- [`05_web_app_integration.md`](05_web_app_integration.md) — the Patient Dashboard contract
- [`06_user_flow_patient.md`](06_user_flow_patient.md) — patient-side flow
- [`07_user_flow_caregiver.md`](07_user_flow_caregiver.md) — caregiver-side flow
- [`08_user_flow_admin.md`](08_user_flow_admin.md) — admin-side flow
- `REFACTOR_DOCUS/deployment_architecture.md` (in repo) — exhaustive two-namespace breakdown
- Existing handover docs at `handover_docs/` — earlier versions, still accurate for narrower audiences

---

## 9. Things that will bite you (system-level)

- **Two namespaces, not one.** All cross-namespace calls are by DNS name (`<svc>.<ns>.svc.cluster.local`). If you forget the namespace suffix, requests fail with confusing DNS errors.
- **`replicas: 1` on inference-server is mandatory.** Per-process state (Prometheus counters, model registry singleton). Do not scale up.
- **MQTT topic structure: `fall/alert/<patient_id>`.** The dashboard subscribes to `fall/alert/#`. If you change the topic shape on the mobile-app side, the dashboard stops getting events.
- **The mobile app must include `observation_id` in MQTT payload.** Without it, retraining cannot link confirmation to features. Drop reason: "ambiguous".
- **InfluxDB is FOCUS's, not ours.** We never read from FOCUS InfluxDB at runtime. Local dev uses a separate cloud InfluxDB that has nothing to do with FOCUS.
- **`MODEL_VERSION` in chart ConfigMap is the persistent default.** Hot-swap is in-memory; pod restart reverts unless you `helm upgrade`.

---

## 10. Contact

| For | Reach out to |
|-----|--------------|
| System architecture, anything cross-cutting | Hayate (MCS) |
| K8s cluster, ingress, networking | FOCUS DevOps |
| Mobile app | Isa (SmarKo) |
| Patient Dashboard UI | FOCUS DevOps (UI team) |
| Clinical / patient population | Charite |
