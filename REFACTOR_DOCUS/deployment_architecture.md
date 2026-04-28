# Deployment Architecture — Fall Detection / FOCUS Integration
**Last updated:** 2026-04-27  
**Status:** Implementation in progress — see todo.md for open items

---

## Two Kubernetes Namespaces

The SmarKo wearable and mobile app are **external to both namespaces** — the wearable
is a physical device and the app runs on the patient's phone. They communicate into
the cluster over the network.

> **Local development note:** `mock_app` (in `local_dev/`) plays the role of the mobile app
> in this diagram.
>
> **Two InfluxDB instances — do not confuse them:**
>
> | Instance | What it is | Role |
> |----------|-----------|------|
> | **FOCUS-hosted InfluxDB** (in diagram) | Runs in FOCUS namespace | Stores biosignals written by real mobile app; read by Patient Dashboard for HR/bio display. **Never touched by our inference code.** |
> | **Our cloud InfluxDB** (`ecosystem-influxdb.smarko-health.de`) | MCS/SmarKo cloud instance | Used **only by `mock_app`** as a fake BLE wearable substitute. Exists because `mock_app` cannot read from the real SmarKo hardware. **Not part of the production system at all.** |
>
> **In production:** the real mobile app reads raw ACC from the SmarKo wearable via BLE and
> POSTs it directly to `inference_server`. Our cloud InfluxDB disappears from the picture
> entirely. The FOCUS InfluxDB remains, but only for the Patient Dashboard biosignal panel —
> our inference pipeline never reads from it.

```
  SmarKo wearable  (physical device — external)
      │ BLE
      ▼
  Mobile App  (patient's phone — external)          [mock_app plays this role in local dev]
      │  writes raw ACC, fall result,
      │  patient confirmation, window timestamps
      │                           │
      │                           │ HTTP POST /predict
      ▼                           ▼
┌──────────────────────┐     ┌──────────────────────────────────────────────────────┐
│   FOCUS NAMESPACE    │     │                  OUR NAMESPACE                       │
│                      │     │                                                      │
│  InfluxDB            │◄────┼─── mobile app writes bio data + fall results        │
│  (FOCUS-hosted)      │     │    [local dev: mock_app reads our cloud InfluxDB     │
│                      │     │     as fake BLE input — unrelated to this box]       │
│                      │     │  inference_server  :8001                             │
│                      │     │    - XGBoost inference                               │
│  FHIR Server  ◄──────┼─────┼─── optional FHIR push (if configured)               │
│  (patient            │     │    HTTP response → fall_detected                     │
│   demographics)      │     │          │ BackgroundTask                            │
│                      │     │          ▼                                           │
│                      │     │    inference_log + feature_snapshot                  │
│  Patient Dashboard   |     │    (Postgres — fall_detection DB)                   │
|  (one web app shown  │     |                                                      |
│   to the caregiver): │     │                                                      │
│   • patient info     │     │  MQTT broker  :1883                                  │
│     panel — FHIR     │     │    fall/alert/<patient_id>                           │
│     demographics     │     │                                                      │
│     (height, weight) │     │                                                      │
│   • biosignal panel  │     │                                                      │
│     — HR etc from    │     │                                                      │
│     InfluxDB         │     │                                                      │
│   • fall panel       │     │                                                      │
│     (fall history +  │     │                                                      │
│     real-time alert) │     │                                                      │
│     ← served by      │     │                                                      │
│       fall_dashboard │     │                                                      │
│    │ reads FHIR  ────┤     │                                                      │
│    │ reads InfluxDB  │     │                                                      │
│    │ reads our API ──┼─────┼──► **fall_dashboard**  :8002                          │
│                      │     │      - subscribes MQTT fall/alert/#                 │
└──────────────────────┘     │      - writes fall_history (Postgres)               │
                             │      - SSE fan-out to dashboard                     │
                             │      - GET /api/falls, /api/patients                │
                             │                                                      │
                             │  Postgres  :5432                                     │
                             │    database: fall_detection                          │
                             │      - inference_log                                 │
                             │      - feature_snapshot                              │
                             │      - fall_history                                  │
                             │      - participant_session                           │
                             │    database: mlflow                                  │
                             │      - MLflow internal tables                        │
                             │                                                      │
                             │  MLflow tracking server  :5000                       │
                             │  Prometheus  :9090                                   │
                             │  Grafana  :3000                                      │
                             │  MinIO  (MLflow artifact store)                      │
                             └──────────────────────────────────────────────────────┘
```

---

## Future: Mock FOCUS Namespace

Once the system is working end-to-end, the goal is to test true cross-namespace
communication by creating a third namespace that simulates what the real FOCUS
environment looks like. This replaces the cloud InfluxDB shortcut with a proper
in-cluster setup.

```
  SmarKo wearable  (physical — external)
      │ BLE
      ▼
  Mobile App  (patient's phone — external)
      │
      ▼
┌──────────────────────────┐     ┌─────────────────────────────────────────────┐
│   MOCK FOCUS NAMESPACE   │     │              OUR NAMESPACE                  │
│                          │     │                                             │
│  InfluxDB                │◄────┼── mobile app writes sensor data             │
│  (new packaged instance) │     │                                             │
│                          │     │  inference_server                           │
│  FHIR Server (mock)      │     │  fall_dashboard                             │
│                          │     │  Postgres                                   │
│  Patient Dashboard       │     │  MQTT broker                                │
│  (one web app shown to   │     │  MLflow / Prometheus / Grafana              │
│   the caregiver —        │     │                                             │
│   patient info panel     │     │                                             │
│   + fall panel)          │     │                                             │
│                          │     │                                             │
└──────────────────────────┘     └─────────────────────────────────────────────┘
```

The Patient Dashboard (Isa's real UI) is one web app shown to the caregiver. It
combines two panels: patient info (demographics from FHIR, biosignals like HR from
InfluxDB) and the fall panel (fall history + real-time alerts, data from our
fall_dashboard API). It lives in the mock FOCUS namespace — mirroring production.
Our namespace only contains backend services. This validates that cross-namespace
HTTP and MQTT connections work correctly before the real FOCUS deployment.

---

## What Lives Where

### External (no namespace — physical devices / phone apps)

| Component | What it does | Owned by |
|-----------|-------------|----------|
| SmarKo wearable | Records ACC + barometer; streams via BLE to mobile app | Charite / patient |
| Mobile App | Reads BLE wearable; calls `/predict`; shows patient confirmation popup; writes all results to InfluxDB | Isa / FOCUS |

> **Local dev substitute:** `mock_app` replaces the mobile app. It reads from our cloud
> InfluxDB instead of BLE, and simulates the 10s confirmation timeout.

### FOCUS Namespace (existing — do not modify)

| Component | What it does | Owned by |
|-----------|-------------|----------|
| InfluxDB | Stores raw ACC/barometer data + fall detection results + patient confirmation + window timestamps (all written by mobile app) | FOCUS (hosted) |
| FHIR Server | Stores patient demographics and identifiers (Patient resources) | FOCUS |
| FOCUS Dashboard — patient info component | Reads FHIR (demographics) + InfluxDB (bio data) for patient overview | Isa / FOCUS |

> **Local dev note:** Our cloud InfluxDB instance (`fd_test` bucket) is used only by
> `mock_app` as a fake BLE wearable substitute — it is **not** a stand-in for the
> FOCUS-hosted InfluxDB. The two instances serve completely different roles and are
> not interchangeable. In production, our cloud InfluxDB plays no role whatsoever.

### Our Namespace (what we build and deliver via Helm)

| Component | What it does | Status |
|-----------|-------------|--------|
| `inference_server` | Receives `/predict` HTTP, runs XGBoost, returns FHIR observation; writes `inference_log` to Postgres via BackgroundTask | Implemented — Postgres write pending |
| `fall_dashboard` | Subscribes to MQTT `fall/alert/#`; writes `fall_history` to Postgres; serves dashboard API + SSE | Implemented — Postgres migration pending |
| `mqtt_broker` | eclipse-mosquitto; routes `fall/alert/<patient_id>` between mobile app and fall_dashboard | Running locally |
| `Postgres` | One instance, two logical databases: `fall_detection` (our data) + `mlflow` (MLflow internals) | Pending |
| `MLflow tracking server` | Logs training runs, metrics, model artifacts; hosts Model Registry | Pending — blocked on data sharing agreement |
| `Prometheus` | Scrapes `/metrics` from inference_server every 15s | Pending (code ready — no infra yet) |
| `Grafana` | 3 dashboards: server overview, model performance, fall events timeline | Pending |
| `MinIO` | Artifact store for MLflow model `.pkl` files. MLflow uses `s3://mlflow-artifacts/` bucket on MinIO. inference_server downloads `.pkl` from MLflow (which reads from MinIO) via `POST /model/switch`. Base model baked into image as fallback if MinIO/MLflow unavailable at startup. | **Decided — in our namespace** (2026-04-17) |

---

## Data Sources — Summary Table

| Storage | Type | What is stored | Who writes | Who reads | Purpose |
|---------|------|---------------|-----------|-----------|---------|
| **InfluxDB** (FOCUS-hosted; our cloud instance for local dev) | Time-series | Raw ACC + barometer from SmarKo wearable; biosignals (HR etc.); fall results + patient confirmation written by mobile app | Mobile app (Isa) writes · `mock_app` writes (local dev only) | **Patient Dashboard** (biosignal display panel) · `mock_app` reads as a local-dev shortcut (replaces BLE wearable input) | **Biosignal display in Patient Dashboard only. NOT used for inference in production.** In production the mobile app reads from the wearable via BLE and sends windows directly to inference_server. `seed_test_data --influxdb` reads from here as a dev utility to populate Postgres without a live inference_server. |
| **FHIR Server** (FOCUS-hosted) | FHIR R4 | Patient demographics, identifiers, Patient resources | FOCUS | Our dashboard (read-only, patient info panel) | Patient identity and medical context |
| **Postgres `fall_detection` DB** (our namespace) | Relational | `inference_log`: every prediction (patient_id, model_version, fall_detected, confidence, latency_ms, detection_time) · `feature_snapshot`: feature name+value per inference · `fall_history`: confirmed alerts (linked to inference_log via FK, patient_confirmed, needs_help) · `participant_session`: fall count per patient | `inference_server` writes inference_log + feature_snapshot · `fall_dashboard` writes fall_history + participant_session | Caregiver dashboard (fall history) · `retrain.py` (labelled training data) | Fall history for dashboard display; labelled dataset for model retraining |
| **Postgres `mlflow` DB** (our namespace) | Relational | MLflow runs, parameters, metrics, model registry stages | MLflow tracking server | MLflow UI · inference_server (`/model/switch` loads from registry) | Experiment tracking and model versioning |
| **MinIO** (our namespace) | Object store | Trained model `.pkl` artifacts, per MLflow run | MLflow (via `retrain.py`) | `inference_server` (loads model from registry) | Model artifact storage for retraining pipeline |

### InfluxDB — decision confirmed (2026-04-27)

**FOCUS-hosted** — InfluxDB lives in the FOCUS namespace and is operated by FOCUS. It is NOT in our Helm chart.

Our cloud InfluxDB (`ecosystem-influxdb.smarko-health.de`) is used only by `local_dev/mock_app/` as a source of fake ACC windows. It has no role in the production system.

---

## Key Cross-Namespace Connections

These are the only points where our namespace communicates with the FOCUS namespace.
All are initiated by us (outbound from our namespace), and all are read-only except the `/predict` call.

| From | To (FOCUS namespace) | Protocol | Direction | Data | Notes |
|------|---------------------|----------|-----------|------|-------|
| `mock_app` (local dev only) | InfluxDB | InfluxDB HTTP API | Read | ACC windows | **Local dev shortcut only.** Replaces BLE wearable input. Not a production connection. |
| Mobile App (production) | InfluxDB | InfluxDB HTTP API | Write | Raw biosignals, fall results, patient confirmation | Mobile app writes sensor data for Patient Dashboard display. |
| Mobile App (production) | `inference_server` | HTTP | Write | POST /predict (ACC window) | **Primary inference path.** Mobile app reads from BLE wearable, sends directly — InfluxDB not involved. |
| `inference_server` | FHIR Server | HTTPS | Write (optional) | FHIR Observation | Only if `FHIR_SERVER_URL` is configured. |
| Patient Dashboard | FHIR Server | HTTPS | Read | Patient demographics | Isa's app reads demographics for the patient info panel. |
| Patient Dashboard | `fall_dashboard` | HTTP/SSE | Read | Fall history, real-time alerts | Fall panel in Patient Dashboard calls our API. |

---

## Postgres Schema — Key Tables

```sql
-- Written by inference_server (BackgroundTask after /predict)
inference_log (
    id              SERIAL PRIMARY KEY,
    patient_id      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    fall_detected   BOOLEAN NOT NULL,
    confidence      FLOAT NOT NULL,
    latency_ms      INT,
    detection_time  TIMESTAMPTZ NOT NULL,
    window_size     INT
)

-- One row per feature per inference (written by inference_server)
feature_snapshot (
    id              SERIAL PRIMARY KEY,
    inference_id    INT REFERENCES inference_log(id),
    feature_name    TEXT NOT NULL,
    feature_value   FLOAT NOT NULL
)

-- Written by fall_dashboard (on MQTT fall/alert arrival)
fall_history (
    id                  SERIAL PRIMARY KEY,
    inference_id        INT REFERENCES inference_log(id),  -- FK to inference_log
    patient_id          TEXT NOT NULL,
    fall_detected       BOOLEAN NOT NULL,
    patient_confirmed   TEXT,     -- 'yes' / 'no' / 'not_answered'
    needs_help          BOOLEAN,
    detection_time      TIMESTAMPTZ,
    alert_time          TIMESTAMPTZ
)
```

The `inference_id` FK is what makes the retraining query clean:

```sql
-- Labelled training dataset for retrain.py
SELECT
    il.confidence,
    il.model_version,
    il.detection_time,
    fs.feature_name,
    fs.feature_value,
    fh.patient_confirmed,
    fh.needs_help
FROM inference_log il
JOIN feature_snapshot fs ON fs.inference_id = il.id
JOIN fall_history fh      ON fh.inference_id = il.id
WHERE il.fall_detected = TRUE
  AND fh.patient_confirmed = 'yes'
```

---

## MinIO — Model Artifact Storage (decided 2026-04-17)

Model `.pkl` files are **not baked into the Docker image** (except as a startup fallback).
They live in MinIO and are downloaded at runtime by the inference server via the MLflow registry.

### How it fits together

```
retrain.py                               inference_server
  │                                           │
  │ mlflow.xgboost.log_model()               │ POST /model/switch
  ▼                                           │   {"mlflow_stage": "Production"}
MLflow tracking server  ────────────────────► │
  │  (run metadata in Postgres mlflow DB)     │  get_model_version_by_alias()
  │  (artifact .pkl → MinIO bucket)           │  downloads .pkl from MinIO
  ▼                                           ▼
MinIO  s3://mlflow-artifacts/            loads into memory (hot-swap)
```

### Startup ordering in K8s

inference_server must not crash if MinIO or MLflow is not yet ready. Strategy:
1. On startup, load the **base model from the baked-in `model/model_v0/` files** (always present in image)
2. An `initContainer` (or retry loop) waits for MinIO and MLflow to be healthy before the main container starts
3. Once running, operators use `POST /model/switch` to load the production-registered model from the registry

This means the system is always functional after startup, even if MinIO is temporarily unavailable.

### Environment variables needed (add to .env for local dev with MinIO)

```
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000    # point MLflow artifact store at MinIO
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
MLFLOW_ARTIFACT_ROOT=s3://mlflow-artifacts/     # bucket name
```

### Local dev order with MinIO

```
1. docker-compose -f infrastructure/docker-compose.yml up   # starts MinIO on :9000
2. Open http://localhost:9002 (MinIO console) → create bucket: mlflow-artifacts
3. mlflow ui --backend-store-uri sqlite:///./mlruns.db --workers 1
4. python -m retrain.retrain --model-version v0 --register   # artifacts now go to MinIO
5. .\scripts\switch_model.ps1 -Stage Production              # inference_server loads from MinIO
```

---

## Unified Caregiver/Admin URL — Role-Based Routing (decided 2026-04-28)

All four dashboards live behind a single ingress hostname (e.g. `dashboard.charite.de`).
Path-based routing dispatches to different namespaces; role claims in the auth token
gate access. **Caregivers and admins are mutually exclusive** — neither sees the other's
views.

```
              dashboard.charite.de  (single Traefik ingress)
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
       /             /admin/ml          /admin/health
   (caregiver)        (admin)             (admin)
        │                 │                 │
        ▼                 ▼                 ▼
  ┌────────────┐    ┌────────────┐    ┌────────────────┐
  │ Patient    │    │ ml_         │    │ server_health  │
  │ Dashboard  │    │ dashboard   │    │ dashboard      │
  │ (Isa's UI) │    │ (FastAPI)   │    │ (FastAPI)      │
  │            │    │             │    │                │
  │ FOCUS NS   │    │ OUR NS      │    │ OUR NS         │
  └────────────┘    └────────────┘    └────────────────┘
        │
        │  fall panel calls  → /api/falls, /api/stream
        └──────────────────► fall_dashboard (OUR NS)
```

| Route | Purpose | Role | Namespace |
|-------|---------|------|-----------|
| `/` | Patient Dashboard — patient info panel + fall panel + biosignals | **caregiver** | FOCUS |
| `/admin/ml` | ml_dashboard — retrain, register, promote, hot-swap | **admin** | ours |
| `/admin/health` | Server health — aggregate `/health` endpoints, plain-language status | **admin** | ours |

### Why split this way

- **Patient Dashboard (caregiver-only):** clinical staff need patient demographics,
  biosignals, and fall alerts. They do not need (and should not see) MLOps controls
  that could change the live model.
- **ml_dashboard (admin-only):** controls that affect the live model serving real
  patients. Hot-swap, promote, retrain are state-changing actions that require an
  audit trail and must be restricted.
- **server_health (admin-only):** SRE-level visibility into pod status. Caregivers
  don't need this; they have a different mental model (patients, not pods).
- **Grafana stays separate:** still admin-only but lives at its own URL or behind
  `/admin/grafana`. It serves a different purpose (time-series ops investigation)
  and is not part of the unified routing decision above.

### Auth flow (planned)

1. User logs into the FOCUS SSO once
2. SSO returns a JWT with a `role` claim (`caregiver` | `admin`)
3. Traefik middleware extracts the role and 403s on disallowed paths
4. Each backend (Patient Dashboard, ml_dashboard, server_health) re-validates the
   role from the JWT — defence in depth, never trust the proxy alone

This requires coordination with FOCUS DevOps + Isa (he owns the Patient Dashboard
auth integration). Tracked in todo.md Step 11.5.

---

## Open Decisions Still Needed

| Decision | Who decides | Impact |
|----------|-------------|--------|
| FHIR output required? | FOCUS | Whether `FHIR_SERVER_URL` is used in production |
| ~~InfluxDB: FOCUS-hosted or our namespace?~~ | ~~FOCUS + us~~ | **Decided:** FOCUS-hosted, not in our Helm chart (2026-04-27) |
| Container registry location | FOCUS DevOps | Step 9 Helm chart delivery |
| Kubernetes namespace names | FOCUS DevOps | Helm chart `values.yaml` |
| ~~Model files: Docker image or mounted volume?~~ | ~~FOCUS DevOps + us~~ | **Decided:** Option B — MinIO in our namespace. See Step 7.3 in todo.md. |
| Data sharing agreement with Charite | Charite + FOCUS | Unlocks Step 11 (MLflow retraining) |


---

**ALL** the services go to K8s. But there's a distinction between "we build the image ourselves" and "we use a prebuilt vendor image":

| Service | Image source | Who builds it | Where it lives in production |
| --- | --- | --- | --- |
| **inference_server** | `inference_server/Dockerfile` (our code) | **us** | FOCUS's container registry |
| **fall_dashboard** | `fall_dashboard/Dockerfile` (our code) | **us** | FOCUS's container registry |
| **mlflow** | `infrastructure/mlflow/Dockerfile` (our small wrapper) | **us** | FOCUS's container registry |
| Postgres | `postgres:16-alpine` | postgres-org | Docker Hub (pulled by K8s) |
| MQTT broker | `eclipse-mosquitto:2` | eclipse-org | Docker Hub |
| MinIO | `minio/minio:latest` | minio-org | Docker Hub |
| Prometheus | `prom/prometheus:latest` | prom-org | Docker Hub |
| Grafana | `grafana/grafana:10.4.0` | grafana-org | Docker Hub |

**Production startup ordering — what runs where:**

`FOCUS K8s cluster (our namespace)
├── postgres pod          ← official image, pulled from Docker Hub
├── mqtt pod              ← official image, pulled from Docker Hub
├── minio pod             ← official image, pulled from Docker Hub
├── prometheus pod        ← official image, pulled from Docker Hub
├── grafana pod           ← official image, pulled from Docker Hub
├── mlflow pod            ← OUR image, pulled from FOCUS registry
├── inference_server pod  ← OUR image, pulled from FOCUS registry
└── fall_dashboard pod    ← OUR image, pulled from FOCUS registry`

All eight pods run inside FOCUS's K8s cluster. The Helm chart just declares them as Deployments/StatefulSets and points each one at the right image. K8s itself handles the pulls.

**Corrected version of my earlier statement:** "the three pieces we package ourselves (inference_server, fall_dashboard, mlflow) need their Docker images built and pushed before deployment. Step 12 specifically verifies the two that contain our application code — the third is too thin to need explicit verification beyond `docker-compose up`, which you already ran successfully."