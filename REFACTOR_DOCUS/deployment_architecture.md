# Deployment Architecture — Fall Detection / FOCUS Integration
**Last updated:** 2026-04-15  
**Status:** Implementation in progress — see todo.md for open items

---

## Two Kubernetes Namespaces

The SmarKo wearable and mobile app are **external to both namespaces** — the wearable
is a physical device and the app runs on the patient's phone. They communicate into
the cluster over the network.

> **Local development note:** `mock_app` plays the role of the mobile app in this diagram.
> Our cloud InfluxDB currently plays the role of the InfluxDB instance that would live in
> the FOCUS namespace. See "Future: Mock FOCUS Namespace" section below.
>
> **Important:** In production, the mobile app reads from the SmarKo wearable via BLE
> and sends ACC windows directly to `inference_server` — **InfluxDB is not in the
> inference path**. `mock_app` reads from InfluxDB as a local-dev shortcut only because
> we do not have the real SmarKo app. InfluxDB's role is biosignal display in the
> Patient Dashboard, not inference input.

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
│  [our cloud InfluxDB │     │                                                      │
│   acts as this now]  │     │  inference_server  :8001                             │
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

> **Local dev substitute:** Our cloud InfluxDB instance (`fd_test` bucket) plays the role
> of the FOCUS-hosted InfluxDB. No code change needed — only the `INFLUXDB_URL` in `.env`
> will change when pointing at the real FOCUS instance.

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
| `MinIO` | Artifact store for MLflow model `.pkl` files; needed when inference server loads from registry | Pending |

---

## Data Sources — Summary Table

| Storage | Type | What is stored | Who writes | Who reads | Purpose |
|---------|------|---------------|-----------|-----------|---------|
| **InfluxDB** (FOCUS-hosted; our cloud instance for local dev) | Time-series | Raw ACC + barometer from SmarKo wearable; biosignals (HR etc.); fall results + patient confirmation written by mobile app | Mobile app (Isa) writes · `mock_app` writes (local dev only) | **Patient Dashboard** (biosignal display panel) · `mock_app` reads as a local-dev shortcut (replaces BLE wearable input) | **Biosignal display in Patient Dashboard only. NOT used for inference in production.** In production the mobile app reads from the wearable via BLE and sends windows directly to inference_server. `seed_test_data --influxdb` reads from here as a dev utility to populate Postgres without a live inference_server. |
| **FHIR Server** (FOCUS-hosted) | FHIR R4 | Patient demographics, identifiers, Patient resources | FOCUS | Our dashboard (read-only, patient info panel) | Patient identity and medical context |
| **Postgres `fall_detection` DB** (our namespace) | Relational | `inference_log`: every prediction (patient_id, model_version, fall_detected, confidence, latency_ms, detection_time) · `feature_snapshot`: feature name+value per inference · `fall_history`: confirmed alerts (linked to inference_log via FK, patient_confirmed, needs_help) · `participant_session`: fall count per patient | `inference_server` writes inference_log + feature_snapshot · `fall_dashboard` writes fall_history + participant_session | Caregiver dashboard (fall history) · `retrain.py` (labelled training data) | Fall history for dashboard display; labelled dataset for model retraining |
| **Postgres `mlflow` DB** (our namespace) | Relational | MLflow runs, parameters, metrics, model registry stages | MLflow tracking server | MLflow UI · inference_server (`/model/switch` loads from registry) | Experiment tracking and model versioning |
| **MinIO** (our namespace) | Object store | Trained model `.pkl` artifacts, per MLflow run | MLflow (via `retrain.py`) | `inference_server` (loads model from registry) | Model artifact storage for retraining pipeline |

### InfluxDB migration plan

Currently (local dev + testing): using our own cloud InfluxDB instance (`fd_test` bucket, existing credentials in `.env`).

Eventually (production Helm deployment): package a new InfluxDB instance inside our Helm namespace so the system is self-contained. The mobile app would then write to this instance rather than the FOCUS-hosted one. This needs to be agreed with FOCUS — either they keep hosting InfluxDB in their namespace, or we bring our own.

**Coordination needed with FOCUS before deciding:**
- Should InfluxDB live in FOCUS namespace (they operate it) or our namespace (we operate it)?
- If FOCUS-hosted: our mock_app/data fetcher reads across namespace boundary (service URL config only, no code change)
- If our namespace: we include InfluxDB as a `StatefulSet` in our Helm chart

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

## Open Decisions Still Needed

| Decision | Who decides | Impact |
|----------|-------------|--------|
| FHIR output required? | FOCUS | Whether `FHIR_SERVER_URL` is used in production |
| InfluxDB: FOCUS-hosted or our namespace? | FOCUS + us | Whether InfluxDB is in our Helm chart |
| Container registry location | FOCUS DevOps | Step 9 Helm chart delivery |
| Kubernetes namespace names | FOCUS DevOps | Helm chart `values.yaml` |
| Model files: Docker image or mounted volume? | FOCUS DevOps + us | Step 7.3 |
| Data sharing agreement with Charite | Charite + FOCUS | Unlocks Step 11 (MLflow retraining) |
