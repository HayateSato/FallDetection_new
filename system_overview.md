# System Overview — Fall Detection (All Three Layers)

The system spans two networks and three functional layers.

## Hosting legend

| Code | Meaning |
|------|---------|
| **F1** | K3s service already running in FOCUS network |
| **F2** | K3s service to be added in FOCUS network |
| **F3** | Other service to be added in FOCUS network (not K3s-managed) |
| **M** | MCS network — deployed in Docker on Hetzner |

---

## High-level picture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  FOCUS Network                                                          │
│                                                                         │
│  ┌──────────────────────────────┐   ┌───────────────────────────────┐  │
│  │  1. Patient Layer            │   │  2. Caregiver Layer           │  │
│  │  (end-user interface)        │   │  (backend + frontend)         │  │
│  │                              │   │                               │  │
│  │  SmarKo             (F3)     │   │  BACKEND                      │  │
│  │  Mobile app         (F3)     │   │    InfluxDB          (F1)     │  │
│  │  MQTT Client A      (F3)  ───┼───┼──► MQTT broker        (F2)    │  │
│  │                              │   │    MQTT Client B      (F2)    │  │
│  │                              │   │    Caregiver webapp   (F1)    │  │
│  │                              │   │    Data table A       (F2)    │  │
│  └──────────────────────────────┘   │  FRONTEND                     │  │
│                                     │    Patient dashboard  (F1)    │  │
│              │ HTTPS /predict        │    Fall dashboard     (F2)    │  │
│              │ HTTPS /confirm        └───────────────────────────────┘  │
└──────────────┼──────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MCS Network (Hetzner)                                                  │
│                                                                         │
│  3. Inference & Post-training Layer                                     │
│  (backend + frontend)                                                   │
│                                                                         │
│  BACKEND                                                                │
│    Inference server           (M)   Postgres table B         (M)       │
│    Postgres table C           (M)   MinIO                    (M)       │
│    MLflow                     (M)   Prometheus               (M)       │
│    Grafana                    (M)   Post-training webapp      (M)       │
│  FRONTEND                                                               │
│    ML dashboard               (M)   Server health dashboard  (M)       │
│    Grafana dashboard          (M)                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Patient Layer — end-user interface (FOCUS network)

| Component | Hosted | Description |
|-----------|:------:|-------------|
| **SmarKo** | F3 | Wearable sensor. Collects physiological data (ACC, barometer, HR, SpO2) and streams it to the mobile app via Bluetooth. |
| **Mobile app** | F3 | Central orchestrator for the patient side. Transmits raw sensor data to FOCUS InfluxDB. Sends POST `/predict` to the inference server to run fall detection. Displays fall confirmation popup to the patient. Injects a fall event marker into InfluxDB after the patient responds. |
| **MQTT Client A** | F3 | Embedded in the mobile app. Publishes a `fall/possible/<pid>` event immediately when a fall is detected, and a `fall/alert/<pid>` event after the patient confirms (or after the 10s timeout). Connects to the MQTT broker via WebSocket (WSS). |

---

## 2. Caregiver Layer — backend + frontend (FOCUS network)

### Backend

| Component | Hosted | Description |
|-----------|:------:|-------------|
| **InfluxDB** | F1 | Stores physiological time-series data (HR, SpO2, ACC, etc.) per patient ID, plus `fall_events` measurement (fall timestamps injected by the mobile app). |
| **Caregiver webapp server** | F1 | Backend for the patient dashboard. Reads patient demographics and real-time biosignals from InfluxDB. Serves the Flutter-based patient dashboard frontend. |
| **Data table A** | F2 | SQLite file (on K3s PVC) inside the fall-dashboard pod. Stores the patient list (patient ID, MAC address). Managed dynamically via the fall-dashboard UI — no values.yaml edit or pod restart needed. |
| **MQTT broker** | F2 | Eclipse Mosquitto. Two listeners: `:1883` TCP (cluster-internal, used by MQTT Client B) and `:9001` WebSocket (used by the mobile app externally via WSS on port 443). |
| **MQTT Client B** | F2 | Embedded in the fall-dashboard pod. Subscribes to `fall/possible/#` and `fall/alert/#` on the broker (:1883 internal). Forwards events to connected caregiver browsers via Server-Sent Events (SSE). |

### Frontend

| Component | Hosted | Description |
|-----------|:------:|-------------|
| **Patient dashboard** | F1 | Flutter web app. Displays registered patients and their personal information (gender, BMI, age, etc.) and live health monitoring status (HR, SpO2, etc.) sourced from InfluxDB. |
| **Fall dashboard** | F2 | Web UI served by the fall-dashboard pod. Shows all recorded falls, filterable by time period, patient, and whether help was requested. For real-time use: displays a "possible fall" notice as soon as the sensor detects a fall event, followed by the patient's confirmation status. |

---

## 3. Inference & Post-training Layer — backend + frontend (MCS network, Hetzner)

### Backend

| Component | Hosted | Description |
|-----------|:------:|-------------|
| **Inference server** | M | FastAPI. Main fall-detection API. Receives `/predict` from the mobile app (HTTPS), runs the XGBoost model on the 9-second sensor window, returns `{ fall_detected, confidence, observation_id }`. Also receives `/inference/{id}/confirm` with the patient's response. |
| **Postgres table B** | M | `inference_log` table. One row per `/predict` call. Stores timestamp, model version, fall_detected, confidence, patient_confirmed (1=yes / 0=no / -1=no_answer), needs_help, observation_id. |
| **Postgres table C** | M | `feature_snapshot` table. One row per feature per prediction. Stores the full 16–22 feature vector so any past prediction can be replayed for retraining. |
| **MinIO** | M | S3-compatible object store. Stores model artifact files (`.ubj` XGBoost binaries) referenced by MLflow. Port 9000 (S3 API), 9002 (admin console). |
| **MLflow** | M | ML experiment tracking and model registry. Records hyperparameters, metrics, and run history for each training job. Manages model versions and stages (Staging / Production). |
| **Prometheus** | M | Scrapes metrics from inference-server (`/metrics`) every 15s. Tracks inference rate, fall detection counts, model confidence distribution, latency. |
| **Grafana** | M | Reads from Prometheus and Postgres. Visualises ML health: inference rate over time, confidence drift, fall count trends. Internal MCS use only. |
| **Post-training webapp server** | M | FastAPI backend for the ML dashboard. Triggers model retraining from `feature_snapshot`, registers new runs in MLflow, and issues hot-swap commands to the inference server without a restart. |

### Frontend

| Component | Hosted | Description |
|-----------|:------:|-------------|
| **ML dashboard** | M | Web UI for MCS admins. Trigger retraining, compare MLflow runs, promote a model to Production, and hot-swap the active model on the running inference server. |
| **Server health dashboard** | M | Web UI showing a status card for each of the 6 MCS services + 2 FOCUS caregiver probes (fall-dashboard, MQTT broker). Refreshes every 30s. |
| **Grafana dashboard** | M | Pre-built dashboards for ML model performance and inference-server metrics. Accessible at Grafana port 3000 (internal). |

---

## Cross-network data flows

| Flow | Path | Protocol |
|------|------|----------|
| Raw sensor data → storage | Mobile app → FOCUS InfluxDB | HTTPS (direct write) |
| Fall detection | Mobile app → Inference server (Hetzner) | HTTPS POST `/predict` |
| Patient confirmation | Mobile app → Inference server (Hetzner) | HTTPS POST `/inference/{id}/confirm` |
| Fall event marker | Mobile app → FOCUS InfluxDB | HTTPS (direct write) |
| Real-time fall alert | Mobile app → MQTT broker (FOCUS K3s) | WebSocket (WSS :443) |
| Alert fan-out to browser | MQTT Client B → Caregiver browser | SSE (Server-Sent Events) |
| Health probing | Server-health (MCS) → fall-dashboard + MQTT broker (FOCUS) | HTTPS / TCP |
