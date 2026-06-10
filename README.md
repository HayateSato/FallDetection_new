# Fall Detection System

Real-time fall detection using XGBoost + SmarKo wearable sensor. The repository contains
multiple integration variants at different stages. Most readers want the **active v3
production system** — start with the table below to find your folder.

---

## Start here — pick your role

| You are... | Read first | Then |
|------------|-----------|------|
| **Mohammed** — deploying MCS inference layer to Hetzner | [`mohammed/mohammed_handover.md`](mohammed/mohammed_handover.md) | [`_6G_integration_v3_docker_mcs/README.md`](_6G_integration_v3_docker_mcs/README.md) → [`FOCUS_devs_handover/production_config_checklist.md`](FOCUS_devs_handover/production_config_checklist.md) |
| **Isa** — mobile-app developer (React Native) | [`isa/00_ISA_local_setup_quickstart.md`](isa/00_ISA_local_setup_quickstart.md) | [`isa/01_ISA_mobile_app_contract.md`](isa/01_ISA_mobile_app_contract.md) |
| **FOCUS DevOps** — deploying caregiver layer to k3s | [`FOCUS_devs_handover/focus_devops_handover.md`](FOCUS_devs_handover/focus_devops_handover.md) | [`FOCUS_devs_handover/production_config_checklist.md`](FOCUS_devs_handover/production_config_checklist.md) → [`_6G_integration_v3_k3s/`](_6G_integration_v3_k3s/) Helm chart |
| **New to project** — want the active production system | this README (sections below) | [`_6G_integration_v3_docker_mcs/README.md`](_6G_integration_v3_docker_mcs/README.md) |
| Working on the legacy full-research stack | [`Docu/HOW_TO_RUN.md`](Docu/HOW_TO_RUN.md) | [`_6G_Integration_v2_redis/README.md`](_6G_Integration_v2_redis/README.md) |

---

## Which version are you looking at?

This repository contains **multiple integration variants** across different folders.

| Folder / branch | Status | Event bus | Use |
|-----------------|--------|:---------:|-----|
| **`_6G_integration_v3_docker_mcs/`** | **Active — MCS inference layer** | MQTT | Docker Compose, 8 services, deploy to Hetzner |
| **`_6G_integration_v3_k3s/`** | **Active — FOCUS caregiver layer** | MQTT | K3s Helm chart, 2 services, deploy to FOCUS cluster |
| `_6G_integration_v3_docker_focus/` | Backup / local test only | MQTT | Docker Compose version of the caregiver layer (two-laptop testing) |
| `_6G_Integration_v2_mqtt/` | **Legacy — frozen** | MQTT | Previous version where FOCUS hosted all 10 services in k3s |
| `_6G_Integration_v2_redis/` | **Legacy — frozen** | Redis | Older design before MQTT was introduced — do not run |
| `_EcoSystem_Integration/` on `6G-integration` branch | Active (internal) | none | Stripped-down internal company integration, no Docker |
| Repo root + `system_operator/ml_server/` | Active (research) | Redis | Full research stack — see [`_6G_Integration_v2_redis/README.md`](_6G_Integration_v2_redis/README.md) |

> **v2 folders are frozen reference only.** Do not run `_6G_Integration_v2_mqtt/` or
> `_6G_Integration_v2_redis/` — they reflect an older architecture where FOCUS hosted
> everything in one k3s cluster. The current split (MCS hosts inference on Hetzner,
> FOCUS hosts 2 caregiver services in their k3s) lives entirely in the `v3` folders.

---

## Active Production System (v3)

Two separate stacks. Each is deployed independently by a different team.

| Stack | Folder | Owner | Runtime | Services |
|-------|--------|-------|---------|----------|
| MCS inference layer | `_6G_integration_v3_docker_mcs/` | Mohammed / MCS | Docker Compose on Hetzner | 8 services |
| FOCUS caregiver layer | `_6G_integration_v3_k3s/` | FOCUS DevOps | K3s Helm chart in FOCUS cluster | 2 services |
| Caregiver layer (local test) | `_6G_integration_v3_docker_focus/` | Hayate (testing) | Docker Compose on laptop | 2 services (backup) |

### MCS Inference Layer services

| Service | Port (host) | Purpose |
|---------|------------|---------|
| `inference-server` | **8001** (public via Nginx) | Main API — `/predict` + `/inference/{id}/confirm`. Runs XGBoost fall detection. |
| `ml-dashboard` | 8004 (internal) | Admin UI — trigger retraining, inspect MLflow runs, hot-swap active model |
| `server-health` | 8006 (internal) | Aggregate health dashboard — probes all 6 services + FOCUS caregiver layer |
| `postgres` | 5432 (internal) | `inference_log`, `feature_snapshot`, MLflow tracking DB |
| `mlflow` | 5000 (internal) | ML experiment tracking + model registry |
| `minio` | 9000/9002 (internal) | S3-compatible store for MLflow model artifacts |
| `prometheus` | 9090 (internal) | Scrapes metrics from inference-server |
| `grafana` | 3000 (internal) | ML dashboards — confidence drift, fall counts, inference rate |
| `db-migrate` | — | One-off: runs Alembic migrations on startup, then exits |
| `minio-setup` | — | One-off: creates `mlflow-artifacts` bucket, then exits |

### FOCUS Caregiver Layer services (K3s)

| Service | Pod internal port | Local test (NodePort) | Production (Traefik) | Purpose |
|---------|:-----------------:|----------------------|---------------------|---------|
| `mosquitto` | 1883 TCP / 9001 WS | NodePort **30901** (WS) | **443** WSS via Traefik → :9001 | MQTT broker. Mobile app connects via WebSocket. fall-dashboard connects via internal TCP :1883. |
| `fall-dashboard` | 8002 | NodePort **30802** | **443** HTTPS via Traefik → :8002 | Caregiver alert dashboard. Subscribes to MQTT. Fans out live alerts via SSE. Shows fall history from FOCUS InfluxDB. Patient add/delete via UI. |

> Port 1883 (raw TCP) is cluster-internal only and is never reachable from outside.
> All external MQTT connections (mobile app, local mock-app on a different machine) must use WebSocket
> because React Native cannot open raw TCP sockets. See [`__Refactoring_docs/port_management.md`](__Refactoring_docs/port_management.md) for the full per-scenario port breakdown.

---

## Data Flow (v3 Active System)

```
SmarKo Wearable
  │  Bluetooth
  ▼
Isa's Mobile App (React Native)
  │
  ├─ HTTPS POST /predict + X-API-Key ──────────────────────────────────┐
  │                                                                     ▼
  │                                            ┌───────────────────────────────────────┐
  │                                            │  inference-server :8001 (Hetzner)     │
  │                                            │                                       │
  │                                            │  1. resample ACC (50 Hz)              │
  │                                            │  2. compose 9s window (450 samples)   │
  │                                            │  3. extract 16-22 features            │
  │                                            │  4. XGBoost.predict_proba             │
  │                                            │     → { fall_detected, confidence,    │
  │                                            │         observation_id }              │
  │                                            │                                       │
  │                                            │  After prediction:                    │
  │                                            │  A → postgres: inference_log +        │
  │                                            │       feature_snapshot                │
  │                                            │  B → prometheus: /metrics counters    │
  │                                            └───────────────────────────────────────┘
  │                                                 │ A               │ B
  │                                                 ▼                 ▼
  │                                            PostgreSQL        Prometheus
  │                                            :5432             :9090
  │                                                 │                 │ scrapes
  │                                                 ▼                 ▼
  │                                            ml-dashboard      Grafana :3000
  │                                            :8004             (ML dashboards)
  │                                            (retrain,
  │                                             hot-swap)
  │
  │  (if fall detected — patient sees popup on phone)
  │
  ├─ HTTPS POST /inference/{id}/confirm ───────────► inference-server
  │    (patient response: confirmed/denied/no_answer)  writes patient_confirmed +
  │                                                    needs_help to inference_log
  │
  ├─ MQTT WebSocket wss://<FOCUS-domain>:443 ──────────────────────────────────────┐
  │    PUBLISH fall/possible/<pid>  (immediate, before confirmation)                ▼
  │    PUBLISH fall/alert/<pid>     (after confirmation or 10s timeout)    ┌────────────────┐
  │                                                                         │  mosquitto     │
  │                                                                         │  K3s :9001 WS  │
  │                                                                         │  :1883 TCP     │
  │                                                                         └───────┬────────┘
  │                                                                                 │ subscribes (TCP :1883)
  │                                                                                 ▼
  │                                                                         ┌────────────────┐
  └─ InfluxDB write (fall_events point) ──► FOCUS InfluxDB (direct)        │ fall-dashboard │
       written after confirmation popup                                     │ :8002          │
                                           ▲                               │                │
                                           │ queries fall history          │ SSE stream ──► caregiver
                                           └───────────────────────────────┤               browser
                                                                            └────────────────┘
```

### MQTT payload (both topics)

```json
{
  "patient_id":       "patient_test_50",
  "observation_id":   "uuid-...",
  "fall_detected":    true,
  "confidence":       0.91,
  "patient_confirmed": 1,
  "needs_help":       true,
  "timestamp":        "2026-06-09T12:00:00+00:00",
  "status":           "confirmed"
}
```

`patient_confirmed` encoding: `1` = confirmed fall, `0` = denied, `-1` = no answer within 10s.

---

## PostgreSQL Tables (v3 MCS Inference Layer)

All tables live in the `fall_detection` database on the MCS Postgres container.
Managed by Alembic (`db-migrate` init job applies all migrations on startup).

| Table | Written by | Read by | What is stored |
|-------|-----------|---------|----------------|
| `inference_log` | inference-server (every `/predict`) | ml-dashboard, Grafana | One row per prediction: `timestamp`, `model_version`, `fall_detected`, `confidence`, `window_size`, `latency_ms`, `participant_id`, `patient_confirmed` (1=yes / 0=no / -1=no_answer / NULL=pending), `needs_help` (same encoding / NULL=pending), `observation_id` (UUID) |
| `feature_snapshot` | inference-server (every `/predict`) | ml-dashboard (retraining pipeline) | One row per feature per prediction: `inference_id` (FK → inference_log), `feature_name`, `feature_value`. Stores the full 16–22 feature vector so any past prediction can be replayed. |

**Key relationship:** `feature_snapshot.inference_id` → `inference_log.id`

**Removed tables (v2 → v3):**
- `fall_history` — dropped in migration 0003; `patient_confirmed` + `needs_help` moved directly into `inference_log`
- `participant_session` — dropped in migration 0005; no longer used in the MCS layer

MLflow tracking data lives in a separate `mlflow` database on the same Postgres instance.

---

## MQTT Topics

Two topics carry fall events. Both are published by the **mobile app** (not by inference-server).

| Topic | When published | Who subscribes |
|-------|---------------|---------------|
| `fall/possible/<patient_id>` | Immediately after `/predict` returns `fall_detected=true`, before patient confirmation | fall-dashboard (shows subtle "possible fall" notice on caregiver UI) |
| `fall/alert/<patient_id>` | After patient confirms fall via popup, OR after 10s timeout with no response | fall-dashboard (shows full alert + SSE fan-out to caregiver browser) |

Inference-server has **no MQTT client**. The mobile app owns both publish steps.

