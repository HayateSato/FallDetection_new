# Data Storage Map — what is stored where

**Audience:** admin / MLOps operator on the running stack.
**Use this doc when:** you need to know which service owns which piece of data, or you're debugging "where did this row come from?" / "why is this metric missing?"

---

## At a glance — six data stores

| Store | Type | What lives in it | Persists? | Used by |
|-------|------|------------------|-----------|---------|
| **Postgres `fall_detection` DB** | Relational | Inference + fall history (operational data) | yes — disk volume | inference_server, fall_dashboard, retrain |
| **Postgres `mlflow` DB** | Relational | MLflow's internal tables (experiments, registry) | yes — same volume | mlflow tracking server only |
| **MinIO** | S3 object store | Trained model `.pkl` artifacts | yes — disk volume | mlflow + inference_server |
| **MQTT broker (Mosquitto)** | Message broker | In-flight `fall/alert` messages | transient (in memory) | mock_app → fall_dashboard |
| **Prometheus** | Time-series DB | Scraped `/metrics` from inference_server | yes — 30 days retention | Grafana queries this |
| **Grafana internal SQLite** | Relational | Dashboards, datasource configs, user accounts | yes — volume | Grafana itself |
| **InfluxDB** (FOCUS-hosted, external) | Time-series | Biosignals (HR, raw sensor) | yes — FOCUS-managed | Patient Dashboard biosignal panel only |
| **Our cloud InfluxDB** (`ecosystem-influxdb.smarko-health.de`) | Time-series | Historical ACC windows | yes — MCS/SmarKo-managed | mock_app local-dev only — NOT production |

There is **one** Postgres instance running two logical databases. There is **no other relational DB** in the stack — caregiver.db (SQLite) only exists if you've explicitly switched DATABASE_URL back to SQLite for zero-Docker dev.

---

## Postgres — `fall_detection` database

Schema managed by Alembic (`shared_db/db/migrations/versions/`).

### `inference_log`

One row per `/predict` call.

| Column | Type | Source |
|--------|------|--------|
| `id` | SERIAL PK | auto |
| `observation_id` | VARCHAR(36) UNIQUE | UUID generated at start of /predict |
| `patient_id` | VARCHAR(100) | from /predict request body |
| `device_id` | VARCHAR(100) | from /predict request body |
| `model_version` | VARCHAR(64) | currently-loaded model name (e.g. `mlflow:Production:v3(v0)`) |
| `fall_detected` | BOOLEAN | XGBoost output |
| `confidence` | FLOAT | XGBoost score 0.0–1.0 |
| `window_size` | INT | number of ACC samples processed |
| `latency_ms` | INT | end-to-end inference time |
| `detection_time` | TIMESTAMPTZ | server time when /predict received |

**Written by:** `inference_server` via a FastAPI BackgroundTask, after the HTTP response is sent. Failures here are logged, not raised — they never surface to the mobile app.

**Read by:**
- `retrain.data_pipeline` (joins this with feature_snapshot + fall_history to build the labelled dataset)
- Grafana SQL dashboards (per-patient activity, per-model statistics)

---

### `feature_snapshot`

One row per feature per inference. Long format. `inference_log` typically has 16 or 20 features so each `/predict` writes 16–20 rows here.

| Column | Type | Source |
|--------|------|--------|
| `id` | SERIAL PK | auto |
| `inference_id` | INT FK → inference_log.id | links back to the call |
| `feature_name` | VARCHAR(50) | e.g. `acc_x_mean`, `acc_y_std` |
| `feature_value` | FLOAT | the computed value |

**Written by:** same BackgroundTask in `inference_server`.

**Read by:** `retrain.data_pipeline` — pivots the long format to wide format, one column per feature, before training XGBoost.

**Why long format:** lets us add new features without an Alembic migration. Each new feature is just more rows.

---

### `fall_history`

One row per confirmed alert (after the patient confirmation popup). Only written when fall_detected=True AND mock_app/mobile_app published the MQTT alert.

| Column | Type | Source |
|--------|------|--------|
| `id` | SERIAL PK | auto |
| `observation_id` | VARCHAR(36) FK → inference_log.observation_id | UUID — links back to the inference |
| `patient_id` | VARCHAR(100) | from MQTT payload |
| `fall_detected` | BOOLEAN | always TRUE (only fall alerts are published) |
| `patient_confirmed` | VARCHAR(20) | `"yes"` / `"no"` / `"not_answered"` |
| `needs_help` | BOOLEAN | from patient popup |
| `detection_time` | TIMESTAMPTZ | when fall was detected |
| `alert_time` | TIMESTAMPTZ | when MQTT alert arrived |

**Written by:** `fall_dashboard` on MQTT receipt of `fall/alert/<patient_id>`.

**Read by:**
- `retrain.data_pipeline` (joins this with inference_log + feature_snapshot for training labels)
- Caregiver dashboard `GET /api/falls`
- ml_dashboard (indirectly, through retraining)

**The `observation_id` UUID** is the cross-reference key. It's generated once in `inference_server`, returned in the HTTP response, carried through MQTT by mock_app, and stored here. This is what links a patient's response back to the original inference for retraining.

---

### `participant_session`

One row per registered patient. Written once per startup so the dashboard can show patients who haven't generated falls yet.

| Column | Type | Source |
|--------|------|--------|
| `id` | SERIAL PK | auto |
| `participant_name` | VARCHAR(100) | patient_id from .env |
| `gender` | VARCHAR(10) | currently always NULL |
| `start_time` | TIMESTAMPTZ | session start |
| `end_time` | TIMESTAMPTZ | currently always NULL |
| `fall_count` | INT | not actively maintained — fall counts are computed from fall_history |

**Written by:** `fall_dashboard.main._start_mqtt_with_callback` — calls `cdb.ensure_session(pid)` for each `PATIENT_IDS` entry on startup.

**Read by:** `GET /api/patients` (caregiver dashboard).

**Note:** the `fall_count` column is legacy — current code computes counts on the fly from `fall_history`. The column stays in case future analytics need it.

---

### `alembic_version`

| Column | Source |
|--------|--------|
| `version_num` | written by `alembic upgrade head` |

**Written by:** Alembic on migration. **Read by:** Alembic to determine which migrations still need to run.

Current value should be `0002` after the migration applied 2026-04-28 (widened model_version to 64 chars).

---

## Postgres — `mlflow` database

Created by `infrastructure/postgres/init.sql`. Owned and managed entirely by the MLflow tracking server. Tables are created automatically on first MLflow startup via MLflow's own Alembic migrations.

| What | Stored where |
|------|--------------|
| Experiments (e.g. `fall-detection-v0`) | `experiments` table |
| Runs (one per `retrain.retrain` call) | `runs` table |
| Params (n_train, threshold, ...) | `params` table |
| Metrics (recall, AUC, F1, ...) | `metrics` table |
| Tags (dataset, feature_set, ...) | `tags` table |
| Registered models (`fall-detection-xgboost`) | `registered_models` table |
| Model versions (v1, v2, ...) | `model_versions` table |
| Aliases (`Production`, `Staging`) | `registered_model_aliases` table |

**Do not touch this database directly with SQL.** Always go through the MLflow API or UI. Direct SQL writes can break MLflow's internal consistency.

**Read/written by:** `mlflow` tracking server pod only. `retrain.py` and `inference_server` talk to MLflow over HTTP (`MLFLOW_TRACKING_URI`) — they never connect to this database directly.

---

## MinIO

S3-compatible object store running at `http://localhost:9000` (API) and `http://localhost:9002` (web console).

**One bucket: `mlflow-artifacts`**

```
s3://mlflow-artifacts/
  └── 1/                                   ← experiment ID
      ├── <run-uuid-1>/
      │   └── artifacts/
      │       └── model/
      │           ├── model.pkl            ← the trained XGBoost binary
      │           ├── conda.yaml
      │           └── MLmodel
      ├── <run-uuid-2>/
      │   └── ...
      └── ...
```

**Written by:** MLflow tracking server (when `retrain.py` calls `mlflow.xgboost.log_model(...)`).

**Read by:** `inference_server` when `POST /model/switch {"mlflow_stage": "Production"}` is called — MLflow returns the artifact URI, inference_server downloads the `.pkl` from MinIO into memory.

**Persistence:** Docker volume `minio_data`. Data survives container restarts. `docker-compose down -v` deletes it (don't run that in production).

---

## MQTT broker (Eclipse Mosquitto)

**Mostly transient.** Messages flow through; broker isn't a long-term store.

| Aspect | Behaviour |
|--------|-----------|
| Routing | `fall/alert/<patient_id>` from publisher → all subscribers of `fall/alert/#` |
| Persistence | Volume `mqtt_data` retains retained messages and persistent sessions across restarts |
| Logs | Volume `mqtt_logs` for broker logs |
| Authentication | None — any client can publish/subscribe |

**Why no persistence of normal messages:** MQTT messages are not retained by default. If `fall_dashboard` is offline when an alert is published, that alert is lost. This is acceptable in our setup because `fall_history` (in Postgres) is the durable record — `fall_dashboard` writes to it as soon as it receives an alert.

**Failure mode:** if `fall_dashboard` crashes for an extended period, alerts published during the outage do not appear in `fall_history`. The caregiver dashboard is missing those events. The `inference_log` row exists, but no `fall_history` row links to it. From the retraining perspective these become "ambiguous" rows (`patient_confirmed=not_answered` from the JOIN side).

---

## Prometheus

Time-series database for metrics scraped from `inference_server`'s `/metrics` endpoint every 15 seconds.

**What is stored:**

| Metric name | Type | Meaning |
|-------------|------|---------|
| `fall_detections_total` | counter | total /predict calls split by patient_id, model_version, fall_detected |
| `inference_latency_seconds` | histogram | latency distribution per model version |
| `model_confidence` | histogram | confidence score distribution per model version |
| (plus FastAPI/uvicorn defaults — request count, response time per route) | various | standard Prometheus client metrics |

**Persistence:** Docker volume `prometheus_data`. Default retention **30 days** (set in `docker-compose.yml` via `--storage.tsdb.retention.time=30d`).

**Read by:** Grafana datasource only. No other component queries Prometheus.

**Lost on:** `docker-compose down -v` deletes the volume. Container restart preserves it.

---

## Grafana — internal SQLite

Grafana stores its **own configuration** (not the metrics it visualises) in an internal SQLite database mounted at `/var/lib/grafana/grafana.db` inside the container.

**What is stored:**

| What | Where |
|------|-------|
| Dashboard JSON definitions (3 dashboards: ml_server_overview, model_performance, fall_events_timeline) | `grafana.db` |
| Datasource configs (Postgres, Prometheus connections) | `grafana.db` |
| User accounts, roles, sessions | `grafana.db` |
| Alert rules (none configured currently) | `grafana.db` |

**Provisioning:** `infrastructure/grafana/provisioning/` mounts datasource and dashboard YAML at startup, so the configs are recreated even if the volume is wiped. The provisioned configs are read-only in the UI.

**Persistence:** Docker volume `grafana_data`.

**Important:** Grafana does **not** store the metrics or query results themselves. It queries Prometheus / Postgres on every page load. If you want historical data it has to be in Prometheus or Postgres.

---

## InfluxDB — two instances, both external to our stack

Neither instance is part of our Helm chart. Both clarified previously, recapped here:

### FOCUS-hosted InfluxDB

- Lives in **FOCUS namespace**, run by FOCUS
- Stores biosignals (HR, SpO2, raw ACC) written by the real mobile app
- Read by the Patient Dashboard's biosignal panel (Isa's UI)
- **Our inference pipeline never touches it** — neither read nor write

### Our cloud InfluxDB (`ecosystem-influxdb.smarko-health.de`)

- MCS/SmarKo cloud, runs outside any Kubernetes cluster
- Holds historical ACC windows we use as a fake BLE source for `local_dev/mock_app`
- **Disappears in production** — the real mobile app reads BLE directly, no InfluxDB query happens anywhere in the inference path

If you're an admin in production troubleshooting "where did this prediction's data come from?", the answer is: **the mobile app POSTed it** — there is no InfluxDB step.

---

## Quick lookup — "where would I find ___?"

| Question | Answer |
|----------|--------|
| Last 5 predictions | Postgres `fall_detection.inference_log` |
| What features the model saw on row 1234 | Postgres `feature_snapshot WHERE inference_id = 1234` |
| Confirmed falls today | Postgres `fall_history WHERE detection_time > today` |
| Which patients are registered | Postgres `participant_session` |
| Model registry — current Production version | Postgres `mlflow` DB OR via `MLflow UI` (recommended) |
| The actual `.pkl` file for a model version | MinIO `mlflow-artifacts/<experiment-id>/<run-uuid>/artifacts/model/model.pkl` |
| Latency over the last hour | Prometheus (queried via Grafana) |
| Fall rate per hour | Prometheus (queried via Grafana) |
| MQTT message history | **NOT STORED** — only the live message gets routed. Look at `fall_history` instead. |
| Caregiver dashboard configs | Grafana `grafana.db` (UI customisations + provisioned defaults) |

---

## Recovery — what can be reconstructed if a store is lost?

| Lost store | Recoverable? | How |
|-----------|:------------:|-----|
| `fall_detection.inference_log` | partially | features cannot be recomputed without raw ACC; new predictions repopulate |
| `fall_detection.fall_history` | no | once an MQTT alert is gone, it's gone — only mobile app could re-publish |
| `mlflow` DB | partially | runs and metrics are lost; **artifacts in MinIO survive** so models can be re-registered manually |
| MinIO bucket | no | model `.pkl` files are unique per training run — if lost, retrain on existing Postgres data to recreate |
| Prometheus | no | scraping resumes from "now"; retroactive metrics impossible |
| Grafana SQLite | yes | provisioned configs recreate dashboards/datasources on container restart |

The most critical store is **`fall_history`** — it's the only place where ground-truth labels (`patient_confirmed`) live. Back this up first if you're choosing what to protect.

The second most critical is **MinIO** — losing it means retraining from scratch, but the data to do so is still in Postgres.

The least critical is **Prometheus** — it's a metrics view, easily reconstructed from new traffic.
