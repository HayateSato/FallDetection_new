# Fall Detection Algorithm Handover

**Audience:** the engineer or data scientist who will own the model after handover — retraining, monitoring, version promotion. Typically the MCS / FOCUS ML admin.
**Repo / branch:** `_6G_Integration_v2_mqtt/` on branch `6G-integration_with_MQTT`.
**Scope:**
- What the model is (input → output)
- How it was trained
- How to monitor its health and when to retrain
- How to manage model versions (registry, hot-swap, rollback)

This doc does **not** cover the K8s deployment (see `01_k8s.md`) or the system-level dataflow (see `03_fall_detection_system.md`). It's purely about the model.

---

## 1. The model in one paragraph

A binary XGBoost classifier that decides whether a 9-second window of accelerometer data (optionally + barometer) contains a fall. It's deterministic per input, runs in <50 ms on commodity CPU, and ships as a single `.pkl` file (~200 KB). There is no neural network, no GPU dependency, no online learning — every retrain produces a new `.pkl` that is hot-swapped into the running inference server.

---

## 2. Data input → model → output

### 2.1 Input (what `/predict` expects)

A single window of raw IMU data from a SmarKo wearable:

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `acc_x`, `acc_y`, `acc_z` | array of int | yes | **Raw LSB integers**, NOT converted to g. The server does the LSB→g conversion. |
| `timestamps_ms` | array of int (Unix ms) | yes | One per ACC sample. Same length as acc_x/y/z. |
| `pressure` | array of float (Pa) | optional | Only when the loaded model uses barometer. Check `GET /model/info → uses_barometer`. |
| `pressure_timestamps_ms` | array of int | required if `pressure` is set | One per pressure sample. |
| `patient_id` | string | yes | FHIR Patient identifier. |
| `device_id` | string | optional | Wearable MAC. Useful for debugging. |

**Window size:** 450 ACC samples = 9 seconds at 50 Hz. If the wearable is at a different rate, set `HARDWARE_ACC_SAMPLE_RATE` in `.env` and the server resamples up to 50 Hz internally. The barometer arrives at 25 Hz and is interpolated up.

### 2.2 What the server does between input and prediction

```
raw acc_x/y/z (LSB)
        │
        ▼
[LSB → g conversion]                     ml_pipeline/data_input/accelerometer_processor/
        │
        ▼
[Resample 50 Hz → 50 Hz (no-op)          ml_pipeline/data_input/accelerometer_processor/acc_resampler.py
 or hardware_rate → 50 Hz]               (skipped if HARDWARE_ACC_SAMPLE_RATE=50)
        │
        ▼
[Magnitude + paper-method ACC features]  magnitude_based_acc_processor_paper.py
        │
        ▼
[(Optional) Barometer EMA filter +       ml_pipeline/data_input/barometer_processor/
 slope-limit features]                   barometer_ema_filter.py + barometer_slope_limit_paper.py
        │
        ▼
[Feature vector (~20–40 features)]
        │
        ▼
[XGBoost model.predict_proba()]          ml_pipeline/core/inference_engine.py
        │
        ▼
{fall_detected: bool, confidence: float}
```

The feature names are explicit per model version (`model/model_<ver>/feature_names.json`). The server reads this on load — adding a feature in retraining is automatically reflected.

### 2.3 Output (what `/predict` returns)

```json
{
  "observation_id": "3a0a603e-cc5a-4355-ad8c-f53f2e4de1b9",
  "patient_id":     "charite-patient-001",
  "device_id":      "6c:1d:eb:04:a9:e6",
  "timestamp":      "2026-04-28T10:00:00+00:00",
  "inference": {
    "fall_detected": true,
    "confidence":    0.8234,
    "threshold":     0.5,
    "result":        "High confidence fall",
    "model_version": "v3",
    "window_size":   450
  },
  "fhir_observation": { /* FHIR R4 Observation */ },
  "fhir_pushed":     false
}
```

- `confidence` is `predict_proba` for the positive class.
- `threshold` is the cutoff used for the boolean decision (default 0.5; we did not tune this — see Section 3.3).
- `result` is a human-readable string.
- `fhir_observation` is the FHIR R4 Observation resource encoded with SNOMED 217082002 ("Fall (event)").
- `observation_id` is generated server-side per call. **Cross-references the inference to the patient confirmation written by `fall_dashboard`.** This is the linchpin of retraining — see Section 4.

---

## 3. Model versions in the repo

```
model/
├── model_v0/
│   ├── model.pkl             ← XGBoost
│   └── feature_names.json
├── model_v0_lsb_int/         ← v0 retrained on LSB-int input (no g conversion needed)
│   └── ...
├── model_v3/                 ← BEST overall (ACC + barometer)
│   └── ...
└── model_v5_lsb/             ← v0 + extra LSB-int features (experimental)
    └── ...
```

The active version is set by `MODEL_VERSION=v3` in `.env`. On startup the inference server loads `model/model_<MODEL_VERSION>/model.pkl`.

### 3.1 Performance (training-time, on labelled data)

| Version | Accuracy | Recall | Precision | F1 | AUC | Uses barometer |
|---------|---------:|-------:|----------:|---:|----:|:--------------:|
| v0      | ~0.91    | 0.86   | 0.84      | 0.85 | 0.94 | no |
| v0_lsb_int | ~0.90 | 0.85  | 0.84      | 0.84 | 0.94 | no |
| v3      | ~0.94    | 0.91   | 0.89      | 0.90 | 0.97 | **yes** |
| v5_lsb  | ~0.92    | 0.88   | 0.86      | 0.87 | 0.95 | no |

> **For safety-critical fall detection, prioritise recall over precision.** A missed fall is worse than a false alarm. Our default threshold (0.5) reflects that — we'd rather wake the patient unnecessarily than miss a real fall.

### 3.2 Choosing a version for production

`v3` is the default in production and what we recommend. The reason it's not "the only" version is that hospitals occasionally run wearables without a barometer (`v3` requires barometer; `v0` does not). If you find a deployment where the barometer is broken or absent, fall back to `v0_lsb_int`.

### 3.3 What we did NOT do (be aware)

- We did **not** tune the decision threshold per patient. The 0.5 cutoff is universal.
- We did **not** do per-patient personalisation (no fine-tuning on the patient's own falls). The retraining pipeline collects per-patient data but the resulting model is shared across all patients.
- We did **not** build an ensemble. Single XGBoost per version.

---

## 4. How the model was trained (and how to retrain)

### 4.1 Training data — origin

Two sources, mixed:

1. **Public fall-detection datasets** (the paper-based features come from this — see references in `ml_pipeline/data_input/accelerometer_processor/magnitude_based_acc_processor_paper.py`).
2. **Real Charite patient data** collected during the trial — the InfluxDB → Postgres pipeline below.

For ongoing retraining, the only data source you need is **Postgres**. InfluxDB is not in the retraining loop.

### 4.2 The retraining loop, end-to-end

```
[Patient wears wearable, falls or doesn't]
        │
        ▼
mobile app → POST /predict → inference_server
        │                          │
        │                          ├─→ writes inference_log (Postgres)        ← features computed at predict-time
        │                          └─→ writes feature_snapshot (Postgres)
        ▼
[Mobile app shows popup, patient confirms]
        │
        ▼
mobile app PUBLISH fall/alert/<patient_id> (MQTT)
        │
        ▼
fall_dashboard → writes fall_history (Postgres, with patient_confirmed + needs_help + observation_id)
        │
        ▼
[Time passes — accumulate labelled examples]
        │
        ▼
retrain/seed_test_data.py (optional — for filling gaps with synthetic data)
        │
        ▼
retrain/data_pipeline.py
   SQL JOIN inference_log + feature_snapshot + fall_history ON observation_id
        │
        ▼
[Labelled DataFrame: features + true label (patient_confirmed)]
        │
        ▼
retrain/retrain.py (train new XGBoost, log to MLflow, store .pkl in MinIO)
        │
        ▼
MLflow Registry: register model under name "fall_detector_v<n>"
        │
        ▼
[Promote stage: None → Staging → Production via MLflow UI or API]
        │
        ▼
inference_server reloads model on next /model/switch call (or pod restart)
```

### 4.3 Why the JOIN works (observation_id)

When `/predict` runs:
- A UUID `observation_id` is generated.
- `inference_log` row is written (BackgroundTask, not blocking) with that UUID.
- `feature_snapshot` rows are written (one per feature) with `inference_id` FK to inference_log.
- The UUID is returned to the mobile app in the HTTP response.

When the patient confirms:
- mobile app publishes the MQTT alert **including the observation_id**.
- `fall_dashboard` writes `fall_history` row with the same observation_id.

So the retraining query is just:

```sql
SELECT il.*, fs.feature_name, fs.feature_value, fh.patient_confirmed, fh.needs_help
FROM inference_log il
JOIN feature_snapshot fs ON fs.inference_id = il.id
JOIN fall_history fh     ON fh.observation_id = il.observation_id
WHERE il.fall_detected = TRUE AND fh.patient_confirmed IN ('yes','no');
```

`patient_confirmed='yes'` → label 1 (true positive).
`patient_confirmed='no'` → label 0 (false positive).
`patient_confirmed='not_answered'` → ambiguous, dropped from retraining.

This is why `observation_id` is **mandatory** in the MQTT payload from the mobile app — without it, the row in `fall_history` cannot be linked to features and the example is unusable.

### 4.4 How to run a retrain

```powershell
# From _6G_Integration_v2_mqtt/ as cwd

# 1. Install retraining deps (one-time)
pip install -r retrain/requirements.txt

# 2. Optional — if you don't have enough real data yet, seed synthetic
python -m retrain.seed_test_data --synthetic 100 --model-version v3

# 3. Optional — if you DO have real Charite data in InfluxDB and want to seed Postgres
python -m retrain.seed_test_data --influxdb

# 4. Dry-run the retraining (no MLflow write)
python -m retrain.retrain --dry-run

# 5. Real retrain — writes MLflow run + .pkl to MinIO
python -m retrain.retrain --model-version v3 --dataset our_data

# 6. Inspect run in MLflow UI
mlflow ui --backend-store-uri ./mlruns      # → http://localhost:5000
```

In production you can wrap step 5 as a Kubernetes CronJob if you want nightly retrains. We've kept it manual during the trial to give the operator full control.

### 4.5 What `retrain.py` actually does

1. Pulls labelled DataFrame from Postgres (4.3 query above).
2. Optionally mixes in the original public-dataset training set (`--dataset combined`).
3. Splits 80/20 train/test, stratified.
4. Trains XGBoost with default hyperparams (we did NOT do hyperparam search per retrain — extension point).
5. Computes test-set metrics: accuracy, recall, precision, F1, AUC, confusion matrix.
6. Logs to MLflow: params, metrics, confusion-matrix PNG, the `.pkl` artifact (stored in MinIO via `s3://mlflow-artifacts/`).
7. Prints a comparison vs the currently-Production model. If recall improved AND F1 didn't drop more than 2 pts, suggests promotion.

The promotion is **manual** — `retrain.py` never auto-promotes. You decide.

---

## 5. Health monitoring — how to know when to retrain

### 5.1 What we track in production

Three places where you can see model health:

| Where | What you see | URL |
|-------|--------------|-----|
| Prometheus (scrape `/metrics`) | request rate, latency, fall-rate, error rate, confidence histogram | `http://prometheus:9090` |
| Grafana — `ml_server_overview` | dashboard view of the above | `http://grafana:3000/d/ml_server_overview` |
| Grafana — `model_performance` | confidence distribution, drift flag, per-version breakdown | `http://grafana:3000/d/model_performance` |
| Grafana — `fall_events_timeline` | falls today, recent events table, confidence scatter (SQL panel from Postgres) | `http://grafana:3000/d/fall_events_timeline` |
| `server_health/` admin UI :8006 | aggregate `/health` probes across all our pods | `http://localhost:8006/` (or via ingress) |

Prometheus metric names exposed by `inference_server/services/metrics_collector.py`:

- `fall_detection_requests_total{model_version, fall_detected}`
- `fall_detection_request_latency_seconds`
- `fall_detection_confidence` (histogram)
- `fall_detection_errors_total{type}`

### 5.2 Triggers for retraining

We don't have a hard rule. Consider retraining when **any** of the following:

| Signal | Where you see it | Action |
|--------|------------------|--------|
| **False positive rate climbing** | `fall_history` SELECT WHERE patient_confirmed='no' / total | Look at confidence distribution of FPs. If they cluster near 0.5–0.6, threshold tuning may help before retrain. If they spread across 0.6–0.9, retrain. |
| **Missed falls reported clinically** | Manual — Charite reports a fall the system didn't catch | This is the most important signal. Retrain immediately if you have the raw window data. |
| **Confidence drift** | `model_performance` Grafana panel — mean confidence shifts >10% from baseline | Indicates input distribution shifted (new wearable batch, calibration change). Retrain. |
| **New cohort enrolled** | Project status meeting | Retrain to include their data. |
| **Quarterly cadence** | Calendar | Default minimum cadence even with no signals. |

### 5.3 What to compare during retrain

In the MLflow UI compare runs side by side. The acceptance criteria:

> **For a safety-critical fall detector: recall > AUC > F1 > precision.**

A new model that boosts precision by 3 pts but drops recall by 2 pts is a regression for our use case. Reject it.

### 5.4 What "drift" looks like in our metrics

Specifically: the `fall_detection_confidence` histogram on the `model_performance` dashboard. We have a baseline distribution from the v3 training set. If the live distribution deviates by KS distance > 0.15 from baseline, the dashboard flags it. The flag does not auto-trigger retraining — it's an alert for the human.

---

## 6. Model version management

### 6.1 The version lifecycle

```
[retrain.py creates a new run] ─► MLflow logs it ─► you call:
    mlflow.register_model(run_uri, "fall_detector")  ─► MLflow Registry creates "fall_detector v<n>"
                                                       (Stage = None by default)
                                                              │
                                                              ▼
[Inspect in MLflow UI: metrics, confusion matrix, comparison]
                                                              │
                                                              ▼
[Promote v<n> to Stage = Staging] ──► run smoke tests against staging
                                                              │
                                                              ▼
[Promote v<n> to Stage = Production] ──► becomes the new "default to switch to"
                                                              │
                                                              ▼
[Hot-swap the running inference_server to load it]
```

### 6.2 Hot-swap (in-memory)

The inference server has two endpoints:

```
GET  /model/list                  — what's loaded + what's available in MinIO + MLflow stages
POST /model/switch                — { "model_version": "v3", "source": "registry" | "file" }
```

Switching to a registry stage:

```powershell
curl -X POST http://localhost:8001/model/switch `
  -H "Content-Type: application/json" `
  -H "X-API-Key: $env:INFERENCE_API_KEY" `
  -d '{"model_version": "Production", "source": "registry"}'
```

Switching to a file (works against the bundled `model/` folder for offline rollback):

```powershell
curl -X POST http://localhost:8001/model/switch `
  -H "Content-Type: application/json" `
  -H "X-API-Key: $env:INFERENCE_API_KEY" `
  -d '{"model_version": "v3", "source": "file"}'
```

Same intent, faster: `local_dev/dev_scripts/switch_model.ps1`.

```powershell
.\local_dev\dev_scripts\switch_model.ps1 -Stage Production
```

### 6.3 Persistence — IMPORTANT gotcha

> **Hot-swap is in-memory only.** If the inference-server pod restarts (rolling upgrade, OOM kill, anything), it boots with whatever `MODEL_VERSION` is set in the chart's ConfigMap.

To make a hot-swap "stick" across restarts, update `values.yaml → inferenceServer.modelVersion` and `helm upgrade`. The hot-swap endpoint is for fast iteration during the trial — for permanent changes, change the chart.

### 6.4 The admin UI for this — `ml_dashboard`

`ml_dashboard/` (port 8004) has buttons for:
- Trigger retrain (calls `retrain.retrain` as subprocess)
- Promote a Staging model to Production (MLflow API)
- Hot-swap inference-server to a Stage or File version

In production, this is the most operator-friendly entry point. It is NOT containerised yet — the operator runs it locally and points at the cluster ingress URL. (See `01_k8s.md` Section 4 — `ml_dashboard` is admin tooling, not a production service.)

### 6.5 Rollback

Three options, fastest first:

1. **Hot-swap to a file version** — `.\switch_model.ps1 -Stage v3` (rollback to the bundled v3 .pkl).
2. **Hot-swap to a previous Registry version** — POST `/model/switch` with `model_version: <previous-stage-name>`.
3. **`helm rollback fall-detection <revision>`** — full chart rollback. Slowest because pods restart, but the only one that "sticks".

In an emergency, option 1 takes ~50 ms. Use it.

---

## 7. Files you should know

| Path | What |
|------|------|
| `inference_server/server.py` | FastAPI app, `/predict`, `/model/switch`, `/model/list`, `/health` |
| `inference_server/services/metrics_collector.py` | Prometheus metrics |
| `inference_server/services/db_writer.py` | BackgroundTask write of inference_log + feature_snapshot |
| `ml_pipeline/core/inference_engine.py` | Where features → XGBoost happens |
| `ml_pipeline/core/model_registry.py` | Loads .pkl from file or MinIO/MLflow |
| `ml_pipeline/data_input/accelerometer_processor/` | ACC pipeline (LSB conversion, resample, paper features) |
| `ml_pipeline/data_input/barometer_processor/` | Barometer EMA + slope-limit features |
| `model/model_<ver>/model.pkl` | The actual XGBoost binaries |
| `model/model_<ver>/feature_names.json` | Feature names this version expects |
| `retrain/data_pipeline.py` | The Postgres JOIN that produces labelled examples |
| `retrain/retrain.py` | The training CLI, writes to MLflow + MinIO |
| `retrain/seed_test_data.py` | Synthetic / InfluxDB data seeders for testing |
| `retrain/_delete_exp.py` | Permanently purge an MLflow experiment (soft-delete is not enough) |
| `local_dev/dev_scripts/switch_model.ps1` | Friendly wrapper for `/model/switch` |
| `shared_db/models.py` | SQLAlchemy ORM: InferenceLog, FeatureSnapshot, FallHistory |
| `infrastructure/mlflow/Dockerfile` | Our custom MLflow image (boto3 + psycopg2 baked in) |

---

## 8. Things that will bite you

- **`MLFLOW_ARTIFACT_ROOT` env var is ignored when MLflow tracking URI is SQLite** — you must pass `artifact_location` explicitly to `client.create_experiment()`. We do this in `retrain.py`. If you fork, don't lose it.
- **MLflow soft-deletes experiments.** `mlflow experiments delete` does not actually free the name — `set_experiment("foo")` then errors with "Cannot set a deleted experiment". To permanently drop, run `retrain/_delete_exp.py` which executes raw SQL.
- **`observation_id` MUST be in the MQTT payload from the mobile app.** Without it, `fall_history` rows can't join to `inference_log` and the row is unusable for retraining. We've made this requirement explicit in the mobile app handover doc.
- **Hot-swap doesn't survive restart** (Section 6.3). Update the chart for permanent changes.
- **Default threshold = 0.5 is not tuned per patient.** If you see consistently calibrated false-positives at a single patient, consider per-patient threshold (extension — not implemented).
- **`patient_confirmed='not_answered'` is dropped from retraining.** This is intentional (label is ambiguous), but it does mean if a patient is unconscious every time they fall, those examples are lost. We accept this.
- **Adding a new feature in retraining requires updating `feature_names.json`.** The inference server reads this on load to know how to assemble the feature vector. If you train with feature X but `feature_names.json` doesn't list X, the server won't compute it at predict time. `retrain.py` writes a fresh `feature_names.json` next to the new `.pkl` — keep them paired.

---

## 9. Cross-references

- [`01_k8s.md`](01_k8s.md) — how the model is deployed (image content, hot-swap mechanics)
- [`03_fall_detection_system.md`](03_fall_detection_system.md) — system context, where the inference server fits
- [`04_mobile_app_integration.md`](04_mobile_app_integration.md) — what the mobile app sends to `/predict`
- [`08_user_flow_admin.md`](08_user_flow_admin.md) — admin actions (trigger retrain, promote, hot-swap) from the UI
- `REFACTOR_DOCUS/mlops_retraining_cycle.md` (in repo) — full MLOps cycle reference

---

## 10. Contact

| For | Reach out to |
|-----|--------------|
| Model architecture, retraining bugs, MLflow issues | Hayate (MCS) |
| Clinical labelling questions, sample-rate decisions, missed-fall reports | Charite |
| Hot-swap orchestration / when to retrain in production | Whoever inherits the ML admin role |
