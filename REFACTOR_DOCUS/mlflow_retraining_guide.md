# MLflow Retraining Guide — Fall Detection

This document covers:
1. How data flows from production into the retraining pipeline
2. How to run the pipeline (with and without Charite data)
3. What metrics to look at and how to decide when to switch models

---

## How the retraining pipeline works

```
Production system (running continuously)
    │
    ├── inference_server receives /predict
    │       → BackgroundTask writes to Postgres:
    │             inference_log      (one row per prediction)
    │             feature_snapshot   (one row per feature per prediction)
    │
    └── mock_app / real mobile app waits for patient confirmation
            → publishes MQTT alert
            → fall_dashboard writes to Postgres:
                  fall_history  (patient_confirmed = 'yes' / 'no' / 'not_answered')

Retraining pipeline (run manually or on a schedule)
    │
    ├── data_pipeline.py  →  SQL JOIN across the three tables → labelled DataFrame
    ├── retrain.py        →  train XGBoost → log to MLflow → save .pkl
    └── inference_server  →  POST /model/switch to hot-swap to the new model
```

**Key design choice:** features are stored at inference time, so retraining
is a single SQL JOIN — no re-running feature extraction, no InfluxDB access needed.

---

## Label logic

| fall_detected | patient_confirmed | Label | Included? |
|--------------|-------------------|-------|-----------|
| True | `yes` | **1 (fall)** | Yes |
| False | — (no MQTT alert) | **0 (no fall)** | Yes |
| True/False | `no` | **0 (no fall)** | Yes — patient corrected the model |
| True | `not_answered` | — | **No — ambiguous, excluded** |

`not_answered` rows are excluded because we don't know if the patient actually fell.
Treating them as falls would add noise; treating them as non-falls would suppress recall.

---

## Prerequisites

```powershell
cd C:\Users\hayat\Documents\6G\FallDetection_new\_6G_Integration_v2_mqtt
.\venv\Scripts\Activate.ps1
pip install -r retrain/requirements.txt
```

`DATABASE_URL` in `.env` must point at Postgres (not SQLite) before running anything:

```
DATABASE_URL=postgresql+psycopg2://fall_user:fall_pass@localhost:5432/fall_detection
```

---

## Step 1 — Check how much data you have

Always do a dry run first. This prints dataset stats without training:

```powershell
python -m retrain.retrain --dry-run
```

Example output:
```
Dataset summary:
  Total labelled rows : 87
  Positive (fall=yes) : 18
  Negative            : 69
  Features            : 16
  Model versions      : ['v0']
```

**If the table is empty** (no production data yet), seed it with test data first
(see Step 1b below), then come back to this step.

---

## Step 1b — Seed test data (when production data is not available)

Use this when:
- Setting up for the first time
- Testing the pipeline without Charite patients
- You want to add more rows quickly

### Option A — Synthetic (no external dependencies)

Generates realistic fall / non-fall feature distributions directly into Postgres.
No InfluxDB, no running inference server needed.

```powershell
# 100 windows, 20% fall rate
python -m retrain.seed_test_data --synthetic 100 --model-version v0

# More data, different model version
python -m retrain.seed_test_data --synthetic 500 --model-version v3
```

### Option B — Real InfluxDB data (our own test bucket)

Fetches real ACC windows from our cloud InfluxDB, runs the full preprocessing
pipeline, stores features in Postgres. Produces more realistic feature distributions.

```powershell
# Last 24 hours of data from InfluxDB
python -m retrain.seed_test_data --influxdb --lookback-hours 24
```

Requires `INFLUXDB_*` credentials set in `.env`.

---

## Step 2 — Train and log to MLflow

```powershell
# Basic: retrain v0 model on whatever is in Postgres
python -m retrain.retrain

# Specify model version and dataset tag
python -m retrain.retrain --model-version v0 --dataset our_data

# Also register in MLflow Model Registry
python -m retrain.retrain --model-version v0 --dataset our_data --register

# Use Charite data (once available)
python -m retrain.retrain --model-version v0 --dataset charite --register
```

**What happens:**
1. Loads labelled data from Postgres (JOIN across 3 tables)
2. Builds feature matrix in the correct column order for the model version
3. Stratified 80/20 train/test split
4. Trains XGBoost with class balancing (`scale_pos_weight`)
5. Logs params, metrics, model, and feature list to MLflow
6. Saves `.pkl` to `model/model_{version}_retrained/`
7. Optionally registers in MLflow Model Registry as `fall-detection-xgboost`

---

## Step 3 — View results in MLflow UI

```powershell
mlflow ui --backend-store-uri ./mlruns
# Open http://localhost:5000
```

In the UI you will see:
- All runs grouped by experiment (`fall-detection-v0`, `fall-detection-v3`, etc.)
- Parameters: `n_train`, `n_features`, `scale_pos_weight`, `threshold`
- Metrics: `recall`, `precision`, `f1`, `auc`, `tp`, `fp`, `tn`, `fn`
- Tags: `dataset`, `model_version`, `window_seconds`
- Artifacts: `model.pkl`, `feature_names.txt`

---

## Step 4 — Decide whether to switch models

### The most important metric for fall detection: Recall

Fall detection is a **safety-critical** system. A missed fall (False Negative) is
far more dangerous than a false alarm (False Positive). The priority order is:

```
Recall  >  AUC  >  F1  >  Precision
```

**Why recall matters most:**
- FN = patient fell but system missed it → no alert sent → caregiver not notified
- FP = patient did not fall but alert fires → caregiver checks on patient → mild inconvenience

### Minimum thresholds to consider switching

| Metric | Minimum to consider | Notes |
|--------|-------------------|-------|
| Recall | ≥ current model | Never switch to a model with lower recall |
| AUC | ≥ 0.85 | Overall discrimination ability |
| F1 | ≥ 0.75 | Balance check — prevents over-tuning recall by flagging everything |
| n_train | ≥ 50 positive examples | Below this, metrics are unreliable |

### Decision table

| New model recall vs current | New model F1 vs current | Decision |
|----------------------------|------------------------|----------|
| Higher | Higher or equal | **Switch** |
| Higher | Lower (but > 0.70) | **Switch** — recall wins for safety |
| Higher | Much lower (< 0.70) | Review — may be predicting all positives |
| Equal | Higher | Consider switching — fewer false alarms |
| Lower | Any | **Do not switch** |

### Red flags — do not switch if:

- `recall = 1.0` and `precision < 0.1` — the model is predicting everything as a fall
- `n_test < 10` — test set too small to trust the metrics
- `AUC < 0.7` — model is barely better than random
- `fn > 0` when `n_positive` is small — any missed fall on a tiny test set is significant

---

## Step 5 — Switch to the new model

Once you are satisfied with the metrics, hot-swap the model without restarting the server:

```powershell
# Check what models are available
curl.exe http://localhost:8001/model/list

# Switch to the retrained model
curl.exe -X POST http://localhost:8001/model/switch `
  -H "Content-Type: application/json" `
  -H "X-API-Key: <your-api-key>" `
  -d '{"version": "v0_retrained"}'

# Verify
curl.exe http://localhost:8001/model/info
```

The retrained `.pkl` is saved to `model/model_v0_retrained/model_v0_retrained.pkl`.
The `version` string to pass to `/model/switch` is `v0_retrained`.

To roll back to the previous model:

```powershell
curl.exe -X POST http://localhost:8001/model/switch `
  -H "Content-Type: application/json" `
  -H "X-API-Key: <your-api-key>" `
  -d '{"version": "v0"}'
```

---

## Step 6 — Monitor after switching

After switching, watch the Grafana dashboards for at least one day:

**`model_performance` dashboard** — look for:
- Fall rate per hour trending to a reasonable level (not spiking or dropping to zero)
- Median confidence staying above 0.7 (low confidence = model is uncertain)
- Low-confidence ratio gauge — if it rises above ~20%, the model may be drifting

**`ml_server_overview` dashboard** — look for:
- p95 latency still under 100ms (model size increase can add latency)
- Error rate at zero

If things look wrong after switching, roll back immediately (see Step 5 above).

---

## Full example — test run from scratch

```powershell
# 1 — Activate venv and make sure Postgres is running
.\venv\Scripts\Activate.ps1
docker-compose -f infrastructure/docker-compose.yml up -d

# 2 — Seed 200 synthetic windows
python -m retrain.seed_test_data --synthetic 200 --model-version v0

# 3 — Check what was seeded
python -m retrain.retrain --dry-run

# 4 — Train and log to MLflow
python -m retrain.retrain --model-version v0 --dataset our_data --register

# 5 — View in MLflow UI
mlflow ui --backend-store-uri ./mlruns
# → http://localhost:5000

# 6 — If metrics look good, switch the running server
curl.exe -X POST http://localhost:8001/model/switch `
  -H "Content-Type: application/json" `
  -H "X-API-Key: 626e6c481b78c77d52db774d0e54a06cefb0553cb245fa15d5c7050fc7424e7a" `
  -d '{"version": "v0_retrained"}'
```

---

## When will we have real training data?

| Data source | Status | Required for |
|------------|--------|-------------|
| Our own InfluxDB (fd_test bucket) | Available now | `--influxdb` mode in seed_test_data.py |
| Synthetic data | Available now | `--synthetic` mode — pipeline testing only |
| Charite patient data | Blocked — data sharing agreement required | `--dataset charite` |

Until the Charite agreement is signed, use `--dataset our_data` and treat all
retrained models as experimental. Do not deploy a model trained only on synthetic
data to production patients.

---

## MLflow Model Registry stages (manual via UI)

When you run with `--register`, the model appears in the Registry under
`fall-detection-xgboost`. Stages are managed manually in the MLflow UI:

| Stage | Meaning | What to do |
|-------|---------|-----------|
| `None` | Just registered | Run evaluation — check metrics |
| `Staging` | Under review | Test on mock_app, compare metrics to Production |
| `Production` | Current best model | Inference server loads this version |
| `Archived` | Superseded | Kept for rollback reference |

To promote via UI: open http://localhost:5000 → Models → fall-detection-xgboost → click the version → Transition to → Staging / Production.

Automated promotion via `/model/switch` loading from registry by stage is not yet
implemented (Step 11.11 in todo.md). Currently the hot-swap endpoint loads from
the local `model/` directory by file path.
