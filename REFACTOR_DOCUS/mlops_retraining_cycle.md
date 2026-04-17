# MLOps Human-in-the-Loop Cycle — Fall Detection

This document covers the full retraining cycle: which tool does what,
when to retrain, and how to promote a new model to production.

---

## The full cycle — tool map

```
┌──────────────────────────────────────────────────────────────────────┐
│  PRODUCTION (running continuously)                                   │
│                                                                      │
│  SmarKo wearable → mobile app → POST /predict → inference_server    │
│                                                        │             │
│                                     Postgres ←─────────┘            │
│                                     inference_log                    │
│                                     feature_snapshot  ← per feature │
│                                     fall_history      ← after MQTT  │
└──────────────────────────────────────────────────────────────────────┘
            │
            │  (data accumulates over time)
            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  MONITORING — Grafana dashboards                                     │
│                                                                      │
│  model_performance.json   → confidence drift, fall rate per hour    │
│  ml_server_overview.json  → latency, error rate                     │
│  fall_events_timeline.json → falls today, confidence scatter        │
│                                                                      │
│  TRIGGER: human sees confidence clustering near 0.5, fall rate      │
│  shifting, or low-confidence ratio rising → time to retrain         │
└──────────────────────────────────────────────────────────────────────┘
            │
            │  human decides to retrain
            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  RETRAINING — retrain/ scripts                                       │
│                                                                      │
│  data source: Postgres only (feature_snapshot JOIN fall_history)     │
│  NO InfluxDB access needed — features already computed at inference  │
│                                                                      │
│  python -m retrain.retrain --model-version v0 --register            │
│                                                                      │
│  Saves: .pkl to model/model_v0_retrained/                           │
│  Logs:  metrics + artifact to MLflow                                 │
└──────────────────────────────────────────────────────────────────────┘
            │
            │  human evaluates metrics
            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  EVALUATION — MLflow UI (http://localhost:5000)                      │
│                                                                      │
│  Compare runs: recall, AUC, F1 side by side                         │
│  Promote in UI: Models → fall-detection-xgboost → Production        │
└──────────────────────────────────────────────────────────────────────┘
            │
            │  human approves switch
            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  DEPLOYMENT — inference_server hot-swap (no restart needed)          │
│                                                                      │
│  Option A (file-based):    POST /model/switch  {"version": "v0_retrained"}          │
│  Option B (registry-based): POST /model/switch  {"mlflow_stage": "Production"}      │
└──────────────────────────────────────────────────────────────────────┘
            │
            │  monitor for at least 1 day after switch
            └──────────────────────────────────────────── back to top
```

### What each tool is responsible for

| Tool | Role | What it does NOT do |
|------|------|---------------------|
| **InfluxDB** | Raw biosignal store (HR, ACC) for Patient Dashboard display | Not used for inference input in production |
| **Postgres** | Training data store: inference_log, feature_snapshot, fall_history | Not used for raw sensor data |
| **Grafana** | Monitor live model behaviour — drift detection, latency | Does not trigger retraining automatically |
| **MLflow** | Experiment tracking, run comparison, model registry | Does not serve predictions |
| **inference_server** | Runs predictions, exposes `/model/switch` | Does not retrain models |
| **retrain.py** | Trains new model, logs to MLflow, saves .pkl | Does not deploy — human decides |

---

## Why InfluxDB is NOT in the retraining loop

This is a common point of confusion.

**InfluxDB holds raw sensor data** (continuous stream of ACC x/y/z, barometer, HR)
used by the Patient Dashboard to show biosignal graphs to the caregiver.

**The retraining pipeline does not touch InfluxDB.** Here is why:

At inference time, `inference_server` already:
1. Receives a 9-second ACC window from the mobile app
2. Runs feature extraction (16 or 20 features)
3. Writes those computed features to Postgres (`feature_snapshot`)

So by the time you want to retrain, the features are already in Postgres — pre-computed,
in the exact same format the model expects. Retraining is a SQL JOIN, not a re-run of
feature extraction from raw sensor data.

```
InfluxDB (raw sensor data) → Patient Dashboard only
                              ↑
                              not touched by retrain.py

Postgres (computed features) → retrain.py data source
```

`seed_test_data --influxdb` is a **development utility only** — it replays our
own cloud InfluxDB bucket to simulate what Postgres would look like after days of
live inference. It is not part of the production retraining flow.

---

## Prerequisites

```powershell
cd C:\Users\hayat\Documents\6G\FallDetection_new\_6G_Integration_v2_mqtt
.\venv\Scripts\Activate.ps1
pip install -r retrain/requirements.txt
```

`DATABASE_URL` in `.env` must point at Postgres (not SQLite):

```
DATABASE_URL=postgresql+psycopg2://fall_user:fall_pass@localhost:5432/fall_detection
```

`MLFLOW_TRACKING_URI` must match between `retrain.py` and `inference_server`
(both read from `.env`):

```
MLFLOW_TRACKING_URI=sqlite:///./mlruns.db
```

---

## Step 1 — Check how much data you have

Always do a dry run first:

```powershell
python -m retrain.retrain --dry-run
```

Example output:
```
Dataset summary:
  Total labelled rows : 954
  Positive (fall=yes) : 146
  Negative            : 808
  Features            : 16
  Model versions      : ['v0']
```

**Minimum before training:** at least 50 positive (fall) rows. Below that,
metrics are unreliable.

---

## Step 1b — Seed test data (when production data is not yet available)

Use this when setting up for the first time or testing the pipeline.

### Option A — Synthetic (no external dependencies)

Generates realistic fall / non-fall feature distributions directly into Postgres.
No InfluxDB, no running inference server needed.

```powershell
python -m retrain.seed_test_data --synthetic 200 --model-version v0
```

### Option B — Replay from our own InfluxDB (development only)

Fetches historical ACC windows from our cloud InfluxDB bucket, runs the full
feature extraction pipeline locally, and writes the results to Postgres.
This produces more realistic feature distributions than synthetic data.

```powershell
python -m retrain.seed_test_data --influxdb --lookback-hours 24
```

**This is a development utility — not part of the production retraining flow.**
In production, Postgres is populated by the live inference_server automatically.

---

## Step 2 — Train and log to MLflow

```powershell
# Retrain v0 on data in Postgres, register in MLflow registry
python -m retrain.retrain --model-version v0 --dataset our_data --register

# Use Charite data once available
python -m retrain.retrain --model-version v0 --dataset charite --register
```

**What happens:**
1. SQL JOIN: `inference_log` + `feature_snapshot` + `fall_history` → labelled DataFrame
2. Feature matrix built in the correct column order for the model version
3. Stratified 80/20 train/test split
4. XGBoost trained with class balancing (`scale_pos_weight`)
5. Metrics + params + model artifact logged to MLflow
6. `.pkl` saved to `model/model_{version}_retrained/`
7. Model registered as `fall-detection-xgboost` in MLflow registry (if `--register`)

---

## Step 3 — View results in MLflow UI

```powershell
mlflow ui --backend-store-uri sqlite:///./mlruns.db --workers 1
# Open http://localhost:5000
```

In the UI:
- **Experiments tab** — all runs grouped by `fall-detection-v0`, `fall-detection-v3`, etc.
  Compare runs: params (`n_train`, `threshold`), metrics (`recall`, `auc`, `f1`), tags (`dataset`)
- **Models tab** — `fall-detection-xgboost` registry; promote versions to Staging / Production

---

## Step 4 — Decide whether to switch models

### Priority order for fall detection (safety-critical)

```
Recall  >  AUC  >  F1  >  Precision
```

A missed fall (False Negative) means no alert sent to caregiver.
A false alarm (False Positive) means caregiver checks on patient unnecessarily.
Missing a fall is far worse.

### Minimum thresholds to consider switching

| Metric | Minimum | Notes |
|--------|---------|-------|
| Recall | ≥ current model | Never switch to lower recall |
| AUC | ≥ 0.85 | Overall discrimination |
| F1 | ≥ 0.75 | Sanity check — prevents tuning recall by flagging everything |
| n_positive | ≥ 50 | Below this, metrics are unreliable |

### Decision table

| New model recall vs current | New model F1 | Decision |
|----------------------------|--------------|----------|
| Higher | Higher or equal | **Switch** |
| Higher | Lower (but > 0.70) | **Switch** — recall wins |
| Higher | Much lower (< 0.70) | Review — may be predicting everything as a fall |
| Equal | Higher | Consider switching — fewer false alarms |
| Lower | Any | **Do not switch** |

### Red flags — do not switch if:

- `recall = 1.0` and `precision < 0.1` — predicting everything as a fall
- `n_test < 10` — test set too small
- `AUC < 0.7` — barely better than random
- `fn > 0` on a tiny test set — any missed fall is significant

---

## Step 5 — Promote in MLflow UI

Before switching the server, promote the version in the registry:

```
http://localhost:5000
→ Models → fall-detection-xgboost → click the version → Transition to → Production
```

Stages:

| Stage | Meaning |
|-------|---------|
| `None` | Just registered — run evaluation |
| `Staging` | Under review — test on mock_app |
| `Production` | Current best — inference server loads this |
| `Archived` | Superseded — kept for rollback |

---

## Step 6 — Switch the live model

Two options. Both hot-swap without restarting the server.

### Option A — Registry-based (recommended after promotion)

```powershell
# Switch to whatever is currently in the Production stage
curl.exe -X POST http://localhost:8001/model/switch `
  -H "Content-Type: application/json" `
  -H "X-API-Key: 626e6c481b78c77d52db774d0e54a06cefb0553cb245fa15d5c7050fc7424e7a" `
  -d '{"mlflow_stage": "Production"}'
```

The server downloads the `.pkl` from the MLflow artifact store and hot-swaps it.
`MLFLOW_TRACKING_URI` in `.env` must point at the same store used by `retrain.py`.

### Option B — File-based (simpler for local testing)

```powershell
curl.exe -X POST http://localhost:8001/model/switch `
  -H "Content-Type: application/json" `
  -H "X-API-Key: 626e6c481b78c77d52db774d0e54a06cefb0553cb245fa15d5c7050fc7424e7a" `
  -d '{"version": "v0_retrained"}'
```

### Verify and roll back

```powershell
# Confirm the switch
curl.exe http://localhost:8001/model/info

# Roll back to original model
curl.exe -X POST http://localhost:8001/model/switch `
  -H "Content-Type: application/json" `
  -H "X-API-Key: 626e6c481b78c77d52db774d0e54a06cefb0553cb245fa15d5c7050fc7424e7a" `
  -d '{"version": "v0"}'
```

---

## Step 7 — Monitor after switching

Watch Grafana for at least one day:

**`model_performance` dashboard:**
- Fall rate per hour trending reasonably (not spiking or dropping to zero)
- Median confidence staying above 0.7
- Low-confidence ratio gauge — if rising above ~20%, model may be drifting again

**`ml_server_overview` dashboard:**
- p95 latency still under 100ms
- Error rate at zero

If things look wrong: roll back immediately (see Step 6 above).

---

## When will we have real training data?

| Data source | Status | Used for |
|-------------|--------|----------|
| Synthetic (`--synthetic`) | Available now | Pipeline testing only — do not deploy |
| Our own InfluxDB replay (`--influxdb`) | Available now | Dev utility — more realistic than synthetic |
| Charite patient data | Blocked — data sharing agreement required | `--dataset charite` — real retraining |

Do not deploy a model trained only on synthetic data to production patients.
