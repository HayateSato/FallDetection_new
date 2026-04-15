# MLflow — Why It Fits the Retraining Pipeline

## The problem MLflow solves

When you retrain a model, you will run the training script many times:
- once with the original dataset
- once with Charite data added
- once with different hyperparameters
- once after fixing a bug in the feature extractor
- ...

Without a tool, you end up with folders like `model_v3_charite_final_REAL_v2.pkl`
and no record of what data, what parameters, or what preprocessing produced each file.
Three months later you cannot reproduce the result or explain why one version was
better than another.

MLflow is a tool that answers four questions for every training run:
1. **What inputs went in?** (parameters, dataset version)
2. **What came out?** (metrics, the model file itself)
3. **Which version is in production right now?**
4. **How do I load it in the inference server?**

---

## The four components and what each one does

### 1. Tracking — "what happened in this run?"

Every time you call `retrain.py`, MLflow opens a **run** and records:

```
Run: 2026-04-15 14:32
  Parameters:   window_seconds=9, sample_rate=50, threshold=0.5, model_version=v0
  Dataset tag:  charite+original
  Metrics:      accuracy=0.94, f1=0.91, auc=0.97, precision=0.89, recall=0.93
  Artifact:     model.pkl  (the trained XGBoost file)
  Duration:     47s
```

All runs are stored in a central database (SQLite locally, Postgres in production).
You can open the MLflow UI and compare any two runs side by side — plot accuracy vs
threshold, or see whether adding Charite data improved recall without hurting precision.

**Why this matters for your project:**
The Charite population (elderly, hospital setting) may behave differently from your
original training data. You need a record of exactly which data split and which
hyperparameters produced the improvement, so you can reproduce it if something breaks.

---

### 2. Artifacts — "where is the model file?"

After training, MLflow stores the model file in an **artifact store**. Locally this
is just a folder on disk. In production it points at object storage (MinIO or S3).

Crucially, the model is stored **with its metadata**: which features it expects, which
framework version trained it, what the input schema looks like. This means you can
load it later with one line:

```python
model = mlflow.sklearn.load_model("runs:/<run_id>/model")
# or by registry name:
model = mlflow.sklearn.load_model("models:/FallDetector/Production")
```

No more `joblib.load("../../model/v3_charite_final.pkl")` with a fragile relative path.

---

### 3. Model Registry — "which version is approved for production?"

The registry is a **promotion pipeline** for model versions:

```
Training run  →  Staging  →  (evaluation)  →  Production  →  (eventually) Archived
```

- **Staging:** model trained, not yet validated. Runs on held-out test set, maybe
  on a shadow deployment that receives real traffic but doesn't act on results.
- **Production:** approved. This is the version `POST /model/switch` should load.
- **Archived:** superseded but kept for audit trail.

This separation means you never accidentally push an undertested model to production.
The promotion from Staging to Production is a deliberate, logged action.

**How it connects to your inference server:**
Instead of `POST /model/switch {"version": "v3"}` loading a local `.pkl` file by
filename, you change it to:

```python
model = mlflow.sklearn.load_model("models:/FallDetector/Production")
```

The inference server always loads whatever version is currently marked Production
in the registry. Promoting a new model becomes a registry operation, not a file copy.

---

### 4. Projects — "how do I reproduce this training run?"

MLflow Projects package your training script with its dependencies so anyone (or any
CI pipeline) can reproduce a run exactly:

```bash
mlflow run . -P dataset=charite -P threshold=0.5
```

This is useful for Step 11d (scheduled retraining trigger) — a cron job or a
threshold-based trigger can kick off a reproducible training run automatically.

---

## How it fits your specific retraining pipeline

```
[Postgres inference_log]          [caregiver fall_history]
 - acc window / features           - patient_confirmed = yes/no
 - confidence, model_version       - detection_time
 - detection_time
        │                                  │
        └──────────── JOIN ────────────────┘
                         │
                  labelled dataset
                  (ACC window + confirmed fall label)
                         │
                   retrain.py
                         │
              mlflow.start_run()
                         │
              ┌──────────┴──────────┐
              log params          log metrics
              log artifact        tag dataset=charite
              (model.pkl)
                         │
               MLflow Model Registry
                  → Staging
                  → (validate)
                  → Production
                         │
              inference_server loads
              "models:/FallDetector/Production"
```

---

## MLflow vs Prometheus+Grafana — which watches what

A common point of confusion:

| Question | Tool |
|----------|------|
| Is the live model performing well right now? | Prometheus + Grafana |
| Is confidence drifting over the last 7 days? | Prometheus + Grafana |
| Which training run produced the best F1 score? | MLflow Tracking |
| What parameters were used in model v3? | MLflow Tracking |
| Which model version is currently in production? | MLflow Registry |
| How do I roll back to the previous model? | MLflow Registry (demote + promote) |
| What was the accuracy on the Charite test split? | MLflow Tracking |

Prometheus watches the **running system**. MLflow watches the **training and versioning process**.
Both are needed. Neither replaces the other.

---

## Practical setup for this project

**Local development (no extra infrastructure):**
```bash
pip install mlflow
mlflow ui          # opens http://localhost:5000
# runs stored in ./mlruns/ (SQLite)
```

**Production (inside the FOCUS Kubernetes cluster):**
- MLflow tracking server as a pod, backed by Postgres + MinIO artifact store
- `MLFLOW_TRACKING_URI=http://mlflow-server:5000` in `.env`
- Same `retrain.py` script works in both environments — only the URI changes

---

## Key takeaway

MLflow is not a replacement for writing a good training script. It is a **wrapper**
around the training script that records everything that went in and came out.
The cost is small (a few extra lines in `retrain.py`). The benefit is that six months
from now, when someone asks "why did the fall detection rate drop after the last
model update?", you have a complete audit trail to answer that question.



--- 
**Data source for retraining: Postgres, not InfluxDB.**

Here's why: the inference server already stores pre-computed features in `feature_snapshot` (one row per feature per prediction). `fall_history` stores the ground truth label (`patient_confirmed`). So retraining is just a SQL JOIN — no InfluxDB archaeology needed.

For testing without Charite patients: `seed_test_data.py` either generates synthetic labelled windows (no InfluxDB needed, fastest) or reads from your own InfluxDB, runs the same preprocessing pipeline, and inserts rows directly into Postgres — simulating what would have happened if patients had been wearing the device.

Now let me implement all of Step 11: