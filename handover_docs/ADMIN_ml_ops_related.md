# MLOps Handover — Model Lifecycle, Retraining, and Training Data

**Audience:** Whoever takes over operating the fall detection inference service and the MLflow retraining loop after Hayate.
**Repo / branch:** `_6G_Integration_v2_mqtt/` on branch `6G-integration_with_MQTT`.
**Scope:** This document only covers the MLOps side — the inference server, the model files, MLflow, MinIO, the retraining script, and the Postgres tables that feed it. Helm/Kubernetes deployment is in `Tech_integrator.md`. Mobile app integration is in `ISA_mobile_app.md`.

---

## 1. The Big Picture

There are three things you need to keep separate in your head:

| Layer | What lives there | When it changes |
|-------|-----------------|------------------|
| **Inference server** (`inference_server/server.py`, port 8001) | The XGBoost model object loaded in memory. One model at a time. | Every `POST /model/switch` swaps the in-memory model. The pod itself does not restart. |
| **Model artifacts** | `.pkl` files. Two possible homes: the `model/` folder baked into the Docker image, or MinIO via the MLflow Model Registry. | Adding a new artifact happens via the retraining script. Promoting one to production happens via `set_registered_model_alias`. |
| **Training data** | Postgres only — `inference_log` + `feature_snapshot` + `fall_history`. **Not InfluxDB.** | Every `/predict` call writes a row to `inference_log` + N rows to `feature_snapshot`. Every confirmed alert (after the patient confirmation popup) writes a row to `fall_history`. |

Once you internalise that the inference server only ever holds one model, that artifacts live in MinIO, and that training data is reconstructed from Postgres, the rest of this document is just operations.

---

## 2. Where Model Files Live and How They Are Loaded

The inference server can load a model from two sources. Three operational situations map onto those two sources:

| Situation | Source | Trigger |
|-----------|--------|---------|
| **Server startup** | `model/` folder (baked into Docker image) | Pod boot — reads `MODEL_VERSION` from env, loads the matching `.pkl` |
| **File-based hot-swap** | `model/` folder | `POST /model/switch {"version": "v0_retrained"}` |
| **MLflow hot-swap** | MinIO via MLflow registry | `POST /model/switch {"mlflow_stage": "Production"}` |

So `model/` covers two situations (startup + dev-time file swap), and MinIO covers the one production case.

### Why both sources exist

- The image-baked `model/` folder is the **startup fallback**. It guarantees the inference pod can serve traffic the moment it starts, even if MinIO or the MLflow tracking server are not yet ready. This matters because all three services come up in parallel under Kubernetes — there is no ordering guarantee.
- MinIO (via MLflow) is the **production path**. After cluster startup, you run `switch_model.ps1 -Stage Production` once. From then on, the in-memory model came from MinIO. Each subsequent retrain → promote → switch cycle replaces it again, and the `model/` folder is never touched at runtime.
- File-based hot-swap (`{"version": "..."}`) is purely a dev convenience. Use it when you want to test a `.pkl` you copied locally without going through MLflow.

### Operational rule

In production you only ever care about **MLflow → MinIO → switch**. The `model/` folder is something you pre-bake into the image and forget about until the next image build.

---

## 3. The Retraining Flow — Step by Step

The retraining script lives in `retrain/`. It pulls labelled examples from Postgres, fits a new XGBoost model, logs the run to MLflow, optionally registers it in the Model Registry, and saves the `.pkl` to disk.

All commands below assume `_6G_Integration_v2_mqtt/` is your current directory and the venv is active.

### Step 1 — Check whether you have enough data

```powershell
python -m retrain.retrain --dry-run
```

This counts labelled rows in Postgres. **Minimum: 50 positive (confirmed-fall) rows.** If you have fewer, seed synthetic data first:

```powershell
python -m retrain.seed_test_data --synthetic 200 --model-version v0
```

`seed_test_data --synthetic` writes fake rows directly into Postgres — no InfluxDB or live inference server needed. There is also a `--influxdb` flag that pulls real ACC windows from our cloud InfluxDB, runs feature extraction locally, and writes the result to Postgres. The `--influxdb` mode is a dev convenience for simulating "what Postgres would look like after days of live inference" — it is **not** part of the production retraining loop.

### Step 2 — Train and (optionally) register

```powershell
python -m retrain.retrain --model-version v0 --dataset our_data --register
```

What this does:
1. Pulls the labelled dataset from Postgres (see section 4 for the SQL).
2. Splits into train/test, fits XGBoost.
3. Saves the new `.pkl` to `model/model_v0_retrained/`.
4. Logs params + metrics to the local MLflow tracking store (`./mlruns/` SQLite or `mlflow:5000` in production).
5. With `--register`, registers the artifact in the Model Registry under the name `fall-detection-xgboost`. Each registration creates a new version number (1, 2, 3 …).

Drop `--register` if you only want to evaluate without polluting the registry.

### Step 3 — Open the MLflow UI and decide whether the new model is good

```powershell
mlflow ui --backend-store-uri sqlite:///./mlruns.db --workers 1
# → http://localhost:5000
```

The UI shows every run side by side. Look at:
- **Recall** — false negatives are the worst failure mode for a fall detector. Your floor: the new recall must be ≥ the current production recall.
- **AUC, F1** — secondary tie-breakers.
- **Confusion matrix** — check that fp count did not balloon while chasing recall.

If the new model is not better, do nothing. The old model stays in production.

### Step 4 — Promote to Production

MLflow ≥ 2.9 replaced "stages" with "aliases". There is no UI button — you set the alias in code:

```powershell
# Replace 2 with the version number from the Models tab in the MLflow UI
python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///./mlruns.db')
mlflow.tracking.MlflowClient().set_registered_model_alias(
    'fall-detection-xgboost', 'Production', 2)
print('Done')
"
```

This points the alias `Production` at version 2. The artifact in MinIO does not move — only the alias pointer changes.

### Step 5 — Hot-swap the live model (no server restart)

```powershell
.\local_dev\dev_scripts\switch_model.ps1 -Stage Production
```

This calls `POST /model/switch {"mlflow_stage": "Production"}`. The inference server downloads the `.pkl` from MinIO and replaces the in-memory model under a thread lock. There is no downtime — `/predict` keeps serving on the old model until the new one is ready, then atomically swaps.

### Step 6 — Verify

```powershell
curl.exe http://localhost:8001/model/info
# look for "loaded_as": "mlflow:Production:v2(v0)"
```

If the response still says `loaded_as: "file:..."`, the swap did not succeed — check the inference server logs for an MLflow connection error (most commonly: MinIO credentials wrong, or the MLflow tracking URI environment variable not set in the pod).

---

## 4. Where Training Data Comes From

The retraining script does **not** read InfluxDB. Features are computed once, at prediction time, by the inference server, and stored in Postgres. This means retraining is fast (no recomputation) and reproducible (the exact features the model saw at inference time are the ones it will see during training).

### The SQL that builds the dataset

`retrain/data_pipeline.py` (around line 85) runs:

```sql
FROM inference_log il
JOIN feature_snapshot fs ON fs.inference_id = il.id
LEFT JOIN fall_history fh ON fh.observation_id = il.observation_id
```

It is a **LEFT JOIN** — `inference_log` is the base. Every `/predict` call is one row in `inference_log` plus N rows in `feature_snapshot` (one per feature). `fall_history` is only joined when one exists, i.e. when the model said fall AND the mobile app published a confirmation alert.

### What the row counts actually mean

A typical Postgres state during testing might be:
- 1014 rows in `inference_log` with `fall_detected=False`
- 165 rows in `fall_history`

That breaks down as:

| Source | What it is | Count (approx) |
|--------|-----------|----------------|
| `fall_detected=False`, no `fall_history` row | Model correctly said "no fall" — true negative | ~1007 |
| `fall_detected=True`, `patient_confirmed='yes'` | Confirmed real fall | ~150 |
| `fall_detected=True`, `patient_confirmed='no'` | **Actual false positive** — model said fall, patient denied | ~7 |
| `fall_detected=True`, `patient_confirmed='not_answered'` | Popup timed out — treated as fall | ~8 |

So out of 1014 negatives, almost all are true negatives. False positives are extremely rare in this dataset (~7 in this snapshot). **You cannot fabricate meaningful false positive cases** — the only way to get more is real usage where patients respond "No, I didn't fall." Synthetic data can only help you with true negatives.

### Re-labelling logic (the part that matters most)

`data_pipeline.py` (lines 133–143) does this **before** training starts:

```python
# label = 1: model said fall AND patient confirmed
pos_mask = (wide_df["fall_detected"] == True) & (wide_df["patient_confirmed"] == "yes")

# label = 0: model said no fall, OR patient explicitly denied
neg_mask = (
    (wide_df["fall_detected"] == False) |
    (wide_df["patient_confirmed"] == "no")   # ← false positive re-labelled to 0
)
```

The translation from raw rows to training labels:

| Original model prediction | Patient response | Training label |
|---------------------------|------------------|----------------|
| fall | yes | **1** — confirmed fall |
| fall | no | **0** — re-labelled as non-fall |
| fall | not_answered | excluded from training |
| no fall | (no MQTT, no response) | **0** — true negative |

The model learns from ground truth (what the patient said), not from its own past predictions. The ~7 false positive cases get re-labelled to 0 and the model is penalised for having predicted 1 on those windows — in theory pushing those feature patterns away from the fall class in the next version.

In practice 7 examples is far too few to meaningfully improve false-positive reduction. Don't expect a retrain to fix the false-positive rate until you have hundreds of "no" responses.

---

## 5. The `observation_id` UUID — why it exists

You will see `observation_id` everywhere: in the `/predict` response, in the MQTT payload, in `inference_log`, in `fall_history`. It is **the key that ties an inference to its later patient confirmation** without forcing a synchronous database write inside the HTTP handler.

The flow:
1. Inference server generates a UUID at the start of every `/predict` call.
2. Returns it in the HTTP response body.
3. Background task writes it into `inference_log.observation_id`.
4. Mobile app carries it through the patient confirmation popup and includes it in the MQTT payload.
5. `fall_dashboard` reads it from MQTT and stores it in `fall_history.observation_id`.

That UUID is what makes the LEFT JOIN in section 4 work. Without it, you would need a foreign key from `fall_history` back to `inference_log.id`, which would require the mobile app to wait for the database row id — coupling we don't want.

If a fall row appears in `fall_history` with no matching `observation_id` in `inference_log`, that is a bug — most commonly the mobile app dropped the field when constructing the MQTT payload.

---

## 6. Things That Will Bite You

- **MLflow tracking URI must be reachable from the inference pod.** When deployed, `MLFLOW_TRACKING_URI=http://mlflow:5000`. In local dev it is `./mlruns/` (file store). If the pod's env says `./mlruns/` but no PVC is mounted, every restart loses your MLflow history.
- **MinIO credentials are not the same as MinIO bucket existence.** When you first deploy, the bucket `mlflow-artifacts` must already exist or MLflow will error out on the first `log_model` call. The Helm chart includes a Job that creates the bucket — verify it ran with `kubectl logs job/create-mlflow-bucket`.
- **`MLFLOW_ARTIFACT_ROOT` is not auto-applied when using a SQLite tracking URI.** `retrain.py` already works around this by passing `artifact_location` explicitly to `client.create_experiment()` on first run. If you switch the tracking URI, double-check that the first run actually writes its artifact to MinIO (look in the MinIO console, not just MLflow).
- **`--workers 1` for `mlflow ui`.** With more workers you sometimes see partial run lists due to SQLite locking.
- **The `model/` baked-in fallback can mask MLflow misconfiguration.** The inference server starts cleanly even if MLflow is broken, because it falls back to the baked-in `.pkl`. Always verify `GET /model/info` shows `mlflow:` in `loaded_as` after deploying — otherwise you're silently serving the fallback.
- **Hot-swap is per-process.** The inference server runs with `--workers 1` deliberately. If you ever scale workers, the swap will only affect the worker that received the HTTP call — the others keep serving the old model. Don't increase `--workers`.

---

## 7. Useful Files at a Glance

| File | What it does |
|------|--------------|
| `inference_server/server.py` | The FastAPI service. `/predict`, `/model/switch`, `/model/info`, `/model/list`, `/health`, `/metrics`. |
| `inference_server/services/db_writer.py` | BackgroundTask that writes `inference_log` + `feature_snapshot`. Never raises — DB write failures are logged and dropped. |
| `inference_server/services/metrics_collector.py` | Prometheus counters/histograms incremented inside `/predict`. |
| `shared_db/db/models.py` | SQLAlchemy ORM: `InferenceLog`, `FeatureSnapshot`, `FallHistory`, `ParticipantSession`. |
| `shared_db/db/migrations/versions/` | Alembic migrations. The Helm chart runs `alembic upgrade head` as a `post-install,post-upgrade` Job. |
| `retrain/data_pipeline.py` | The Postgres → wide-format DataFrame pipeline. Re-labelling lives here. |
| `retrain/retrain.py` | Training script. `--dry-run`, `--register`, `--dataset our_data\|charite`. |
| `retrain/seed_test_data.py` | `--synthetic N` (Postgres-only) or `--influxdb` (dev simulation). |
| `local_dev/dev_scripts/switch_model.ps1` | Wrapper around `POST /model/switch`. `-Stage Production` or `-Version v0_retrained`. |
| `infrastructure/docker-compose.yml` | Local stack: Postgres, MQTT, Prometheus, Grafana, MinIO. Run this before any local test. |
