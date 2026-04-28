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

---

## Hot-swap? 

"Hot-swap" means replacing a component while the system keeps running — no restart, no downtime, no dropped requests.

In your case:

Cold swap (what you're avoiding)	Hot-swap (what /model/switch does)
Stop the inference server	Server keeps running
Edit .env (MODEL_VERSION=v0_retrained)	No config change needed
Restart the server	No restart
Re-warm caches, re-establish connections	Already warm
Mobile app's /predict calls fail during downtime	/predict keeps responding the whole time
How it works in your code:

POST /model/switch triggers the server to:

Download the new .pkl from MinIO (or read from model/ folder)
Acquire a threading.Lock for ~50ms
Replace the in-memory model object with the new one
Release the lock — next /predict call uses the new model
Any /predict request that arrives during those 50ms either gets the old model (already in flight) or waits a few ms for the lock. Nothing fails.

Why it matters here: the inference server runs in a hospital monitoring context. You can't say "we're upgrading the model, no fall detection for 30 seconds." Hot-swap means the patient's safety coverage never drops.

The opposite term is "cold swap" — stopping the process entirely, swapping the component, restarting. Required for things like Python dependency upgrades or kernel changes, but not for model files.

The term comes from physical hardware (hot-swappable hard drives — pull one out, plug a new one in, server keeps serving). Same idea applied to software components.

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


----

## mlflow component 

Two separate things here. Let me be precise — my earlier wording conflated them.

**Where the model version lives right now:**

| Piece | Storage | Why |
| --- | --- | --- |
| Registry entry (`fall-detection-xgboost` v1, alias=Production, metrics, params) | **SQLite** at `mlruns.db` | Tracking backend |
| Model artifact (the `.pkl` file itself) | **MinIO** bucket `s3://mlflow-artifacts/` | Artifact store |

The version *number* you see in the MLflow UI lives in `mlruns.db`. The actual file the inference server downloads and loads lives in MinIO. Two stores, different purposes.

**Why still SQLite — the honest answer:**

When I said "your Postgres has the mlflow database ready" I meant the *empty database container* exists (created by `init.sql`). But there is no MLflow tracking server pod actually connecting to it. The full Postgres-backed setup needs three pieces:

`[retrain.py / inference_server]
        │
        │ HTTP
        ▼
[MLflow tracking server pod] ← does NOT exist yet
        │
        │ SQL
        ▼
[Postgres "mlflow" database] ← exists but empty`

Right now we have only the bottom box. The middle box — an actual `mlflow server --backend-store-uri postgresql://...` process — has not been deployed. Local dev skips the middle box entirely and writes to SQLite directly, which works because `retrain.py` and `inference_server` are on the same machine and can both reach the same file.

**To switch to Postgres you would need to:**

1. Add an `mlflow` service to `docker-compose.yml` (or the Helm chart) running `mlflow server --backend-store-uri postgresql://fall_user:fall_pass@postgres:5432/mlflow --default-artifact-root s3://mlflow-artifacts/`
2. Change `MLFLOW_TRACKING_URI=http://mlflow:5000` in `.env`
3. Re-register your models (the SQLite ones don't auto-migrate)

This is a deferred deployment task — fine to leave on SQLite while testing the switch flow locally. The pipeline you're verifying right now (registry → alias → `/model/switch`) works identically against either backend, because both `retrain.py` and `inference_server` just speak `MLFLOW_TRACKING_URI`.



--- 

## mlflow steps 

1. Find the version number


mlflow ui --backend-store-uri sqlite:///./mlruns.db --workers 1
# → http://localhost:5000
Open the Models tab → fall-detection-xgboost → note the version number (e.g. Version 1).

2. Promote that version to the Production alias


# Replace 1 with your actual version number
python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///./mlruns.db')
mlflow.tracking.MlflowClient().set_registered_model_alias('fall-detection-xgboost', 'Production', 1)
print('Done')
"
3. Hot-swap the live model on the inference server

Make sure the inference server is running on :8001, then:


.\local_dev\dev_scripts\switch_model.ps1 -Stage Production
Or the curl equivalent:


curl.exe -X POST http://localhost:8001/model/switch `
  -H "Content-Type: application/json" `
  -H "X-API-Key: 626e6c481b78c77d52db774d0e54a06cefb0553cb245fa15d5c7050fc7424e7a" `
  -d '{\"mlflow_stage\": \"Production\"}'
4. Verify


curl.exe http://localhost:8001/model/info
Look for "loaded_as": "mlflow:Production:v1(v0)" in the response. If you see something like that, the switch worked. If you see "loaded_as": "v0", it didn't pick up the registry version.

5. Roll back to confirm switching works both ways


.\local_dev\dev_scripts\switch_model.ps1 -Version v0
curl.exe http://localhost:8001/model/info   # should now show "loaded_as": "v0"


--- 

## structures between retraining and model switch via mlflow

## Conceptual model — your two-layer mental model needs adjustment

Your version: "Layer 1 = models that might be used / Layer 2 = trained but not used."

The real structure has **three things**, not two layers:

`┌─────────────────────────────────────────────────────────────┐
│ EXPERIMENT RUNS  (every training attempt — kept forever)    │
│   run-abc1: recall=0.82, AUC=0.91, .pkl in MinIO            │
│   run-abc2: recall=0.79, AUC=0.88, .pkl in MinIO            │
│   run-abc3: recall=0.85, AUC=0.93, .pkl in MinIO   ★        │
│   run-abc4: recall=0.91, AUC=0.95, .pkl in MinIO   ★        │
└─────────────────────────────────────────────────────────────┘
          ★ = registered (you ran retrain.retrain --register)
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ MODEL REGISTRY: "fall-detection-xgboost"                    │
│   Version 1 ← from run-abc3                                  │
│   Version 2 ← from run-abc4         ◄── alias: Production   │
└─────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                                [inference_server loads this]`

**Three concepts:**

1. **Experiment runs** — every `retrain.retrain` call creates one. ALL runs stay forever. `-register` ALSO copies that run into the registry.
2. **Registered model versions** — Version 1, 2, 3, … under the name `fall-detection-xgboost`. Once registered, a version stays forever (until you explicitly delete it).
3. **Aliases** — *labels that point to one specific version*. `Production` is just a name. When you "promote v2 to Production," you're moving the label from v1 to v2. Nothing is deleted. v1 is still registered, still in MinIO.

**Answers to your specific questions:**

- *"Is registration = promotion?"* — **No, two different actions.**
    - Register = "add this run to the registry as the next version number" (`-register` flag)
    - Promote = "attach the Production alias to this version" (UI click or Python one-liner)
- *"Where do old/switched models go after I switch?"* — **Nowhere. They stay registered.** When you switch from v1 → v2:
    - v1 is still Version 1 in the registry
    - v1's `.pkl` is still in MinIO
    - v1 simply has no alias on it anymore (or you can give it `Archived`)
    - You can switch back to v1 anytime via `{"version": 1}` or by moving the Production alias back
- *"Are unused trained models in a separate layer?"* — **No.** Whether something is "in use" is determined by which version the Production alias points to. There's no separate bucket for unused models — they're all registered versions, just without the Production label.

The mental model is: **one registry, many versions, aliases are pointers.** Switching = moving a pointer, not moving a model.

in experiment runs bucket = all models trained 
inside it, there is another bucket called model registry = all models ready to be used 
isnide it, you can tell mlflow to pick one model (any model that is assigned as "Production" as its alias is the choosen one?)

is my paraphrasing correct?

Show less

Yes — your paraphrasing is correct. Let me just sharpen one detail on the third bullet:

| Layer | What it holds | How something gets in |
| --- | --- | --- |
| **Experiment runs** | All models ever trained | Every `retrain.retrain` call lands here automatically |
| **Model Registry** | Models ready to be used | You explicitly opt in with `--register` (or click "Register Model" in the UI) |
| **Aliases** | Pointer to which version is "the chosen one" | You manually assign — UI click or Python one-liner |

**One tweak on bullet 3:** MLflow doesn't decide anything. *You* tell MLflow which version to label `Production`, and the inference server then asks "which version has the Production alias?" when you call `/model/switch {"mlflow_stage": "Production"}`. So:

- You assign the alias → MLflow just stores the pointer
- inference_server reads the pointer when asked → loads that version's `.pkl` from MinIO

Multiple aliases can coexist on different versions — `Production` on v3, `Staging` on v5 (under evaluation), `Archived` on v1 (kept for rollback). The aliases are arbitrary labels; `Production` is convention, not a built-in concept.

So the full mental picture, with your wording:

`Experiment Runs  (all trained models — every retrain run)
   │
   │  --register  (you opt in)
   ▼
Model Registry   (models ready to be used)
   │
   │  set_registered_model_alias("Production", v3)   ← you choose
   ▼
Production alias points to v3
   │
   │  /model/switch {"mlflow_stage": "Production"}
   ▼
inference_server loads v3's .pkl from MinIO`

---

Quick terminology check first — MLflow 3.x renames a few things:

| What you said | Correct term | What it means |
| --- | --- | --- |
| Retraining | retraining | Running training again on new data |
| Registration | **registering** a model | Creating a Version in the registry |
| Staging | applying the **Staging alias** | Labelling a version as "under review" |
| Promoting | applying the **Production alias** | Labelling the current best version |

Stages were a separate concept in MLflow 2.x and got replaced by aliases in 3.x. In your UI you'll see "Aliases" buttons, not "Transition to Production."

---

## What can be done from the UI

| Step | UI? | How |
| --- | --- | --- |
| Retrain | ❌ | Terminal only — `python -m retrain.retrain` |
| Register a run as a Model Version | ✅ | Run page → Artifacts → click **Register Model** |
| Apply Staging / Production alias | ✅ | Models page → version → **+ Alias** → type the name |
| Remove an alias | ✅ | Same dropdown → click the X next to the alias |
| Hot-swap the live inference server | ❌ | Terminal only — `switch_model.ps1` or curl `/model/switch` |

The UI is for **registry management**. It cannot start training jobs and cannot tell the inference server to load a different model — those are separate concerns.

---

## UI walkthrough — what to click

**To register a new run as a Model Version (replaces `--register`):**

1. Open [http://localhost:5000](http://localhost:5000/)
2. Click the experiment (e.g. `fall-detection-v0`)
3. Click your run name (the green-check row)
4. Scroll to the **Artifacts** section
5. Click on the model folder (e.g. `model`)
6. Click the **Register Model** button (top right of the artifact panel)
7. Select existing model `fall-detection-xgboost` → Register

→ A new Version is created. You can now skip the `--register` CLI flag if you prefer the UI workflow.

**To apply a Staging alias (review checkpoint):**

1. **Models** tab (left sidebar) → click `fall-detection-xgboost`
2. Find the Version row you want to mark
3. In the **Aliases** column, click **+ Alias**
4. Type `Staging` → Save

→ The version is now your "candidate." Mock_app or staging environment can load it via `{"mlflow_stage": "Staging"}` in `/model/switch`.

**To promote to Production:**

1. Same Models page
2. Click **+ Alias** on the version that passed evaluation
3. Type `Production` → Save

→ This is the click-equivalent of your Python one-liner. The previous Production version still exists; the alias just points to a different row now.

**To roll back:**

1. Same Models page
2. Click the X next to `Production` on the current version
3. Click **+ Alias** on the older version → type `Production`

→ One alias only ever exists once — moving it from one version to another is the rollback.

---

## What the UI does NOT do

After every alias change, you still need to tell the running inference server to reload:

`.\local_dev\dev_scripts\switch_model.ps1 -Stage Production`

Without this, the server keeps running whatever model it loaded on its last `/model/switch` call. The MLflow UI manages "what's the desired version?" — your inference server reads "what's currently loaded?" Two different states.

In production this gap is usually closed in one of two ways:

- A small admin button on your dashboard that POSTs `/model/switch` (one-click reload)
- An automatic mechanism: inference server polls `Production` alias every N seconds and self-reloads on change

Both are future work. For now, the UI handles registry, the CLI handles the swap.

---


## 1. The two "Production" tags are NOT the same

Look closely at the image:

- Version 3: `@ production` (lowercase **p**)
- Version 1: `@ Production` (capitalized **P**)

These are **two different aliases** because alias names are case-sensitive. You probably typed them differently when assigning them. MLflow treats `production` and `Production` as completely separate labels.

**The hot-swap rule:** an alias name is **unique per registered model**. If you assign `Production` to Version 2, MLflow *automatically* removes it from Version 1. So you can never have two versions both tagged `Production` — but you can have `Production` on one and `production` on another, which is what you're seeing.

**What `switch_model.ps1 -Stage Production` does:** sends `{"mlflow_stage": "Production"}` (capitalized) to `/model/switch`. The server asks MLflow "which version has the `Production` alias?" → gets Version 1 → loads it. It never sees Version 3 because that's tagged `production`, not `Production`.

**Fix:** delete the lowercase one. In the UI, click the X next to `@ production` on Version 3, then add `Production` (capitalized) if that's the version you actually want live.

---

## 2. Staging vs Production — what's the difference?

**Technically: nothing.** Both are arbitrary text labels with no built-in behaviour. You could call them `alpha` and `beta` and the system would work identically. The names are conventions, not enforced concepts.

**The convention people follow:**

| Alias | Convention meaning | Who reads it |
| --- | --- | --- |
| `Staging` | "Under evaluation — not yet trusted with live traffic" | A staging copy of the inference server, or mock_app |
| `Production` | "Currently serving real users" | The live inference server that real patients depend on |

**Why this matters in practice:** you'd run *two* inference server instances:

- A staging instance polling `Staging` alias — used for shadow testing or QA
- A production instance polling `Production` alias — serves real traffic

You promote a model by first applying `Staging`, watching it for a few days against real-world data, then moving the `Production` alias to the same version once you're confident.

**For your local dev:** you only have one instance, so `Staging` is unused. Once you have FOCUS data and a real cluster, you'd add a second deployment that polls Staging.

---

## 3. ml_dashboard idea + Grafana opinion

## My opinion on the Grafana question

**Keep Grafana, don't try to replace it.** Here's the breakdown:

| Tool | What it's good at | Effort to replicate |
| --- | --- | --- |
| **Grafana** | Time-series queries against Prometheus, alerting, drift detection, dashboard layouts you didn't write | Hundreds of hours; you'd build a worse version |
| **MLflow UI** | Experiment comparison tables, registry management, alias controls | Possible to replicate but ~50% of the value is the run-comparison UI which is non-trivial |
| **ml_dashboard (proposed)** | The 5 buttons you actually click during a model rollout | Genuinely small — a weekend project |

The "three UIs" pain isn't really three UIs — it's two UIs (Grafana + MLflow) plus a missing one (ml_dashboard). The reason it feels like too many is that today you have to **leave** the dashboards to type commands in a terminal. ml_dashboard fixes that gap. It doesn't need to *contain* Grafana or MLflow; it just needs to give you the actions you currently type by hand.

**The trick:** ml_dashboard cross-links into the others.

`ml_dashboard
├── "Retrain" button             → starts job, shows logs
├── "Promote v3 → Production"    → MLflow alias change + auto hot-swap
├── "Currently live: v2"         → shows server state
├── Link: "View metrics"         → opens Grafana on the relevant panel
└── Link: "Compare runs"         → opens MLflow experiment view`

You stay on ml_dashboard for actions. You hop to Grafana when you need to *investigate* (latency drift, fall-rate spike). You hop to MLflow when you need to *compare* (recall vs F1 across 10 runs). Each tool does what it's best at — none of them are redundant.

**Where Grafana actually overlaps** with ml_dashboard is the "is the new model behaving sensibly?" check after a hot-swap. For that one specific question, you could embed two Prometheus queries (latency p95, confidence median over the last 10 minutes) directly in ml_dashboard. That's a small win and avoids the context-switch *only at the moment it matters most* — right after pressing the hot-swap button. Everything else, leave in Grafana.

