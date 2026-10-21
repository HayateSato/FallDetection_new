# Runbook — Retrain and Hot-Swap the Production Model

**Audience:** Admin / MLOps operator on the running 6G fall-detection stack.
**Use this doc when:** you have decided a retrain is needed and you want to walk through it step by step.
**This is a runbook**, not an explainer — for the *why* and the conceptual model see [`ADMIN_ml_ops_related.md`](ADMIN_ml_ops_related.md).

---

## Pre-flight — confirm the stack is healthy

Before you retrain, verify the four moving parts are up.

| What to check | Where | How | Expected |
|---------------|-------|-----|----------|
| Postgres | Docker | `docker ps \| findstr postgres` | `Up`, healthy |
| MinIO | Docker | `docker ps \| findstr minio` | `Up`, healthy |
| MLflow tracking server | Docker | `docker ps \| findstr mlflow` | `Up` (not `Restarting`) |
| Inference server | host venv / pod | `curl.exe http://localhost:8001/health` | 200 OK with current `model_version` |

If any are down, fix that first — do not start retraining.

**Things to clarify if you're not sure:**
- Are you in local dev or production? Local dev runs from `_6G_Integration_v2_mqtt/` with `docker-compose -f infrastructure/docker-compose.yml up`. Production runs in Kubernetes — same flow but commands target the cluster.
- Which MLflow are you talking to? Check `MLFLOW_TRACKING_URI` in `.env`. If it's `http://localhost:5000` you're using the dockerised MLflow. If it's `sqlite:///./mlruns.db` you're on the legacy file-based setup.

---

## Step 1 — Decide whether retrain is even worthwhile

**Where:** Postgres + Grafana
**How:**

```powershell
python -m retrain.retrain --dry-run
```

Reads how many labelled rows exist in Postgres without training anything.

**Verify before continuing:**
- At least **50 confirmed-fall rows** (positives). Below this, metrics are unreliable.
- The dataset includes recent data — not just rows from weeks ago. Check `detection_time` in the SQL output if unsure.

**Things to clarify:**
- Is there a *signal* that retraining is needed? Open Grafana → `model_performance` dashboard. Triggers: confidence clustering near 0.5, fall rate per hour drifting from baseline, low-confidence ratio rising above ~20%.
- If you have no operational signal and the data has not changed meaningfully, you may not need to retrain at all.

---

## Step 2 — Train and register

**Where:** host machine (where the venv lives)
**How:**

```powershell
python -m retrain.retrain --model-version v0 --dataset our_data --register
```

What this does:
1. Pulls labelled data from Postgres
2. Trains XGBoost (80/20 stratified split)
3. Logs run + metrics to MLflow
4. Saves `.pkl` to `model/model_v0_retrained/`
5. Registers in MLflow as `fall-detection-xgboost` Version N (next free number)

**Verify after running:**
- Open MLflow UI: http://localhost:5000 → Experiments tab → your run shows `recall`, `auc`, `f1` columns populated.
- Models tab → `fall-detection-xgboost` → new Version N is listed.
- MinIO console (http://localhost:9002) → bucket `mlflow-artifacts` → folder corresponding to the new run contains `model.pkl`.

**Things to clarify:**
- **Did the metrics improve?** New recall must be **≥ current production recall**. If lower, do not promote — stop here.
- **Did the registry version actually appear?** If the run finished but no version shows up under Models, your `--register` flag was missing or MLflow URI was wrong. Re-run with `--register` explicitly.
- **Which version number do I use later?** Look at the "Version" column in the Models tab — that's the number you pass to `set_registered_model_alias` in step 3.

---

## Step 3 — Promote to Production

**Where:** terminal (the MLflow UI's "Stages" button is deprecated as of MLflow ≥ 2.9, replaced by aliases)
**How:**

```powershell
# Replace 2 with the version number from step 2
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.tracking.MlflowClient().set_registered_model_alias(
    'fall-detection-xgboost', 'Production', 2)
print('Done')
"
```

This points the alias `Production` at Version 2. The previous Production version is still registered — only the alias pointer moves.

**Verify after running:**
- MLflow UI → Models → `fall-detection-xgboost` → Version 2 row should show `@ Production` in the Aliases column.
- The previous version no longer shows `@ Production` (the alias is unique per model — moves automatically).

**Things to clarify:**
- **Watch the case.** `Production` ≠ `production`. Aliases are case-sensitive. The hot-swap script in step 4 only matches `Production` (capitalised). If you accidentally typed `production`, delete that alias in the UI and redo this step.
- **Want a Staging step first?** Set the `Staging` alias on Version 2, point a non-prod inference server at it, watch for a few hours, then move `Production` only if behaviour is sane. In a single-instance dev environment this step is skipped.

---

## Step 4 — Hot-swap the live inference server

**Where:** terminal (any machine that can reach the inference server)
**How:**

```powershell
.\local_dev\dev_scripts\switch_model.ps1 -Stage Production
```

Or curl equivalent:

```powershell
curl.exe -X POST http://localhost:8001/model/switch `
  -H "Content-Type: application/json" `
  -H "X-API-Key: <YOUR_API_KEY>" `
  -d '{\"mlflow_stage\": \"Production\"}'
```

The inference server downloads `.pkl` from MinIO and replaces the in-memory model under a thread lock. **No restart, no downtime** — `/predict` keeps serving on the old model until the new one is ready.

**Verify after running:**

```powershell
curl.exe http://localhost:8001/model/info
```

Expected: `"loaded_as": "mlflow:Production:v2(v0)"` — version 2, base type v0.

If you see `"loaded_as": "v0"` (without `mlflow:`) the swap silently fell back to the file-baked startup model. Check the inference server logs for an MLflow connection error. Common causes:
- `MLFLOW_TRACKING_URI` not set in the inference server's environment
- MinIO credentials missing or wrong
- The bucket `mlflow-artifacts` does not exist
- Network: in K8s, the inference pod cannot reach the `mlflow` service

**Things to clarify:**
- **Hot-swap is in-memory only — does not survive restart.** If the inference pod restarts (crash, K8s reschedule, image upgrade), it will boot back into whatever `MODEL_VERSION` says in `.env`. To make Production sticky across restarts you need to either re-call `/model/switch` on every boot (e.g. as a startup hook) or change the loader to resolve `MLFLOW_TRACKING_URI` at startup. Currently neither is implemented — see todo.md Step 11.5.
- **`/predict` calls during the swap?** They either get the old model (already in flight) or wait ~50ms for the lock. Nothing fails.

---

## Step 5 — Validate in production

**Where:** Grafana + a few synthetic `/predict` calls
**How:**

1. Run mock_app or send a few real `/predict` calls. Watch confidence distributions in Grafana → `model_performance` dashboard.
2. Wait at least 1 day with the new model in production before declaring victory.

**Verify after running:**
- Confidence distribution looks reasonable (no clustering near 0.5)
- Fall rate per hour stays in a believable range
- p95 latency stays under 100ms (`ml_server_overview` dashboard)
- No spike in error rate

**Things to clarify:**
- **What if it gets worse?** Roll back immediately — see "Rollback" below. Do not wait to see if it recovers.

---

## Rollback

If the new model misbehaves:

```powershell
# Option A — go back to the immediately previous registry version
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.tracking.MlflowClient().set_registered_model_alias(
    'fall-detection-xgboost', 'Production', 1)   # the version that was working before
print('Done')
"
.\local_dev\dev_scripts\switch_model.ps1 -Stage Production

# Option B — drop back to the file-baked baseline
.\local_dev\dev_scripts\switch_model.ps1 -Version v0
```

Verify with `curl.exe http://localhost:8001/model/info` after either option.

---

## Common failure modes — quick lookup

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `python -m retrain.retrain` says "0 positive rows" | Postgres empty or wrong `DATABASE_URL` | Check `.env`, run `python -m retrain.seed_test_data --synthetic 200` for testing |
| `--register` ran but no version appears in MLflow UI | `MLFLOW_TRACKING_URI` mismatched between client and server | Check `.env`; client and server must be the same MLflow instance |
| `mlflow.exceptions.MlflowException: API ... 404 != 200` | MLflow client version newer than server | Pin both to the same version in `infrastructure/mlflow/Dockerfile` and `requirements.txt` |
| `/model/switch` returns 200 but `/model/info` still shows old version | Server restart happened between switch and check, or wrong worker | Re-run the switch; verify `--workers 1` is set on inference server |
| `psycopg2.errors.StringDataRightTruncation: ... character varying(20)` on `model_version` | Migration 0002 not applied to this DB | `alembic upgrade head` from `_6G_Integration_v2_mqtt/` |
| `pkg_resources` ModuleNotFoundError when starting MLflow container | Python 3.12 base image stripped setuptools | Use `python:3.11-slim` in `infrastructure/mlflow/Dockerfile` |

---

## Decision summary — what kind of change can use this runbook?

| Change | Use this runbook? | Why |
|--------|:-----------------:|-----|
| Retrain on more data, same features | yes | Identical input contract — pure swap |
| Tune hyperparameters | yes | Identical input contract — pure swap |
| Add a new feature (e.g. barometer) | **no** | Input contract changes — coordinated rollout with mobile app, code changes, new registry name |
| Switch model framework (XGBoost → neural net) | yes (with care) | MLflow stores any framework, but the loading logic in inference_server may need adjustment |
| Change feature units or scaling | **no** | Silent behaviour change — full rollout, not a hot-swap |

If your change is in the "no" column, this runbook is the wrong tool. See [`ADMIN_ml_ops_related.md`](ADMIN_ml_ops_related.md) section 6 ("Things That Will Bite You") and the "Adding a new sensor" discussion for how to plan a contract change.

---

## Open questions to track

- [ ] When `ml_dashboard` (Step 11.5) is built, steps 2–4 of this runbook collapse into three button clicks. Update this doc when that happens.
- [ ] Hot-swap persistence across pod restarts is not implemented. Until it is, every restart silently reverts to the `MODEL_VERSION` in `.env`. Document a startup hook (or `MLFLOW_TRACKING_URI`-based loader) when added.
- [ ] Automatic Grafana-driven retraining triggers are deferred — currently fully manual. If/when an alerting threshold is wired to a CronJob, document the override here.
