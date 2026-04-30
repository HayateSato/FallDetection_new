MLflow tracking (experiments, runs, metrics, params) is in **SQLite** (`mlruns.db`) — that's intentional and separate from your main Postgres.

Two databases, two different purposes:

| Store | What's in it | Why |
| --- | --- | --- |
| **Postgres** (`fall_detection` DB) | `inference_log`, `feature_snapshot`, `fall_history`, `participant_session` | Operational data — written continuously by running services |
| **SQLite** (`mlruns.db`) | MLflow experiment runs, metrics, params, model registry | ML tracking only — written by `retrain.py`, read by MLflow UI and inference_server for model loading |

SQLite is fine for MLflow here because only one process writes to it at a time (`retrain.py` runs manually, not continuously). It requires no Docker service, no connection string management, and MLflow UI works directly against it.

In production on Kubernetes, you'd point `MLFLOW_TRACKING_URI` to a proper MLflow tracking server (`http://mlflow:5000`) backed by Postgres — but that's a deployment concern, not a code change. The `.env` comment already notes this:

`MLFLOW_TRACKING_URI=sqlite:///./mlruns.db   # change to http://mlflow:5000 in production`

Model artifacts (the `.pkl` files) are stored separately in **MinIO** (configured via `MLFLOW_S3_ENDPOINT_URL`), not in the SQLite file.

in production, what would be a problem to keep using SQLite?

Three concrete problems:

**1. File access across pods**
In Kubernetes, `mlruns.db` is a file on the local filesystem of whatever pod runs `retrain.py`. The `inference_server` pod can't see that file — it's on a different pod, potentially a different node. So `POST /model/switch {"mlflow_stage": "Production"}` would fail because inference_server can't reach the SQLite file to look up the registry.

**2. No concurrent writes**
SQLite locks the entire file on write. If you ever run two retrain jobs simultaneously (e.g. retraining v0 and v3 at the same time), one will block or crash. Not a problem now with manual runs, but becomes one if you add a CronJob.

**3. No persistence guarantees**
A pod restart or reschedule can wipe the local filesystem. You'd lose your entire experiment history and model registry unless you mount a PersistentVolume specifically for the SQLite file — which is awkward and fragile.

**The fix is one line in `.env`:**

`MLFLOW_TRACKING_URI=http://mlflow:5000`

That points both `retrain.py` and `inference_server` at a shared MLflow tracking server pod (backed by Postgres), which is already included in the Helm chart. Model artifacts stay in MinIO regardless — that part doesn't change.