# ml_dashboard — admin UI for retrain + hot-swap

Standalone FastAPI app for the model lifecycle: trigger retrain, register, set
the `Production` alias, and hot-swap the live inference server. Replaces the
terminal-only flow with click-driven actions.

| | |
|---|---|
| Port | `8004` |
| Run | `python -m ml_dashboard.main` |
| Audience | **admin only** (warning banner shown — auth gate deferred per todo 11.5.4) |
| Production target | runs in `mcs-fall-detection` namespace, exposed at `/admin/ml` behind ingress |

---

## What it does

```
operator clicks "Start retrain"
   ▼
ml_dashboard spawns subprocess: python -m retrain.retrain ...
   ▼
retrain trains, logs to MLflow, registers as Version N
   ▼
operator picks the version in the UI, clicks "Set Production"
   ▼
ml_dashboard calls MLflow API: set_registered_model_alias("Production", N)
   ▼
operator clicks "Swap to Production alias"
   ▼
ml_dashboard POSTs /model/switch to inference_server
   ▼
inference_server downloads .pkl from MinIO, atomically swaps in-memory model
```

Same flow as the runbook in `handover_docs/ADMIN_runbook_retrain_and_hotswap.md`,
just collapsed into three button clicks.

---

## REST API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | dashboard HTML (current state, retrain panel, versions table, hot-swap, drift guide) |
| GET | `/api/status` | currently-loaded model + Production alias version + drift warning |
| GET | `/api/versions` | all registered versions of `fall-detection-xgboost` with aliases |
| POST | `/api/retrain` | spawn `retrain.retrain` subprocess; returns `job_id` |
| GET | `/api/retrain/{job_id}` | poll status + accumulated stdout |
| POST | `/api/promote` | `set_registered_model_alias(name, alias, version)` |
| POST | `/api/switch` | proxy to inference_server `/model/switch` |

Subprocess output is captured into an in-memory ring buffer keyed by `job_id`
and polled at 1.5s intervals from the browser. Survives only while the
`ml_dashboard` process is running.

---

## Files

```
ml_dashboard/
├── main.py              ← uvicorn entry + banner + signal handling
├── web.py               ← FastAPI app + all routes + subprocess + MLflow client
├── dashboard/
│   ├── index.html       ← layout: status panel, playbook, retrain, versions, hot-swap, drift guide
│   ├── app.js           ← polling, button handlers, in-place cell updates
│   └── style.css        ← dark theme matching fall_dashboard
├── Dockerfile
└── requirements.txt
```

The Dockerfile copies `retrain/`, `shared_db/`, `ml_pipeline/`, `config/`, and
`model/` alongside `ml_dashboard/` because retrain runs as a subprocess from
inside the container — it needs the same code/data layout the venv has.

---

## Configuration

| Var | Default | Purpose |
|-----|---------|---------|
| `ML_DASHBOARD_HOST` | `0.0.0.0` | bind address |
| `ML_DASHBOARD_PORT` | `8004` | HTTP port |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | tracking server (Postgres-backed in production) |
| `INFERENCE_SERVER_URL` | `http://localhost:8001` | target for `/model/switch` calls |
| `INFERENCE_API_KEY` | (required for production) | sent as `X-API-Key` to inference_server |
| `MLFLOW_REGISTERED_MODEL` | `fall-detection-xgboost` | name in MLflow registry |

---

## Production safety

This UI controls the live model serving real patients. Before exposing it
beyond localhost:

- [ ] Auth gate — JWT with `role=admin` claim (todo.md 11.5.4)
- [ ] Audit log — record every retrain trigger / alias change / hot-swap with operator identity
- [ ] Confirmation dialogs already exist on `Set Production` and on hot-swap

The warning banner at the top of the page reflects the current state:
*"Auth not yet enforced — local / cluster-internal use only."*

---

## What it does NOT do

- Metric comparison → use **MLflow UI** at `http://<mlflow>:5000` (link from playbook)
- Drift / latency / fall rate → use **Grafana** at `http://<grafana>:3000`
- Pod / service status → use **server_health** at port 8006

ml_dashboard is the **action** layer; investigation lives in those other tools.
The playbook section at the top of the page explicitly tells the operator
which action lives here vs which lives elsewhere.
