---
name: 6G Charite/FOCUS Integration — MQTT version current state
description: Current folder structure, what's implemented, run commands, and open items for the MQTT-based partner integration
type: project
---

## Folder structure (updated 2026-04-27)

`_6G_Integration_v2_redis/` — original Redis version (reference / frozen)
`_6G_Integration_v2_mqtt/`  — active working version

**Folder layout rule (2026-04-27):** everything at the root of `_6G_Integration_v2_mqtt/` ships to production. `local_dev/` never ships.

```
_6G_Integration_v2_mqtt/
  inference_server/      ← production service #1 (:8001) — has README.md
  fall_dashboard/        ← production service #2 (:8002) — has README.md (added 2026-04-28)
  ml_dashboard/          ← admin UI :8004 — retrain + promote + hot-swap (added 2026-04-28) — has README.md
  server_health/         ← admin UI :8006 — aggregate /health probes (added 2026-04-28) — has README.md
  ml_pipeline/           ← ML signal processing + inference engine (was: app/)
  shared_db/             ← SQLAlchemy ORM models + session factory (was: shared/)
  config/                ← hardware + app settings
  model/                 ← .pkl model files
  retrain/               ← MLflow retraining pipeline
  helm/
    fall-detection/      ← production Helm chart (ships to FOCUS)
    mock-focus/          ← DRY-RUN ONLY — simulates FOCUS namespace for cross-NS testing (added 2026-04-28)
  infrastructure/        ← Docker Compose (local dev)
    mlflow/              ←   custom Dockerfile for MLflow tracking server (added 2026-04-28)
  local_dev/             ← LOCAL TESTING ONLY — never ships to K8s
    mock_app/            ←   simulates SmarKo mobile app
    mock_focus/          ←   simulates FOCUS FHIR server
    dev_scripts/         ←   developer CLI tools (was: scripts/)
      switch_model.ps1
```

Key renames (2026-04-27): `app/` → `ml_pipeline/`, `shared/` → `shared_db/`, `scripts/` → `local_dev/dev_scripts/`
Run mock_app: `python -m local_dev.mock_app.main` (from `_6G_Integration_v2_mqtt/` as cwd)
Run switch_model: `.\local_dev\dev_scripts\switch_model.ps1 -Stage Production`

## Architecture — _6G_Integration_v2_mqtt

```
mock_app/                  ← simulates mobile app
  influx_fetcher.py        ← queries InfluxDB for raw ACC data
  api_caller.py            ← POSTs to inference server /predict via HTTP
  poller.py                ← polling loop; on fall: routes through patient_server, publishes fall/alert to MQTT
  patient_server.py        ← patient confirmation popup server at :8005 (browser UI for local dev)
  main.py                  ← entry point: python -m local_dev.mock_app.main

inference_server/          ← FastAPI :8001 — HTTP only, NO MQTT client
  server.py                ← receives /predict, runs XGBoost, returns result + observation_id in HTTP response
  services/
    metrics_collector.py   ← Prometheus metrics (fall_detections_total, latency, confidence)
    db_writer.py           ← BackgroundTask write of inference_log + feature_snapshot to Postgres

fall_dashboard/            ← FastAPI :8002
  mqtt_listener.py         ← FallEventBroker: subscribes to fall/alert/# → on_fall callback → SSE
  main.py                  ← wires on_fall (DB write + SSE fan-out), starts broker
  web.py                   ← dashboard + /api/stream SSE + /api/falls + /api/patients
  db.py                    ← imports from shared.db.models; writes fall_history + participant_session

shared/db/
  models.py                ← SQLAlchemy ORM: InferenceLog, FeatureSnapshot, FallHistory, ParticipantSession
  session.py               ← SessionLocal factory; get_db(); init_db(); SQLite default / Postgres via env
  migrations/              ← Alembic: env.py + versions/0001_initial_schema.py

retrain/
  data_pipeline.py         ← reads Postgres (JOIN + pivot); returns labelled DataFrame
  retrain.py               ← train XGBoost + log to MLflow (CLI script)
  seed_test_data.py        ← seed Postgres for testing: --synthetic N or --influxdb
  requirements.txt         ← mlflow>=2.10, xgboost, scikit-learn, etc.
```

mock_focus/                ← simulates FOCUS namespace (local dev only — never ships to K8s)
  fhir_server.py           ← FastAPI :8003 — FHIR R4 Patient + Observation endpoints (synthetic data)
  requirements.txt

**Deleted:** `influx_marker_writer.py` — removed; colleague writes InfluxDB markers directly from their side.

## MQTT client count: 2

| Component | Role | Topic |
|-----------|------|-------|
| mock_app | publisher | `fall/alert/<patient_id>` |
| fall_dashboard | subscriber | `fall/alert/#` |

Inference server has **no MQTT client** — fall result is returned in the HTTP response to the mobile app.

## Confirmed alert flow

```
mock_app → HTTP POST /predict → inference_server → HTTP response (fall_detected=True, observation_id=<UUID>)
mock_app → patient_server.notify_fall(event)   ← browser popup at http://localhost:8005/ (Yes/No, 10s countdown)
mock_app → patient_server.wait_for_response()  ← blocks until patient responds or timeout
mock_app → PUBLISH fall/alert/<patient_id>  payload includes: observation_id, patient_confirmed, needs_help
              → [broker] → fall_dashboard → DB write (fall_history, always)
                                          → SSE fan-out to caregiver dashboard (conditional — see below)
```

## Caregiver alert filtering (fall_dashboard/main.py — added 2026-04-27)

All confirmed MQTT alerts are written to Postgres `fall_history` regardless of outcome (needed for retraining labels).
SSE fan-out to the caregiver dashboard only fires when caregiver action is needed:

| Condition | Stored in DB | Shown on dashboard |
|-----------|:------------:|:------------------:|
| `patient_confirmed="not_answered"` | yes | **yes** — patient couldn't respond, assume serious |
| `patient_confirmed="yes"` + `needs_help=True` | yes | **yes** — patient confirmed + asked for help |
| `patient_confirmed="yes"` + `needs_help=False` | yes | no — patient confirmed but says they're okay |
| `patient_confirmed="no"` | yes | no — false positive, no caregiver action needed |

## observation_id cross-reference key

UUID generated at the start of every `/predict` call. Returned in the HTTP response. Included in MQTT alert payload by mock_app. Stored in both `inference_log.observation_id` and `fall_history.observation_id`. Enables the retraining JOIN without a synchronous DB write inside the HTTP handler.

Retraining query:
```sql
SELECT il.*, fs.feature_name, fs.feature_value, fh.patient_confirmed, fh.needs_help
FROM inference_log il
JOIN feature_snapshot fs ON fs.inference_id = il.id
JOIN fall_history fh      ON fh.observation_id = il.observation_id
WHERE il.fall_detected = TRUE AND fh.patient_confirmed = 'yes'
```

## Run order (6 terminals, from _6G_Integration_v2_mqtt/ as cwd)

```powershell
# 1 — Infrastructure (Postgres, MQTT, MinIO, MLflow, Prometheus, Grafana)
docker-compose -f infrastructure/docker-compose.yml up

# 2 — Inference server
uvicorn inference_server.server:app --host 0.0.0.0 --port 8001

# 3 — Fall dashboard (caregiver view)
python -m fall_dashboard.main         # → http://localhost:8002/

# 4 — Mock mobile app (InfluxDB → /predict → MQTT alert)
python -m local_dev.mock_app.main
# Patient popup also auto-starts at http://localhost:8005/

# 5 — ml_dashboard (admin UI for retrain + hot-swap)
python -m ml_dashboard.main           # → http://localhost:8004/

# 6 — server_health (admin status dashboard, added 2026-04-28)
python -m server_health.main          # → http://localhost:8006/
```

Each component has its own `README.md` covering endpoints, env vars, files,
and production notes — the central README links into them.

## Test retraining pipeline (no Charite data needed)

```powershell
pip install -r retrain/requirements.txt
python -m retrain.seed_test_data --synthetic 100 --model-version v3
python -m retrain.retrain --dry-run
python -m retrain.retrain --model-version v3 --dataset our_data
mlflow ui --backend-store-uri ./mlruns   # → http://localhost:5000
```

Data source for retraining: **Postgres only** (feature_snapshot + fall_history). InfluxDB is NOT in the retraining loop.

**Why not needed for retraining**: inference_server already computes features at prediction time and writes them to Postgres (feature_snapshot). Retraining is a SQL JOIN — no re-running feature extraction, no InfluxDB access.

**seed_test_data --influxdb**: development utility only. Replays our own cloud InfluxDB bucket to simulate what Postgres would look like after live inference. NOT part of the production retraining flow — in production, Postgres is populated automatically by the running inference_server.

## InfluxDB — two instances, critical distinction (confirmed 2026-04-27)

| Instance | What it is | Role in our system |
|----------|-----------|-------------------|
| **FOCUS-hosted InfluxDB** | Runs in FOCUS namespace | Written by real mobile app (biosignals). Read by FOCUS patient dashboard backend for HR display. **Never touched by our code — not in our config, not in our env.** |
| **Our cloud InfluxDB** (`ecosystem-influxdb.smarko-health.de`) | MCS/SmarKo cloud | Used **only by `local_dev/mock_app`** as a dev workaround — mock_app cannot receive BLE data from the real wearable, so it fetches from InfluxDB instead. **Completely absent in production.** |

**In production:** real mobile app reads raw ACC from wearable via BLE → POSTs directly to `/predict`. No InfluxDB query anywhere in our inference pipeline.

**Handover implication:** FOCUS does not need to give us any InfluxDB credentials. We do not need theirs. There is no InfluxDB config to exchange at handover time.

## helm/fall-detection chart status (2026-04-29 — SMOKE TEST PASSED)

The production chart at `_6G_Integration_v2_mqtt/helm/fall-detection/` is templated,
`helm lint`-clean, and smoke-tested on Docker Desktop K8s. **10 services** (was 8 — ml-dashboard
and server-health added 2026-04-29).

**Services in chart:**

| Service | Port | Template folder |
|---------|------|----------------|
| inference-server | 8001 | templates/inference-server/ |
| fall-dashboard | 8002 | templates/fall-dashboard/ |
| ml-dashboard | 8004 | templates/ml-dashboard/ (added 2026-04-29) |
| server-health | 8006 | templates/server-health/ (added 2026-04-29) |
| mqtt-broker | 1883/9001 | templates/mqtt-broker/ |
| postgres | 5432 | templates/postgres/ |
| mlflow | 5000 | templates/mlflow/ |
| minio | 9000 | templates/minio/ |
| prometheus | 9090 | templates/prometheus/ |
| grafana | 3000 | templates/grafana/ |

**configmap.yaml additions (2026-04-29):** `INFERENCE_SERVER_URL` and `FALL_DASHBOARD_URL`
added so server-health and ml-dashboard can reach other services via K8s DNS without
hardcoding URLs in their templates.

**Fixes applied during testing (2026-04-29):**
- values.yaml `namespaces.ours`: `fall-detection` → `mcs-fall-detection` (namespace mismatch)
- values.yaml `mlflow.limits.memory`: 1Gi → 2Gi (gunicorn workers OOM-killed)
- Deployment templates: image name uses `registry.example.com/` prefix; local dev requires `docker tag` to match. See project_k8s_local_testing.md gotcha #5.
- migrate-job.yaml: added explicit `imagePullPolicy: {{ .Values.images.pullPolicy }}` (K8s defaults to `Always` for `latest` tags — gotcha #6)
- migrate-job.yaml: `hook-delete-policy` changed to `before-hook-creation,hook-succeeded` (gotcha #7)
- fall-dashboard deployment: added `initContainer` running `alembic upgrade head` before main app starts — fixes race condition where `init_db()` created tables before the post-install alembic hook ran (gotcha #8)
- helm/mock-focus/test.ps1: replaced `wget` with `python3 urllib` for python:3.11-slim containers, `curl` for influxdb (gotcha #9)

**mock-focus dry-run (Step 12.5): COMPLETE (2026-04-29)**
- install.ps1: both namespaces up, all 10 pods in mcs-fall-detection + 3 in mock-focus Running ✓
- mock-focus test.ps1: 4/4 cross-namespace tests PASS ✓
- fall-detection test.ps1 (added 2026-04-29): 8/8 in-cluster probes PASS — confirms all gotcha #12-16 fixes are still working together ✓
- Manual SSE test: MQTT alert delivered, alert shown on caregiver dashboard, fall_history written to Postgres, fall history tab updated, click-to-acknowledge clears card colour ✓
- NetworkPolicy: skipped — Docker Desktop CNI does not enforce NetworkPolicy (informational only)

**Full local-equivalent flow verified in K8s (2026-04-29):**
- ml-dashboard retrain (UI button → subprocess `python -m retrain.retrain` → MLflow run logged in mlflow pod) ✓
- ml-dashboard hot-swap to new model + rollback to v0 (both via in-cluster `/model/switch`) ✓
- server-health all 6 probes return healthy (uses K8s DNS via INFERENCE_SERVER_URL / FALL_DASHBOARD_URL) ✓
- Per-card "View logs" snippet in server-health UI emits the right `kubectl logs deploy/<name>` (or `pod/<name>-0` for the StatefulSets postgres + minio) — Tier 1 of log access, no backend
- Grafana K8s dashboards rendering correctly (Prometheus + Postgres datasources auto-provisioned, all 3 dashboards visible in "Fall Detection" folder) ✓
- 5 K8s-only gotchas hit and resolved during this verification:
  - retrain Dockerfile didn't install `retrain/requirements.txt` → ModuleNotFoundError xgboost (gotcha #13)
  - kubelet auto-injected `ML_DASHBOARD_PORT=tcp://...` and `SERVER_HEALTH_PORT=tcp://...` URLs → `int(os.getenv(...))` crashed; fixed with `enableServiceLinks: false` (gotcha #12)
  - MLflow 3.x DNS rebinding protection rejected `Host: mlflow:5000` → added `--allowed-hosts "mlflow:*,..."` flag (gotcha #14)
  - ml-dashboard `/api/switch` got 401 because `INFERENCE_API_KEY` env var was unset → injected from `fall-detection-secrets/api-keys`. Caveat: works only when `apiKeys` value is a single key, not a comma-separated list (gotcha #15)
  - Grafana provisioning ConfigMap mounted at root of `/etc/grafana/provisioning/`, but Grafana scans subdirectories — fixed by projecting `datasources.yaml` / `dashboards.yaml` into `.../datasources/` and `.../dashboards/` via `items` in the volume spec (gotcha #16). Diagnostic confusion: on Windows with Compose Grafana also running, `localhost:3000` hits Compose (`::1`) and `127.0.0.1:3000` hits K8s — so K8s Grafana looked broken because we were unknowingly looking at Compose.

**Orchestration scripts (added 2026-04-29) — preferred over manual commands for local testing:**
```powershell
# from _6G_Integration_v2_mqtt/ as cwd
.\helm\fall-detection\build.ps1            # builds all 4 custom images with registry.example.com/ prefix
.\helm\fall-detection\install.ps1          # helm upgrade --install + --wait + shows pod status
.\helm\fall-detection\port-forward.ps1     # opens 7 tunnels in separate PowerShell windows
.\helm\fall-detection\test.ps1             # 8 in-cluster probes (self + cross-service + Grafana provisioning)
.\helm\fall-detection\teardown.ps1         # uninstall + delete namespace
```
install.ps1 refuses to run on non-`docker-desktop` kubectl context as a safety check.
For per-service code changes: `.\helm\fall-detection\build.ps1` then `kubectl rollout restart deploy/<name> -n mcs-fall-detection` — `helm upgrade` alone does NOT restart pods (gotcha #11).
For production deployment to a real cluster, use the manual workflow in `handover_docs_2/01_k8s.md` §6 (the scripts assume `pullPolicy: Never` + `registry.example.com` placeholder + Docker Desktop context).

**Remaining before handoff to FOCUS DevOps:**
- Confirm TODO values in values.yaml with FOCUS DevOps: `registry`, `ingress.host`, `namespaces.focus`, storage class
- Step 5: FHIR output (still blocked on FOCUS confirming FHIR server URL)

**See [project_k8s_local_testing.md](project_k8s_local_testing.md)** for the
diagnostic playbook + non-obvious gotchas hit so far.

## mock-focus dry-run chart (added 2026-04-28)

`helm/mock-focus/` — second Helm chart used ONLY for local two-namespace testing
on Docker Desktop K8s before handing the real chart to FOCUS DevOps.
Implements REFACTOR_DOCUS/todo.md Step 12.5.

**What's in it:** mock-fhir-server (reuses local_dev/mock_focus/fhir_server.py),
mock-influxdb (StatefulSet, influxdb:2.7), mock-patient-dashboard (FastAPI proxy +
HTML SPA at NodePort 30090 that consumes the real fall_dashboard SSE feed across
namespaces). Plus build/install/test/teardown PowerShell scripts and two
NetworkPolicy YAMLs in `extras/` for the deny→allow recovery test.

**What it validates:** DNS-based service discovery across namespaces, Helm
install/upgrade flow, NetworkPolicy enforcement, StatefulSet PVC binding, SSE
through cross-namespace pod→pod traffic. Failures here surface gaps before they
hit FOCUS's cluster.

**Run order:** `helm/mock-focus/build.ps1` → `install.ps1` → `test.ps1` →
manual SSE check at http://localhost:30090/ → optional NetworkPolicy test
(apply `extras/deny-all-cross-namespace.yaml`, see it break, apply
`extras/allow-mock-focus.yaml`, see it recover) → `teardown.ps1`.

**Never ships.** Mock images (`fd-mock-fhir`, `fd-mock-patient-dashboard`) are
built locally only. The real production deployment uses FOCUS's actual FHIR /
InfluxDB / Patient Dashboard.

## Infrastructure (Step 6c — added 2026-04-15)

`infrastructure/` folder — Docker Compose for all supporting services:

```
infrastructure/
  docker-compose.yml        ← postgres, mqtt, prometheus, grafana
  postgres/init.sql         ← creates fall_detection + mlflow databases
  mosquitto/mosquitto.conf  ← MQTT broker config
  prometheus/prometheus.yml ← scrapes host.docker.internal:8001/metrics
  grafana/
    provisioning/
      datasources/datasources.yaml   ← Prometheus + PostgreSQL auto-provisioned
      dashboards/dashboards.yaml     ← loads JSON files from dashboards/
    dashboards/
      ml_server_overview.json        ← request rate, latency, falls/hour, error rate
      model_performance.json         ← confidence distribution, drift alert, per-version breakdown
      fall_events_timeline.json      ← SQL: falls today, recent events table, confidence scatter
```

Start infra: `docker-compose -f infrastructure/docker-compose.yml up`
Grafana: http://localhost:3000 (admin/admin)
Prometheus: http://localhost:9090

Python services still run manually in terminals (not in Docker during dev).

To use Postgres instead of SQLite, set in .env:
`DATABASE_URL=postgresql+psycopg2://fall_user:fall_pass@localhost:5432/fall_detection`
Then run: `alembic upgrade head`

## What's implemented

- [x] Step 1: MQTT removed from inference_server — HTTP-only; returns fall result + observation_id in response
- [x] Step 2: mock_app/ separated (data fetcher + API caller + confirmation + MQTT publisher)
- [x] Step 3: fall_dashboard subscribes to fall/alert/# (confirmed alerts only)
- [x] Step 4: InfluxDB at 50Hz — HARDWARE_ACC_SAMPLE_RATE=50 in .env; resampler kept but skipped
- [x] Step 6a: Prometheus metrics (metrics_collector.py + wired in server.py)
- [x] Step 6b: Postgres shared schema — shared/db/ + Alembic migrations + BackgroundTask DB write in /predict
- [x] Step 6c: Grafana dashboards — infrastructure/docker-compose.yml + 3 auto-provisioned dashboards
- [x] Step 7: Model hot-swap (POST /model/switch + GET /model/list in server.py)
- [x] Step 9 (partial): Helm chart written — all 10 services templated, smoke-tested on Docker Desktop K8s
- [x] Step 11: MLflow retraining pipeline — retrain/ folder with data_pipeline, retrain.py, seed_test_data.py
- [x] Step 11.5.2: ml_dashboard MVP — admin UI for retrain + hot-swap (port 8004). **Now in Helm chart.**
- [x] Step 11.5.3: server_health MVP — aggregate service health probes (port 8006). **Now in Helm chart.**
- [x] Step 12.5: Local two-namespace dry-run — PASSED (2026-04-29)
- [x] Step 7.3: MinIO artifact store — MinIO in our namespace, confirmed working locally (2026-04-17).
      retrain.py → .pkl stored in MinIO → switch_model.ps1 -Stage Production downloads from MinIO.

## Mock FHIR server (mock_focus/ — added 2026-04-16)

`mock_focus/fhir_server.py` — FastAPI, port 8003. Simulates FOCUS's FHIR server for local dev.
Endpoints: `GET /fhir/Patient`, `GET /fhir/Patient/{id}`, `GET /fhir/Observation?patient={id}`
Added to docker-compose as `mock_focus_fhir` service (labeled: NOT part of our production namespace).
**Replace with real FOCUS FHIR_SERVER_URL in K8s — this mock never ships.**

## Patient Dashboard naming clarification (2026-04-16)

- **Patient Dashboard** = FOCUS DevOps's Flutter web app (FOCUS namespace; NOT Isa — corrected 2026-04-29). Combines:
  - Demographics from FHIR server (`mock_focus` locally, real FOCUS FHIR in prod)
  - Biosignals from InfluxDB (FOCUS side)
  - Fall panel from our fall_dashboard API (/api/falls, /api/stream SSE)
- **fall_dashboard** = our backend service (:8002) — feeds fall panel only, not a standalone UI

## Still open

- Step 5: FHIR output format (blocked on FOCUS confirming FHIR server exists); mock server at mock_focus/ unblocks local dev
- Step 8: Patient Dashboard fall panel (**FOCUS DevOps** builds UI in their Flutter web app — NOT Isa). API docs at handover_docs/FOCUS_patient_dashboard_integration.md (Flutter/Dart code samples).
- Step 9: Helm chart templates complete. Still blocked on FOCUS DevOps answers: registry URL, real namespace names, ingress host, whether NetworkPolicy is enforced.
- Step 10: End-to-end test on real cluster (all backend complete — ready to run once FOCUS cluster access available)
- Step 11.5.4/11.5.5: Auth gate + audit log for ml_dashboard and server_health (not safe to expose without auth)

## InfluxDB — two instances, completely different roles (confirmed 2026-04-27)

| Instance | Role | In production? |
|----------|------|---------------|
| **FOCUS-hosted InfluxDB** | Biosignal display in Patient Dashboard (HR etc.) Written by real mobile app. Read by the FOCUS Flutter Patient Dashboard only. | Yes — in FOCUS namespace |
| **Our cloud InfluxDB** (`ecosystem-influxdb.smarko-health.de`) | Used ONLY by `local_dev/mock_app/` as fake BLE wearable input. mock_app cannot read from real SmarKo hardware so it reads this instead. | No — disappears in production |

**Our inference pipeline never reads from any InfluxDB at runtime.** In production: real mobile app reads BLE → POSTs directly to `/predict`. InfluxDB is not in the inference path.

## Database decisions (2026-04-15)

- **One Postgres instance, two logical databases:**
  - `fall_detection`: inference_log, feature_snapshot, fall_history, participant_session
  - `mlflow`: MLflow internal tracking tables (kept separate to avoid migration conflicts)
- Both `inference_server` and `fall_dashboard` write to the same Postgres instance (different tables).
- Cross-reference: `observation_id` (UUID string) is the FK between inference_log and fall_history (not integer FK).
- Local dev: `DATABASE_URL=sqlite:///./caregiver.db` (default in .env — no Docker needed)
- Production: `DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/fall_detection`
- InfluxDB: currently using our own cloud instance for dev/testing. Eventually FOCUS decides if InfluxDB lives in their namespace or ours.
- MinIO: **in our namespace, confirmed working locally (2026-04-17)**. MLflow artifact store = `s3://mlflow-artifacts/` on MinIO (:9000). Console at :9002 (port 9001 taken by MQTT WebSocket). Bucket name: `mlflow-artifacts`. Requires `boto3` + four env vars in `.env` (currently uncommented and active). `MLFLOW_ARTIFACT_ROOT` is NOT auto-applied to existing MLflow experiments when using SQLite tracking URI — `retrain.py` explicitly passes `artifact_location` to `client.create_experiment()`. Use `retrain/_delete_exp.py` to permanently purge old experiments if artifact location needs resetting.

## FOCUS cluster confirmed info (2026-04-27)

- Ingress controller: **Traefik** (not nginx). `ingress.yaml` updated: removed nginx annotations, set `spec.ingressClassName: traefik`. SSE works without extra annotations in Traefik.
- Total RAM: **32 GB**. Resource limits added to `values.yaml` and all 8 deployment/statefulset templates. Total limits ~8 CPU / ~10 Gi.
- StorageClass: left as `""` (blank = use cluster default). Self-resolves if FOCUS cluster has a default StorageClass.
- Still need from FOCUS DevOps: container registry URL, real namespace names, real ingress host, whether NetworkPolicy is enforced.

## Two Kubernetes namespaces

- **FOCUS namespace**: SmarKo mobile app, InfluxDB, FHIR server, FOCUS dashboard (patient info)
- **Our namespace**: inference_server, fall_dashboard, MQTT broker, Postgres, MLflow, Prometheus, Grafana, MinIO
- Cross-namespace traffic: read-only HTTP from our namespace into FOCUS (InfluxDB reads, FHIR reads). Mobile app POSTs to our inference_server.
- See `REFACTOR_DOCUS/deployment_architecture.md` for full breakdown and data source table.

## Key design decisions

- Inference server is HTTP-only. MQTT_FALL_TOPIC and paho-mqtt removed entirely.
- Only 2 MQTT clients: mock_app (publisher) and fall_dashboard (subscriber).
- Patient confirmation happens in mock_app via PatientConfirmationServer (:8005 browser popup), not in fall_dashboard. Dead auto-confirm timer code removed from web.py (2026-04-27).
- fall_dashboard reads `patient_confirmed` + `observation_id` + `needs_help` from MQTT alert payload.
- FallEventBroker.on_fall is set in client.py before broker.start() to avoid race condition.
- BackgroundTask pattern: inference_log write happens AFTER HTTP response is sent — never adds latency to /predict.
- MLFLOW_TRACKING_URI=./mlruns (local file store default); change to http://mlflow:5000 for production.

## .env key settings

```
MQTT_BROKER_HOST=127.0.0.1    # must be 127.0.0.1, NOT localhost — Windows resolves localhost to ::1 (IPv6) and port-forward only binds IPv4
MQTT_ALERT_TOPIC=fall/alert
MOCK_PATIENT_RESPONSE_TIMEOUT=10
MOCK_PATIENT_SERVER_PORT=8005
DATABASE_URL=postgresql+psycopg2://fall_user:fall_pass@localhost:5432/fall_detection
# DATABASE_URL=sqlite:///./caregiver.db   ← uncomment to switch back to SQLite
MLFLOW_TRACKING_URI=sqlite:///./mlruns.db
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
MLFLOW_ARTIFACT_ROOT=s3://mlflow-artifacts/
HARDWARE_ACC_SAMPLE_RATE=50
```

## Documentation files created (in REFACTOR_DOCUS/)

- `isa_integration_guide.md` — API reference for Isa: /api/patients, /api/falls, /api/stream SSE, MQTT WebSocket on port 9001 for React Native
- `helm_guide.md` — 15-step Helm deployment guide: Dockerfiles, chart structure, two-namespace setup, Postgres StatefulSet, Alembic as K8s Job, cross-namespace access, open questions for FOCUS DevOps
- `mlops_retraining_cycle.md` — full MLOps cycle guide: tool map (InfluxDB/Postgres/Grafana/MLflow/inference_server roles), label logic, seeding, training CLI, when to switch models (recall > AUC > F1 > precision for safety-critical), registry-based and file-based hot-swap commands, MLflow Registry stages

## Gotchas discovered during testing

- `python-dotenv` does NOT strip inline `#` comments — `VAR=   # comment` reads as `"   # comment"` (truthy). Move comments to line above.
- InfluxDB query must include `r["_measurement"] == "SMART_DATA"` — without it, zero records returned
- `MAC_IDS` uses positional mapping to `PATIENT_IDS` (comma-separated lists) — previous `key:value` format broke on MAC addresses containing `:`
- `asyncio.get_event_loop()` fails in non-main threads — capture `asyncio.get_running_loop()` in the async startup hook instead
- Caregiver dashboard Fall History tab had no auto-refresh — only patients tab had `setInterval`. Fixed: both tabs now refresh every 15s in `dashboard/app.js`
- SQLite "database is locked" can occur if an external DB editor holds the file. Fix: `timeout=30` in connect_args, or delete `caregiver.db` and restart.
- Patient tab was showing MAC address instead of patient name — root cause: patient card used `mac_id || patient_id` (MAC wins when MAC_IDS is set). Fixed (2026-04-17): patient card now always shows `patient_id`; MAC shown as a secondary badge. Fall History "Device (MAC)" column still uses `mac_id || patient_id` (correct — that column is for the device ID).
- Old `caregiver.db` (pre Step 6b) has incompatible schema (`fall_detection` column instead of `fall_detected`, missing `observation_id`). Delete it on first run after upgrade — new schema created automatically by `init_db()`.
- `DATABASE_URL` in `.env` was SQLite by default — switched to Postgres (2026-04-16). Services must be restarted after changing `.env` to pick up the new value.
- `psycopg2-binary` must be installed separately — not in requirements.txt by default. Run `pip install psycopg2-binary` before `alembic upgrade head`.
- `/api/falls` is a fall_dashboard endpoint (:8002), not inference_server (:8001) — easy to hit the wrong port.
- Interactive Postgres session: `docker exec -it fall_postgres psql -U fall_user -d fall_detection`. Column is `detection_time`, not `created_at`.
- MinIO console originally put on port 9001 — conflicts with MQTT WebSocket listener. Moved to 9002.
- MLflow `MLFLOW_ARTIFACT_ROOT` env var is ignored when using SQLite tracking URI (client-side mode). Must pass `artifact_location` explicitly to `client.create_experiment()`. See `retrain/_delete_exp.py` to purge old local-path experiments.
- MLflow soft-deletes experiments (`delete_experiment`) — `set_experiment()` on a soft-deleted name raises "Cannot set a deleted experiment". Must permanently delete via direct SQL: `DELETE FROM experiments / runs WHERE experiment_id = X`.
- FHIR push via BackgroundTask decided NOT to implement (deferred) — see `REFACTOR_DOCUS/FHIR_facade.md`. `FHIR_SERVER_URL` stays blank until FOCUS confirms they need it.
