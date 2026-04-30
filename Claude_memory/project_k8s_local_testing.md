---
name: K8s local testing — lessons from the helm install dry-run
description: Non-obvious gotchas hit while running helm/fall-detection on Docker Desktop K8s. Preserves diagnostic shortcuts so future debugging is faster.
type: project
---

## Setup we test against

- **Cluster:** Docker Desktop K8s on Windows (PowerShell shell)
- **Charts:** `helm/fall-detection/` (production) and `helm/mock-focus/` (dry-run only)
- **Image source:** local Docker daemon (no registry) → `pullPolicy: Never` in `values.yaml`

The whole "is the chart correct" question is being validated locally per todo.md Step 12.5 before the chart goes to FOCUS DevOps.

## Gotchas that cost us time (won't be obvious from reading files)

### 1. Docker Desktop K8s containerd cache ≠ Docker daemon cache

Symptom: `kubectl get pods` shows `ErrImageNeverPull` even though `docker images` lists the image.
Reason: Docker Desktop K8s uses containerd; the daemon's `docker images` list is a different store, and they don't always sync.
Fix: rebuild the image (`docker build -t ... .`) and `kubectl delete pod -l app=<name>` to force re-creation. New pod re-checks containerd's cache, usually picks it up.

### 2. MLflow OOM-killed with 1Gi memory limit

Symptom: `mlflow` pod cycles `Running 0/1 → OOMKilled → Running 0/1` forever; logs show "child process died" repeatedly.
Reason: gunicorn forks 4 workers; each loads MLflow + sqlalchemy + boto3 + psycopg2 → ~300 MiB per worker → ~1.2 GiB total → exceeds 1Gi limit.
Fix (applied 2026-04-29): bumped `mlflow.limits.memory` from 1Gi to 2Gi in `helm/fall-detection/values.yaml`. Watch for similar issues on other Python services as features grow.

### 3. `Temporary failure in name resolution` is usually a startup-timing issue, not actual DNS

If a pod logs DNS errors but the target Service definitely exists, kill the pod (`kubectl delete pod -n <ns> -l app=<name>`). The replacement spawns after CoreDNS is fully ready and resolves cleanly. Don't immediately blame the connection string.

### 4. `helm install --create-namespace` + a chart's own namespace.yaml template = duplicate namespaces

### 5. `ErrImageNeverPull` when template prefixes registry in image name

Symptom: pods show `ErrImageNeverPull` even though `docker images` shows the image.
Reason: the deployment template builds `image: {{ .Values.registry }}/repo:tag`, resolving to e.g. `registry.example.com/inference-server:latest`. Docker Desktop K8s with `pullPolicy: Never` looks for that exact name — which doesn't exist locally (local name is just `inference-server:latest`).
Fix: retag the image to match: `docker tag inference-server:latest registry.example.com/inference-server:latest`. Applies to every image that goes through the `{{ .Values.registry }}/` prefix.
Proper long-term fix: make registry optional in templates — `{{ if .Values.registry }}{{ .Values.registry }}/{{ end }}repo:tag` — and leave `registry: ""` in values.yaml for local dev.

### 6. migrate-job `ImagePullBackOff` — missing explicit `imagePullPolicy`

Symptom: alembic-migrate job hits `ImagePullBackOff` while deployments with the same image work fine.
Reason: K8s defaults `imagePullPolicy` to `Always` when the image tag is `latest`. The deployment templates explicitly set `imagePullPolicy: Never` but the Job template didn't — so the job tried to pull from the registry and failed.
Fix (applied 2026-04-29): added `imagePullPolicy: {{ .Values.images.pullPolicy }}` to `templates/migrate-job.yaml`.

### 7. Failed helm hook job blocks future `helm upgrade` with "field is immutable"

Symptom: `helm upgrade` fails with `Job ... spec.template: Invalid value: ... field is immutable`.
Reason: a Job's pod template spec cannot be modified in place. The previous hook job was still in the cluster (it only gets deleted on success with `hook-delete-policy: hook-succeeded`). Helm tried to update it and K8s rejected it.
Fix (applied 2026-04-29): changed to `hook-delete-policy: before-hook-creation,hook-succeeded` — Helm deletes the old job before creating the new one every time, regardless of previous success/failure. Manual recovery: `kubectl delete job alembic-migrate -n mcs-fall-detection`.

### 8. Alembic race condition — `init_db()` runs before the alembic hook

Symptom: alembic-migrate job fails with `DuplicateTable: relation "inference_log" already exists`.
Reason: `fall_dashboard/web.py` calls `init_db()` on startup → `Base.metadata.create_all()` creates all tables. The alembic-migrate job runs as a `post-install` hook, i.e. AFTER fall-dashboard is already Running. So alembic finds tables but no `alembic_version` entry.
Fix (applied 2026-04-29): added an `initContainer` to the fall-dashboard Deployment that runs `alembic upgrade head` before the main container starts. This ensures alembic always owns schema creation; `init_db()` becomes a no-op since tables already exist.
Recovery for existing desync: `kubectl run alembic-stamp ... --command -- alembic stamp head` to write the current head version into `alembic_version` without re-running DDL.

### 10. paho-mqtt connects to `::1` (IPv6) on Windows — port-forward only listens on `127.0.0.1` (IPv4)

Symptom: mock_app logs "MQTT publisher connected" and "ALERT published" but `Handling connection for 1883` never appears in the port-forward terminal, and fall_history has 0 rows.
Reason: On Windows, `localhost` resolves to `::1` (IPv6) first. `kubectl port-forward` only binds `127.0.0.1` (IPv4) for some services. paho-mqtt's `connect()` doesn't block or verify success — it just queues the connection attempt. `publish()` with QoS=0 also logs success even when the TCP connection was never established.
Fix: set `MQTT_BROKER_HOST=127.0.0.1` (not `localhost`) when running mock_app locally against a K8s port-forward.
K8s is unaffected: pods inside the cluster connect to `mqtt-broker:1883` via internal DNS — no port-forward, no IPv4/IPv6 issue.

### 16. Grafana provisioning silently loaded zero providers — ConfigMap mounted at wrong directory level

Symptom: K8s Grafana UI shows empty "Browse Dashboards" page despite ConfigMap holding all the JSON files. No errors in pod status, but `helm install` succeeded.
Reason: Grafana scans `/etc/grafana/provisioning/datasources/*.yaml` and `/etc/grafana/provisioning/dashboards/*.yaml` (note the SUBDIRECTORIES). Our ConfigMap had `datasources.yaml` and `dashboards.yaml` as keys; mounting the whole ConfigMap at `/etc/grafana/provisioning` placed those files at the root, not in subdirs. Grafana logs the error but continues startup, so provisioning silently produces 0 providers.
Logs to spot it: `level=error msg="can't read datasource provisioning files from directory" path=/etc/grafana/provisioning/datasources error="...no such file or directory"` followed by `starting to provision dashboards` → `finished to provision dashboards` with no per-provider lines in between.
Fix (applied 2026-04-29): project each key into its expected subdir using `items` in the volume spec — one ConfigMap, two scoped volumeMounts.
Diagnostic confusion vector: on Windows with Compose Grafana also running, `localhost:3000` resolves to `::1` and bypasses the `kubectl port-forward` (IPv4-only), hitting the Compose Grafana which DOES have dashboards. `127.0.0.1:3000` is the only way to verify the K8s Grafana actually works. Easy to think K8s Grafana is fine when you're really looking at Compose.

### 15. ml-dashboard hot-swap step 5 returned `X-API-Key header required` — INFERENCE_API_KEY missing in pod env

Symptom: ml-dashboard UI's hot-swap (step 5: POST to inference-server `/model/switch`) returned
`Failed: {"detail":"{\"detail\":\"X-API-Key header required.\"}"}`
Reason: ml_dashboard/web.py only sends `X-API-Key` header when `os.getenv("INFERENCE_API_KEY")` is non-empty. The K8s deployment never set that env var, so the header was omitted. inference-server rejected with 401.
Fix (applied 2026-04-29): added `INFERENCE_API_KEY` env var to the ml-dashboard deployment, reading the same `fall-detection-secrets/api-keys` value as inference-server.
Caveat: `apiKeys` is a comma-separated list on the server side (e.g. `"key1,key2"`). ml-dashboard sends the entire value verbatim as a single header. Works only while there's exactly one key configured. With multiple keys, ml-dashboard needs its own dedicated single-key secret entry.

### 14. MLflow 3.x DNS rebinding protection blocks K8s service-DNS Host headers

Symptom: retrain.py inside ml-dashboard pod fails on `client.get_experiment_by_name(...)` with
`MlflowException: API request to endpoint /api/2.0/mlflow/experiments/get-by-name failed with error code 403 != 200. Response body: 'Invalid Host header - possible DNS rebinding attack detected'`
Reason: MLflow 3.x added DNS rebinding protection (PR #22095). The middleware compares the request's `Host` header against a hardcoded allowlist (default: localhost variants). Inside K8s, requests to `http://mlflow:5000` carry `Host: mlflow:5000`, which the allowlist rejects.
Fix (applied 2026-04-29): pass `--allowed-hosts "mlflow:*,mlflow.<namespace>.svc.cluster.local:*,localhost:*,127.0.0.1:*"` to `mlflow server` in the deployment. Patterns use fnmatch-style wildcards.
Alternative: env var `MLFLOW_SERVER_ALLOWED_HOSTS` with the same comma-separated patterns.
Local dev unaffected because retrain.py talks to `localhost:5000` via port-forward — already whitelisted by default.

### 13. ml-dashboard Dockerfile didn't install `retrain/requirements.txt` — subprocess `python -m retrain.retrain` failed with `ModuleNotFoundError: No module named 'xgboost'`

Symptom: clicking "Retrain" in the ml-dashboard UI returned a traceback ending in `ModuleNotFoundError: No module named 'xgboost'`.
Reason: ml_dashboard's Dockerfile only ran `pip install -r ml_dashboard/requirements.txt`, which has fastapi/uvicorn/mlflow/httpx but NOT the ML libs. ml_dashboard launches `python -m retrain.retrain` as a subprocess — that subprocess needs xgboost, sklearn, numpy, pandas, sqlalchemy, boto3 etc. (everything in `retrain/requirements.txt`).
Fix (applied 2026-04-29): updated `ml_dashboard/Dockerfile` to install both `ml_dashboard/requirements.txt` AND `retrain/requirements.txt`. Image is ~500MB bigger but the retrain subprocess works.
Lesson: when one service spawns another as a subprocess, both `requirements.txt` files must be installed in the same image.

### 12. K8s service-link env vars override `<SERVICE>_PORT` with a URL — breaks `int(os.getenv(...))`

Symptom: ml-dashboard / server-health pods CrashLoopBackOff with
`ValueError: invalid literal for int() with base 10: 'tcp://10.103.188.161:8004'`
Reason: For every Service in the namespace, kubelet auto-injects env vars into every pod
(default `enableServiceLinks: true`). For a Service named `ml-dashboard`, you get:
- `ML_DASHBOARD_SERVICE_HOST=10.103.188.161`
- `ML_DASHBOARD_SERVICE_PORT=8004`
- `ML_DASHBOARD_PORT=tcp://10.103.188.161:8004`  ← URL, NOT integer port
Our Python `main.py` reads `os.getenv("ML_DASHBOARD_PORT", "8004")` expecting an integer port.
inference-server / fall-dashboard avoided this by using non-colliding names (`SERVER_PORT`, `CAREGIVER_PORT`).
Fix (applied 2026-04-29): added `enableServiceLinks: false` to ml-dashboard and server-health pod specs.
We use DNS-based discovery (`INFERENCE_SERVER_URL` etc. from configmap) instead, so we don't need the auto-injected vars.
Lesson: when adding new services to the chart, either (a) avoid env-var names that match `<SERVICE_NAME>_PORT`, or (b) set `enableServiceLinks: false` on the pod spec.

### 11. `helm upgrade` does NOT restart pods when image tag stays `latest` + `pullPolicy: Never`

Symptom: `helm upgrade` succeeds but the pod is still running the old code (e.g. click-to-acknowledge fix not visible in browser after helm upgrade).
Reason: K8s only restarts a pod when the pod spec changes. With `pullPolicy: Never` and a fixed `latest` tag, there is no spec change — so existing pods keep running the old image.
Fix: rebuild the image, then force a rollout restart:
```powershell
docker build -f <service>/Dockerfile -t registry.example.com/<service>:latest .
kubectl rollout restart deploy/<service-name> -n mcs-fall-detection
```
`helm upgrade` is still needed when values or chart templates changed; the `rollout restart` is the additional step needed to swap the image. Run both after code changes.

### 9. `wget` not available in `python:3.11-slim` images

Symptom: `kubectl exec ... -- wget ...` or test scripts using `wget` fail with `wget: not found`.
Reason: `python:3.11-slim` (Debian-based) does not include wget. Applies to: inference-server, fall-dashboard, mock-fhir, mock-patient-dashboard.
Fix: use `curl` for Debian-based official images (influxdb, postgres), or `python3 -c "import urllib.request; print(urllib.request.urlopen('http://...').read().decode())"` for python:3.11-slim containers. Applied in `helm/mock-focus/test.ps1` and `helm/fall-detection/README.md`.

Confirmed 2026-04-29. The chart had `templates/namespace.yaml` creating `{{ .Values.namespaces.ours }}` while values.yaml said `fall-detection` and the operator passed `--namespace mcs-fall-detection --create-namespace`. Result: two namespaces (one orphan).
Fix already applied: values.yaml now says `mcs-fall-detection`. Future cleanup: drop the chart's namespace.yaml template entirely (covered by `--create-namespace`) — tracked but not done.

## Diagnostic command cheatsheet

```powershell
# Pod-level state
kubectl get pods -n mcs-fall-detection
kubectl get pods -n mcs-fall-detection -w                       # live watch

# Why is X pod not ready?
kubectl describe pod <name> -n mcs-fall-detection
kubectl logs <name> -n mcs-fall-detection --tail=200
kubectl logs <name> -n mcs-fall-detection --previous            # if it crashed and restarted

# Filtered logs (PowerShell — Select-String, NOT findstr)
kubectl logs deploy/mlflow -n mcs-fall-detection | Select-String -Pattern "Error|Exception" -Context 0,5

# What images are referenced?
kubectl describe deploy <name> -n mcs-fall-detection | Select-String "Image:"

# What's actually running where?
kubectl get all -n mcs-fall-detection
kubectl get svc -n mcs-fall-detection                            # for service-name lookups
```

## Apply chart changes

```powershell
helm upgrade mcs-fall-detection .\helm\fall-detection -n mcs-fall-detection --wait --timeout 5m
```

`--wait` blocks until all pods are Ready or the timeout expires. If it times out, the resources are still in-cluster — diagnose, fix values.yaml or templates, then `helm upgrade` again. No need to uninstall first.

## Status as of 2026-04-29

- [x] MLflow OOMKilled — fixed via 2Gi memory bump
- [x] Two-namespace bug — values.yaml fixed
- [x] ErrImageNeverPull on inference-server / fall-dashboard — fixed: retag images with registry prefix (gotcha #5)
- [x] alembic-migrate ImagePullBackOff — fixed: added explicit `imagePullPolicy` to Job template (gotcha #6)
- [x] alembic-migrate "field is immutable" on upgrade — fixed: `before-hook-creation,hook-succeeded` delete policy (gotcha #7)
- [x] alembic DuplicateTable race condition — fixed: initContainer in fall-dashboard runs alembic before app starts (gotcha #8)
- [x] All 10 pods Running in `mcs-fall-detection` (8 original + ml-dashboard + server-health added 2026-04-29)
- [x] mock-focus chart installed; cross-namespace tests 4/4 PASS
- [x] Step 5.4: manual SSE end-to-end test — COMPLETE (2026-04-29). MQTT working, alert shown on dashboard, fall_history written to Postgres, fall history tab updated. click-to-acknowledge works.
- [x] mock-patient-dashboard click-to-acknowledge: clicking a red/yellow card clears flag and resets card to white. Required `kubectl rollout restart` after image rebuild (gotcha #11 — helm upgrade alone not enough).
- [-] Step 5.5: NetworkPolicy deny→allow test — SKIPPED (Docker Desktop CNI won't enforce it anyway; needs kind+Calico)
- [ ] Eventually: drop `templates/namespace.yaml` (covered by `--create-namespace`)

## Smoke test (Step 12.5) result: PASSED (2026-04-29)

All pass criteria met:
- install.ps1: all pods Running ✓
- test.ps1: 4/4 cross-namespace tests PASS ✓
- Step 5.4 manual SSE test: alert appears on dashboard, Postgres written ✓
- Step 5.5 NetworkPolicy: skipped (Docker Desktop CNI does not enforce — informational only)
