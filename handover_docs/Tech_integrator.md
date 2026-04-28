# Tech Integrator Handover — FOCUS DevOps

**Audience:** FOCUS DevOps engineers responsible for deploying the fall-detection workload alongside the existing FOCUS namespace (FHIR server, InfluxDB, Patient Dashboard).
**Repo / branch:** `_6G_Integration_v2_mqtt/` on branch `6G-integration_with_MQTT`.
**Helm chart location:** `_6G_Integration_v2_mqtt/helm/fall-detection/`.
**Purpose of this document:** Give you everything you need to plug our Helm chart into your cluster — what we deliver, what cross-namespace traffic we expect, what answers we still need from you, and how to install / upgrade / verify.

If you only read one section, read **section 3** (open questions we need answered before `helm install`) and **section 7** (install).

---

## 1. What We're Delivering

One Helm chart that deploys our fall-detection backend into a single Kubernetes namespace inside your cluster. The chart owns:

- 2 stateless services: `inference-server`, `fall-dashboard`
- 1 message broker: `mqtt-broker` (eclipse-mosquitto)
- 2 stateful services: `postgres` (StatefulSet, 10Gi PVC), `minio` (StatefulSet, 20Gi PVC)
- 3 observability services: `mlflow`, `prometheus`, `grafana`
- 1 Ingress (Traefik) exposing `/predict` and `/api`
- 1 Alembic migration Job (Helm hook: post-install / post-upgrade)
- 1 MinIO bucket-creation Job (Helm hook: post-install)

What we are **not** asking you to install: anything in your existing FOCUS namespace. We don't touch your FHIR server, InfluxDB, or Patient Dashboard.

We deliver the chart + the two Docker images. You run `helm install`.

---

## 2. The Two-Namespace Picture

The mobile app and the SmarKo wearable are external — they live on the patient's phone and on the patient's body, not in any namespace.

```
[SmarKo wearable]   (BLE)
       │
       ▼
[Mobile App]  ←  patient's phone
   │       │
   │       │ writes biosignals (HR etc.)
   │       ▼
   │   ┌────────────────────────┐         ┌──────────────────────────────────────┐
   │   │  FOCUS NAMESPACE       │         │  OUR NAMESPACE                       │
   │   │  (yours — unchanged)   │         │  (fall-detection — Helm chart)      │
   │   │                        │         │                                      │
   │   │  InfluxDB  ←───────────┤         │  inference-server  :8001            │
   │   │  FHIR Server           │ ←───────┤  (optional FHIR push, only if       │
   │   │                        │         │   FHIR_SERVER_URL configured)        │
   │   │  Patient Dashboard ←───┼─────────┤  fall-dashboard    :8002            │
   │   │  (Isa's UI)            │  HTTP/  │                                      │
   │   │                        │   SSE   │  mqtt-broker       :1883 / :9001 ws │
   │   │                        │         │  postgres          :5432            │
   │   └────────────────────────┘         │  mlflow            :5000            │
   │                                       │  prometheus        :9090            │
   │   HTTP POST /predict                  │  grafana           :3000            │
   └─────────────────────────────────────► │  minio             :9000            │
                                            └──────────────────────────────────────┘
   MQTT PUBLISH fall/alert/<patient_id>  (mobile app → broker, ws://9001)
```

### Cross-namespace traffic — exhaustive list

These are the only paths that cross the namespace boundary. All inbound to your FOCUS namespace are read-only except optional FHIR push.

| From | To | Protocol | Direction | What | Required? |
|------|-----|----------|-----------|------|-----------|
| Mobile App (external) | `inference-server` (ours) | HTTPS via Ingress | Inbound to ours | `POST /predict` | Yes — primary path |
| Mobile App (external) | MQTT broker (ours) | MQTT over WebSocket (9001) | Inbound to ours | `PUBLISH fall/alert/<patient_id>` | Yes |
| Mobile App (external) | InfluxDB (FOCUS) | InfluxDB HTTP API | Inbound to FOCUS | Biosignal writes | Existing — not our concern |
| Patient Dashboard (FOCUS) | `fall-dashboard` (ours) | HTTP / SSE | Cross-namespace | `GET /api/patients`, `GET /api/falls`, `GET /api/stream` | Yes |
| Patient Dashboard (FOCUS) | FHIR Server (FOCUS) | HTTPS | Inside FOCUS | Demographics | Existing |
| `inference-server` (ours) | FHIR Server (FOCUS) | HTTPS | Outbound from ours | Optional FHIR Observation push | Optional — only if `FHIR_SERVER_URL` is set in our values.yaml |

### What we DON'T do

- We don't read your InfluxDB. The mobile app reads ACC from BLE and POSTs it directly to `/predict` — InfluxDB is not in our inference path. (For local dev we use a separate cloud InfluxDB that has nothing to do with your namespace.)
- We don't write to InfluxDB. The mobile app writes biosignals to your InfluxDB on its own.
- We don't touch your FHIR server unless `FHIR_SERVER_URL` is explicitly set, and even then the only operation is a POST of one Observation per detected fall.

---

## 3. Open Questions We Need You to Answer Before `helm install`

These are tracked in `_6G_Integration_v2_mqtt/REFACTOR_DOCUS/todo.md` (Step 9.12 / 9.14 / 14.1–14.3). Until they're answered, the chart is written but cannot be deployed.

| # | Question | Field that depends on it |
|---|----------|--------------------------|
| 1 | **Container registry URL** — where should we push our Docker images? | `values.yaml → registry` (currently `registry.example.com`) |
| 2 | **Namespace names** — is `fall-detection` acceptable for our namespace? What is your FOCUS namespace called? | `values.yaml → namespaces.ours`, `namespaces.focus` |
| 3 | **Ingress host / domain** — what FQDN should our services be reachable at? | `values.yaml → ingress.host` (currently `fall-detection.example.com`) |
| 4 | **Default StorageClass** — what's the cluster's default PVC class? | `values.yaml → postgres.storageClass`, `minio.storageClass` |
| 5 | **NetworkPolicy** — does the cluster enforce default-deny cross-namespace traffic? | If yes, we need to add a NetworkPolicy allowing your FOCUS namespace → our namespace for the Patient Dashboard → fall-dashboard call. |
| 6 | **Patient Dashboard host model** — does the dashboard run server-side (a pod in FOCUS namespace) or as a browser SPA? | Determines whether Isa uses the cluster-internal DNS name or the public Ingress URL. Doesn't change our chart, but Isa needs the answer. |
| 7 | **FHIR server URL & whether FHIR push is required** | `values.yaml → inferenceServer.fhirServerUrl`. Empty = no push. |
| 8 | **Patient ID format** | Doesn't change the chart; affects the data Isa puts in FHIR + `/predict`. |
| 9 | **MQTT broker authentication policy** | Currently anonymous. Before production we need to decide on user/pass or mTLS. |

Once 1–5 are answered we can deploy. 6–9 can come slightly later but must be in place before go-live.

---

## 4. Resource Footprint

Cluster total confirmed: **32 GB RAM**. Our chart's totals are well within that.

| Component | CPU request | CPU limit | RAM request | RAM limit |
|-----------|-------------|-----------|-------------|-----------|
| inference-server | 500m | 2 | 512Mi | 2Gi |
| fall-dashboard | 250m | 1 | 256Mi | 512Mi |
| mqtt-broker | 100m | 500m | 128Mi | 256Mi |
| postgres | 250m | 1 | 512Mi | 2Gi |
| mlflow | 250m | 1 | 256Mi | 1Gi |
| minio | 250m | 1 | 512Mi | 2Gi |
| prometheus | 250m | 1 | 512Mi | 2Gi |
| grafana | 100m | 500m | 128Mi | 512Mi |
| **Total** | **~2 CPU req / ~8 CPU lim** | | **~3 Gi req / ~10 Gi lim** | |

PVCs: `postgres-data` 10Gi, `minio-data` 20Gi.

If your cluster has tighter quotas, all of these are configurable via `values.yaml → resources`. Reasonable to halve the limits if you need headroom — none of them are CPU-bound under normal load.

---

## 5. The Helm Chart Structure

```
helm/fall-detection/
├── Chart.yaml
├── values.yaml                    ← single file you change per environment
├── files/                         ← postgres init.sql, mosquitto.conf, etc.
└── templates/
    ├── namespace.yaml
    ├── secrets.yaml               ← postgres pwd, minio pwd, api key, grafana pwd
    ├── configmap.yaml             ← shared env (DATABASE_URL, MQTT, MLflow, etc.)
    ├── migrate-job.yaml           ← Alembic Job (Helm hook: post-install,post-upgrade)
    ├── inference-server/  (deployment + service)
    ├── fall-dashboard/    (deployment + service + ingress)
    ├── mqtt-broker/       (deployment + service + configmap)
    ├── postgres/          (statefulset + service)
    ├── minio/             (statefulset + service + bucket-creation Job)
    ├── mlflow/            (deployment + service)
    ├── prometheus/        (deployment + service + configmap)
    └── grafana/           (deployment + service + dashboards configmap)
```

### Key chart decisions you should know about

- **Postgres is a StatefulSet, not a Deployment.** PVC via `volumeClaimTemplates`. One Postgres instance hosts two logical databases: `fall_detection` (our app data) and `mlflow` (MLflow internals). Both created by `files/init.sql`.
- **MinIO is a StatefulSet** with a separate PVC. MLflow's artifact store points at `s3://mlflow-artifacts/`. The bucket itself is created by a one-shot Job on `post-install`.
- **Alembic runs as a Helm-hook Job.** Hook: `post-install,post-upgrade`. Weight `-5`. `hook-delete-policy: hook-succeeded`. You never need to run migrations manually.
- **Ingress uses Traefik.** `spec.ingressClassName: traefik`, no nginx annotations. Traefik handles SSE natively (no `proxy-buffering` annotation needed). If you use a different controller, this is one file to swap (`fall-dashboard/ingress.yaml`).
- **`replicas: 1` for inference-server.** This is **not** a defensive default — Prometheus counters and Python threading state are per-process. Do not scale this up.
- **MQTT broker exposes TCP 1883 + WebSocket 9001.** Mobile apps (React Native) require WebSocket; backend services use TCP. Both ports are needed.

### `values.yaml` — what changes per environment

The chart is fully driven by `values.yaml`. Below are the placeholders you need to fill in. Everything else has sane defaults.

```yaml
namespaces:
  ours: fall-detection        # confirm acceptable to you
  focus: focus-ns             # your FOCUS namespace name

registry: registry.example.com   # your registry URL

images:
  pullPolicy: Always              # change from "Never" (which is for local Docker Desktop)
  inferenceServer:
    repository: inference-server
    tag: v1.0.0                   # bump per release
  fallDashboard:
    repository: fall-dashboard
    tag: v1.0.0

ingress:
  enabled: true
  host: fall-detection.example.com   # your FQDN

postgres:
  storageClass: ""              # your default StorageClass name

minio:
  storageClass: ""              # your default StorageClass name

inferenceServer:
  fhirServerUrl: ""             # empty = no FHIR push. Set to your FHIR URL if push is required.
```

Secrets are passed via `--set` at install time, not committed to the repo.

---

## 6. Building & Pushing the Images

We build two images. Both are Python 3.11-slim based, FROM scratch (no base image inheritance from anything you don't control).

```bash
# from _6G_Integration_v2_mqtt/ as cwd
REGISTRY=<your-registry-url>
TAG=v1.0.0

docker build -f inference_server/Dockerfile -t $REGISTRY/inference-server:$TAG .
docker build -f fall_dashboard/Dockerfile   -t $REGISTRY/fall-dashboard:$TAG .

docker push $REGISTRY/inference-server:$TAG
docker push $REGISTRY/fall-dashboard:$TAG
```

If your registry needs auth, the chart accepts an `imagePullSecrets` reference — let us know your secret's name and we'll wire it into the deployments.

### What's in the images

- `inference-server`: Python deps + `inference_server/`, `ml_pipeline/`, `shared_db/`, `model/` (XGBoost `.pkl` baked in as startup fallback), `config/`, `fhir_converter.py`. Entrypoint `uvicorn inference_server.server:app --workers 1`.
- `fall-dashboard`: Python deps + `fall_dashboard/`, `ml_pipeline/`, `shared_db/`, `config/`, `alembic.ini` (so the migrate Job can `alembic upgrade head`). Entrypoint `python -m fall_dashboard.main`.

We do **not** include training code, model training data, or any patient data in either image.

### Source code visibility

Once you have the images, you can run them but cannot easily read our Python source. Standard `docker exec` will let a determined user browse the filesystem — this is obfuscation, not encryption, which is acceptable for a hospital research integration. If stronger protection is needed for production we can add PyArmor.

---

## 7. Installing

### One-shot install

```bash
# Create namespace if you don't have automation that does it
kubectl create namespace fall-detection

# Dry-run to validate
helm install fall-detection ./helm/fall-detection \
  --namespace fall-detection \
  --dry-run --debug

# Real install
helm install fall-detection ./helm/fall-detection \
  --namespace fall-detection \
  --set postgres.password=<real-pg-password> \
  --set inferenceServer.apiKeys=<real-api-key> \
  --set grafana.adminPassword=<real-grafana-password> \
  --set minio.rootPassword=<real-minio-password>
```

### Verify

```bash
# All pods running / completed
kubectl get pods -n fall-detection

# PVCs Bound
kubectl get pvc -n fall-detection

# Migrate job succeeded
kubectl logs -n fall-detection job/alembic-migrate

# Bucket-creation job succeeded
kubectl logs -n fall-detection job/create-mlflow-bucket

# Inference server reachable internally
kubectl run -n fall-detection --rm -it --image=curlimages/curl test-curl -- \
  curl http://inference-server:8001/health

# Inference server reachable via Ingress
curl https://<ingress-host>/predict   # 405 expected — POST endpoint
```

### Upgrades

```bash
# Rebuild + push new image
docker build -f inference_server/Dockerfile -t $REGISTRY/inference-server:v1.1 .
docker push $REGISTRY/inference-server:v1.1

# Roll forward
helm upgrade fall-detection ./helm/fall-detection \
  --namespace fall-detection \
  --reuse-values \
  --set images.inferenceServer.tag=v1.1
```

Alembic migrations run automatically as a Helm hook on every upgrade.

---

## 8. NetworkPolicy (if cluster enforces default-deny)

If your cluster enforces NetworkPolicy (which is the right default), we'll need a policy allowing the FOCUS namespace to reach `fall-dashboard` on port 8002. Sketch:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-focus-to-fall-dashboard
  namespace: fall-detection
spec:
  podSelector:
    matchLabels:
      app: fall-dashboard
  policyTypes: ["Ingress"]
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: focus-ns   # ← your FOCUS namespace
      ports:
        - protocol: TCP
          port: 8002
```

Tell us your FOCUS namespace label and we'll add the manifest. Plus a similar policy if you want `inference-server → FHIR Server` allowed in the FOCUS namespace.

---

## 9. Observability — What You Get

- **`/metrics` on `inference-server:8001`** — Prometheus scrape target. Already wired into our Prometheus instance, but if you have a cluster-wide Prometheus (Operator), point it at our Service and we can drop ours.
- **Grafana on port 3000** — three pre-provisioned dashboards: ml_server_overview, model_performance, fall_events_timeline. Datasources auto-provisioned from the chart.
- **Logs** — stdout/stderr on every pod. JSON-structured for the inference server.

If your cluster has Loki / cluster-wide observability, our app logs are already structured and ready to be picked up.

---

## 10. Local Test Path (`Docker Desktop K8s`)

Useful for us to sanity-check changes before pushing to your cluster. Skip this section if not relevant.

```powershell
# Build images locally tagged with the placeholder registry name
docker build -f inference_server/Dockerfile -t registry.example.com/inference-server:latest .
docker build -f fall_dashboard/Dockerfile   -t registry.example.com/fall-dashboard:latest .

# values.yaml has images.pullPolicy=Never for this case — kubelet won't try to pull
helm install fall-detection ./helm/fall-detection \
  --namespace fall-detection --create-namespace \
  --set postgres.password=testpass \
  --set minio.rootPassword=testpass \
  --set grafana.adminPassword=testpass \
  --set inferenceServer.apiKeys=testkey
```

Tear down:

```powershell
helm uninstall fall-detection -n fall-detection
kubectl delete namespace fall-detection
```

When deploying to your real cluster, switch `images.pullPolicy` to `Always` and set the real `registry` value.

---

## 11. Things You Should Push Back On

A few decisions you might want to question:

- **InfluxDB stays in your namespace.** We took this position because biosignals are FOCUS data and the FHIR server is yours. If for any reason you'd rather we host an InfluxDB too, our chart can be extended — but your existing InfluxDB is the source of truth for biosignals and the Patient Dashboard, so dual-hosting would be wasteful.
- **MinIO inside our namespace.** Alternative was using your existing object storage. We went with bundled MinIO because the chart is then self-contained — no shared credentials with you. If you'd prefer we use an existing S3/MinIO, we can swap (one config block).
- **Postgres inside our namespace.** Same trade-off as MinIO. If FOCUS already runs Postgres-as-a-service, we can drop the StatefulSet and just consume a connection string — gives you backups/HA management for free.
- **`replicas: 1` everywhere.** Inference server cannot be scaled out (per-process state). The others could be HA, but we kept everything single-replica for simplicity given the trial scope. Easy to revisit per-service if you want HA.

---

## 12. Reference — Other Docs You Can Read

| File | Audience | What's in it |
|------|----------|--------------|
| `REFACTOR_DOCUS/deployment_architecture.md` | Architecture-level | Detailed two-namespace diagram, full table of who-writes-what, Postgres schema, MinIO/MLflow integration |
| `REFACTOR_DOCUS/helm_guide.md` | Anyone touching the chart | Annotated walkthrough of every template, including manifests for components not covered above |
| `REFACTOR_DOCUS/mqtt_architecture.md` | Anyone debugging the MQTT path | Topic conventions, paho client IDs, the difference between TCP and WebSocket listeners |
| `REFACTOR_DOCUS/todo.md` | Project tracker | Open items for our side and for FOCUS DevOps, with status |
| `handover_docs/ADMIN_ml_ops_related.md` | Whoever runs MLflow / retraining | Model lifecycle, retraining flow, training data semantics |
| `handover_docs/ISA_mobile_app.md` | Mobile app developer | `/predict` and MQTT contracts |

---

## 13. Contact / Escalation

- **Hayate** — overall architecture, inference server, MLflow, Helm chart. Primary integration contact during this hand-over period.
- **Isa** — mobile app + Patient Dashboard UI.
- **Charite** — clinical data, sample rate decisions, data sharing agreements.



----




**ALL** the services go to K8s. But there's a distinction between "we build the image ourselves" and "we use a prebuilt vendor image":

| Service | Image source | Who builds it | Where it lives in production |
| --- | --- | --- | --- |
| **inference_server** | `inference_server/Dockerfile` (our code) | **us** | FOCUS's container registry |
| **fall_dashboard** | `fall_dashboard/Dockerfile` (our code) | **us** | FOCUS's container registry |
| **mlflow** | `infrastructure/mlflow/Dockerfile` (our small wrapper) | **us** | FOCUS's container registry |
| Postgres | `postgres:16-alpine` | postgres-org | Docker Hub (pulled by K8s) |
| MQTT broker | `eclipse-mosquitto:2` | eclipse-org | Docker Hub |
| MinIO | `minio/minio:latest` | minio-org | Docker Hub |
| Prometheus | `prom/prometheus:latest` | prom-org | Docker Hub |
| Grafana | `grafana/grafana:10.4.0` | grafana-org | Docker Hub |

**Production startup ordering — what runs where:**

`FOCUS K8s cluster (our namespace)
├── postgres pod          ← official image, pulled from Docker Hub
├── mqtt pod              ← official image, pulled from Docker Hub
├── minio pod             ← official image, pulled from Docker Hub
├── prometheus pod        ← official image, pulled from Docker Hub
├── grafana pod           ← official image, pulled from Docker Hub
├── mlflow pod            ← OUR image, pulled from FOCUS registry
├── inference_server pod  ← OUR image, pulled from FOCUS registry
└── fall_dashboard pod    ← OUR image, pulled from FOCUS registry`

All eight pods run inside FOCUS's K8s cluster. The Helm chart just declares them as Deployments/StatefulSets and points each one at the right image. K8s itself handles the pulls.

**Corrected version of my earlier statement:** "the three pieces we package ourselves (inference_server, fall_dashboard, mlflow) need their Docker images built and pushed before deployment. Step 12 specifically verifies the two that contain our application code — the third is too thin to need explicit verification beyond `docker-compose up`, which you already ran successfully."