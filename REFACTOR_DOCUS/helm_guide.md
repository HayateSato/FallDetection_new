# Helm Deployment Guide — Fall Detection / FOCUS Integration

**Status:** Step 9 — blocked on FOCUS DevOps answers (namespace names, container registry).
This guide is written with placeholders where those answers are still needed.

---

## Key mental model — your laptop and the cluster are separate machines

`localhost` on your laptop is not the same as `localhost` inside the cluster. When a cluster
node pulls a Docker image, it is the node making the network request — not your laptop.
A registry at `localhost:5000` on your laptop is invisible to the cluster.

```
YOUR LAPTOP                         FOCUS CLUSTER (separate machines)
  Docker Desktop                      Kubernetes Control Plane
    └── Docker Engine                 Node 1
        └── container (local test)      └── Pod: inference-server  ← image runs HERE
  kubectl / helm  (CLI tools only)    Node 2
  localhost = your laptop               └── Pod: postgres
                                      localhost = that node, NOT your laptop
```

### Container registry options

| Option | Notes |
|--------|-------|
| FOCUS internal registry | **Best option** — private, already trusted by their cluster. Ask for the URL. |
| `ghcr.io` (GitHub) | Free for public images. Good fallback. |
| Docker Hub | Free, public. Simple to use. |
| `localhost` | **Never works** from a cluster node. Local Docker testing only. |

---

## Prerequisites

- `kubectl` installed and connected to the target cluster
- `helm` v3 installed
- Docker images built and pushed to a container registry (see Step 1)
- FOCUS DevOps has confirmed:
  - [ ] Kubernetes namespace names (placeholders used below: `focus-ns` for theirs, `mcs-fall-detection` for ours)
  - [ ] Container registry URL (placeholder: `registry.example.com`)
  - [ ] Ingress controller type (nginx / traefik / other)
  - [ ] Ingress host / domain for our services
  - [ ] Default StorageClass name for PVCs

---

## Two-Namespace Overview

```
┌─────────────────────────────┐     ┌──────────────────────────────────────────┐
│   FOCUS NAMESPACE           │     │   OUR NAMESPACE                          │
│   (focus-ns)                │     │   (mcs-fall-detection)                   │
│                             │     │                                          │
│   InfluxDB                  │     │   inference-server   ClusterIP :8001     │
│   FHIR Server               │     │   fall-dashboard     ClusterIP :8002     │
│   Patient Dashboard         │     │   mqtt-broker        ClusterIP :1883     │
│     (Isa's UI —             │     │   postgres           ClusterIP :5432     │
│      reads our API)  ───────┼─────┼──► fall-dashboard /api/*                │
│                             │     │   mlflow             ClusterIP :5000     │
│                             │     │   prometheus         ClusterIP :9090     │
│                             │     │   grafana            ClusterIP :3000     │
│                             │     │   minio              ClusterIP :9000     │
└─────────────────────────────┘     └──────────────────────────────────────────┘

External traffic (mobile app) ──────────────────► inference-server (Ingress)
```

Cross-namespace calls (all outbound from our namespace):
- `inference_server` → FHIR Server (FOCUS) — optional FHIR push
- `mock_app` / mobile app → InfluxDB (FOCUS) — sensor data read
- Patient Dashboard (FOCUS) → `fall_dashboard` (ours) — reads fall API

---

## Step 1 — Write Dockerfiles

One Dockerfile per Python service. Place each in the component's folder.

### `inference_server/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy shared dependencies first (better layer caching)
COPY shared_db/ shared_db/
COPY ml_pipeline/ ml_pipeline/
COPY config/ config/
COPY model/ model/
COPY fhir_converter.py .

# Install inference server dependencies
COPY inference_server/requirements.txt inference_server/requirements.txt
RUN pip install --no-cache-dir -r inference_server/requirements.txt

COPY inference_server/ inference_server/

EXPOSE 8001
CMD ["uvicorn", "inference_server.server:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
```

### `fall_dashboard/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY shared_db/ shared_db/
COPY ml_pipeline/ ml_pipeline/
COPY config/ config/

COPY fall_dashboard/requirements.txt fall_dashboard/requirements.txt
RUN pip install --no-cache-dir -r fall_dashboard/requirements.txt

COPY fall_dashboard/ fall_dashboard/
COPY alembic.ini .

EXPOSE 8002
CMD ["python", "-m", "fall_dashboard.main"]
```

### Build and push images

```bash
# Replace registry.example.com with your actual registry URL
REGISTRY=registry.example.com

docker build -f inference_server/Dockerfile -t $REGISTRY/inference-server:latest .
docker build -f fall_dashboard/Dockerfile   -t $REGISTRY/fall-dashboard:latest .

docker push $REGISTRY/inference-server:latest
docker push $REGISTRY/fall-dashboard:latest
```

---

## Step 2 — Create the Helm chart structure

```
helm/
└── fall-detection/
    ├── Chart.yaml
    ├── values.yaml
    └── templates/
        ├── namespace.yaml
        ├── secrets.yaml
        ├── configmap.yaml
        ├── inference-server/
        │   ├── deployment.yaml
        │   └── service.yaml
        ├── fall-dashboard/
        │   ├── deployment.yaml
        │   ├── service.yaml
        │   └── ingress.yaml
        ├── mqtt-broker/
        │   ├── deployment.yaml
        │   ├── service.yaml
        │   └── configmap.yaml
        ├── postgres/
        │   ├── statefulset.yaml
        │   ├── service.yaml
        │   └── pvc.yaml
        ├── mlflow/
        │   ├── deployment.yaml
        │   └── service.yaml
        ├── prometheus/
        │   ├── deployment.yaml
        │   ├── service.yaml
        │   └── configmap.yaml
        └── grafana/
            ├── deployment.yaml
            └── service.yaml
```

Create the chart scaffold:

```bash
mkdir -p helm/fall-detection/templates/{inference-server,fall-dashboard,mqtt-broker,postgres,mlflow,prometheus,grafana}
```

---

## Step 3 — Chart.yaml

```yaml
# helm/fall-detection/Chart.yaml
apiVersion: v2
name: fall-detection
description: Fall Detection — 6G / FOCUS Charite Integration
type: application
version: 0.1.0
appVersion: "1.0.0"
```

---

## Step 4 — values.yaml

This is the single file you change per environment (dev / staging / production).

```yaml
# helm/fall-detection/values.yaml

# ── Namespaces ────────────────────────────────────────────────────────────────
namespaces:
  ours: mcs-fall-detection    # confirm with FOCUS DevOps
  focus: focus-ns             # confirm with FOCUS DevOps

# ── Container registry ───────────────────────────────────────────────────────
registry: registry.example.com   # confirm with FOCUS DevOps

# ── Image tags ───────────────────────────────────────────────────────────────
images:
  inferenceServer:
    repository: inference-server
    tag: latest
  fallDashboard:
    repository: fall-dashboard
    tag: latest

# ── Inference server ─────────────────────────────────────────────────────────
inferenceServer:
  port: 8001
  modelVersion: v0
  apiKeys: "changeme"
  fhirServerUrl: ""           # set to FOCUS FHIR server URL in production

# ── Fall dashboard ────────────────────────────────────────────────────────────
fallDashboard:
  port: 8002

# ── MQTT broker ───────────────────────────────────────────────────────────────
mqtt:
  port: 1883
  websocketPort: 9001
  image: eclipse-mosquitto:2

# ── Postgres ──────────────────────────────────────────────────────────────────
postgres:
  image: postgres:16-alpine
  port: 5432
  database: fall_detection
  user: fall_user
  password: fall_pass         # override via --set or external secret in production
  storageSize: 10Gi

# ── MLflow ────────────────────────────────────────────────────────────────────
mlflow:
  image: ghcr.io/mlflow/mlflow:latest
  port: 5000

# ── MinIO (MLflow artifact store) ─────────────────────────────────────────────
minio:
  image: minio/minio:latest
  port: 9000
  storageSize: 20Gi
  rootUser: minio
  rootPassword: minio123      # override in production

# ── Prometheus ────────────────────────────────────────────────────────────────
prometheus:
  image: prom/prometheus:latest
  port: 9090

# ── Grafana ───────────────────────────────────────────────────────────────────
grafana:
  image: grafana/grafana:10.4.0
  port: 3000
  adminPassword: admin        # override in production

# ── Ingress ───────────────────────────────────────────────────────────────────
ingress:
  enabled: true
  host: fall-detection.example.com    # confirm with FOCUS DevOps
  inferenceServerPath: /predict
  fallDashboardPath: /api
```

---

## Step 5 — Namespace template

```yaml
# helm/fall-detection/templates/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: {{ .Values.namespaces.ours }}
```

---

## Step 6 — Secrets

Keep secrets out of `values.yaml` in production. Create a secret template:

```yaml
# helm/fall-detection/templates/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: fall-detection-secrets
  namespace: {{ .Values.namespaces.ours }}
type: Opaque
stringData:
  postgres-password: {{ .Values.postgres.password | quote }}
  minio-root-password: {{ .Values.minio.rootPassword | quote }}
  api-keys: {{ .Values.inferenceServer.apiKeys | quote }}
  grafana-admin-password: {{ .Values.grafana.adminPassword | quote }}
```

In production, override secrets at install time:

```bash
helm install fall-detection ./helm/fall-detection \
  --set postgres.password=<real-password> \
  --set inferenceServer.apiKeys=<real-key>
```

---

## Step 7 — ConfigMap (shared .env values)

```yaml
# helm/fall-detection/templates/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fall-detection-config
  namespace: {{ .Values.namespaces.ours }}
data:
  DATABASE_URL: "postgresql+psycopg2://{{ .Values.postgres.user }}:{{ .Values.postgres.password }}@postgres:{{ .Values.postgres.port }}/{{ .Values.postgres.database }}"
  MQTT_BROKER_HOST: "mqtt-broker"
  MQTT_BROKER_PORT: "{{ .Values.mqtt.port }}"
  MQTT_ALERT_TOPIC: "fall/alert"
  MODEL_VERSION: "{{ .Values.inferenceServer.modelVersion }}"
  FHIR_SERVER_URL: "{{ .Values.inferenceServer.fhirServerUrl }}"
  MLFLOW_TRACKING_URI: "http://mlflow:{{ .Values.mlflow.port }}"
  CAREGIVER_PORT: "{{ .Values.fallDashboard.port }}"
  HARDWARE_ACC_SAMPLE_RATE: "50"
  ACC_SENSOR_TYPE: "bosch"
```

---

## Step 8 — Postgres (StatefulSet)

Postgres must be a `StatefulSet` (not `Deployment`) so the data volume is stable across pod restarts.

```yaml
# helm/fall-detection/templates/postgres/statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: {{ .Values.namespaces.ours }}
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: {{ .Values.postgres.image }}
          ports:
            - containerPort: {{ .Values.postgres.port }}
          env:
            - name: POSTGRES_USER
              value: {{ .Values.postgres.user }}
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: fall-detection-secrets
                  key: postgres-password
            - name: POSTGRES_DB
              value: {{ .Values.postgres.database }}
          volumeMounts:
            - name: postgres-data
              mountPath: /var/lib/postgresql/data
            - name: init-sql
              mountPath: /docker-entrypoint-initdb.d
          readinessProbe:
            exec:
              command: ["pg_isready", "-U", "{{ .Values.postgres.user }}"]
            initialDelaySeconds: 5
            periodSeconds: 5
      volumes:
        - name: init-sql
          configMap:
            name: postgres-init-sql
  volumeClaimTemplates:
    - metadata:
        name: postgres-data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: {{ .Values.postgres.storageSize }}
```

```yaml
# helm/fall-detection/templates/postgres/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: {{ .Values.namespaces.ours }}
spec:
  selector:
    app: postgres
  ports:
    - port: {{ .Values.postgres.port }}
      targetPort: {{ .Values.postgres.port }}
  clusterIP: None   # headless — required for StatefulSet
```

---

## Step 9 — MQTT broker (Deployment)

```yaml
# helm/fall-detection/templates/mqtt-broker/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mqtt-broker
  namespace: {{ .Values.namespaces.ours }}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mqtt-broker
  template:
    metadata:
      labels:
        app: mqtt-broker
    spec:
      containers:
        - name: mosquitto
          image: {{ .Values.mqtt.image }}
          ports:
            - containerPort: {{ .Values.mqtt.port }}
            - containerPort: {{ .Values.mqtt.websocketPort }}
          volumeMounts:
            - name: mosquitto-config
              mountPath: /mosquitto/config/mosquitto.conf
              subPath: mosquitto.conf
      volumes:
        - name: mosquitto-config
          configMap:
            name: mosquitto-config
```

```yaml
# helm/fall-detection/templates/mqtt-broker/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: mqtt-broker
  namespace: {{ .Values.namespaces.ours }}
spec:
  selector:
    app: mqtt-broker
  ports:
    - name: mqtt
      port: {{ .Values.mqtt.port }}
      targetPort: {{ .Values.mqtt.port }}
    - name: websocket
      port: {{ .Values.mqtt.websocketPort }}
      targetPort: {{ .Values.mqtt.websocketPort }}
```

---

## Step 10 — Inference server (Deployment)

```yaml
# helm/fall-detection/templates/inference-server/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-server
  namespace: {{ .Values.namespaces.ours }}
spec:
  replicas: 1           # must stay 1 — Prometheus counters are per-process
  selector:
    matchLabels:
      app: inference-server
  template:
    metadata:
      labels:
        app: inference-server
    spec:
      containers:
        - name: inference-server
          image: {{ .Values.registry }}/{{ .Values.images.inferenceServer.repository }}:{{ .Values.images.inferenceServer.tag }}
          ports:
            - containerPort: {{ .Values.inferenceServer.port }}
          envFrom:
            - configMapRef:
                name: fall-detection-config
          env:
            - name: API_KEYS
              valueFrom:
                secretKeyRef:
                  name: fall-detection-secrets
                  key: api-keys
          readinessProbe:
            httpGet:
              path: /health
              port: {{ .Values.inferenceServer.port }}
            initialDelaySeconds: 10
            periodSeconds: 10
```

```yaml
# helm/fall-detection/templates/inference-server/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: inference-server
  namespace: {{ .Values.namespaces.ours }}
spec:
  selector:
    app: inference-server
  ports:
    - port: {{ .Values.inferenceServer.port }}
      targetPort: {{ .Values.inferenceServer.port }}
```

---

## Step 11 — Fall dashboard (Deployment + Ingress)

```yaml
# helm/fall-detection/templates/fall-dashboard/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fall-dashboard
  namespace: {{ .Values.namespaces.ours }}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fall-dashboard
  template:
    metadata:
      labels:
        app: fall-dashboard
    spec:
      containers:
        - name: fall-dashboard
          image: {{ .Values.registry }}/{{ .Values.images.fallDashboard.repository }}:{{ .Values.images.fallDashboard.tag }}
          ports:
            - containerPort: {{ .Values.fallDashboard.port }}
          envFrom:
            - configMapRef:
                name: fall-detection-config
          readinessProbe:
            httpGet:
              path: /api/patients
              port: {{ .Values.fallDashboard.port }}
            initialDelaySeconds: 10
            periodSeconds: 10
```

The Ingress exposes `fall_dashboard` to the FOCUS namespace so the Patient Dashboard (Isa's UI) can call `/api/falls` and `/api/stream`:

```yaml
# helm/fall-detection/templates/fall-dashboard/ingress.yaml
{{- if .Values.ingress.enabled }}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fall-dashboard-ingress
  namespace: {{ .Values.namespaces.ours }}
  annotations:
    nginx.ingress.kubernetes.io/proxy-buffering: "off"       # required for SSE
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"   # keep SSE connections alive
spec:
  rules:
    - host: {{ .Values.ingress.host }}
      http:
        paths:
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: fall-dashboard
                port:
                  number: {{ .Values.fallDashboard.port }}
          - path: /predict
            pathType: Prefix
            backend:
              service:
                name: inference-server
                port:
                  number: {{ .Values.inferenceServer.port }}
{{- end }}
```

---

## Step 12 — Run Alembic migrations as a Kubernetes Job

Tables must be created before the services start. Use a Kubernetes `Job`:

```yaml
# helm/fall-detection/templates/migrate-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: alembic-migrate
  namespace: {{ .Values.namespaces.ours }}
  annotations:
    helm.sh/hook: post-install,post-upgrade
    helm.sh/hook-weight: "-5"
    helm.sh/hook-delete-policy: hook-succeeded
spec:
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: alembic
          image: {{ .Values.registry }}/{{ .Values.images.fallDashboard.repository }}:{{ .Values.images.fallDashboard.tag }}
          command: ["alembic", "upgrade", "head"]
          envFrom:
            - configMapRef:
                name: fall-detection-config
```

The `helm.sh/hook: post-install,post-upgrade` annotation means this Job runs automatically after every `helm install` or `helm upgrade` — you never need to run migrations manually in production.

---

## Step 13 — Cross-namespace access for Patient Dashboard

Two addresses exist simultaneously. Which one Isa uses depends on where the Patient Dashboard runs:

| Address | Used by |
|---------|---------|
| `http://fall-dashboard.mcs-fall-detection.svc.cluster.local:8002` | Server-side app running inside the FOCUS namespace |
| `https://fall-detection.example.com/api` | Browser app (JavaScript in the user's browser, outside the cluster) |

**Clarify with Isa:** Is the Patient Dashboard a browser app (React/Vue SPA) or a server-side app?
- **Browser app** → must use the Ingress / public URL. The browser is outside the cluster.
- **Server-side app** → can use the internal DNS name directly.

Both can exist at the same time — they are not mutually exclusive. The internal DNS name is for
pod-to-pod calls inside the cluster. The Ingress URL is for anything outside.

For SSE (`/api/stream`), the `nginx.ingress.kubernetes.io/proxy-buffering: "off"` annotation in
`ingress.yaml` ensures the connection stays alive. This annotation is already set.

---

## Step 14 — Install

```bash
# Create the namespace first (if not auto-created by the template)
kubectl create namespace mcs-fall-detection

# Dry run — check what would be deployed
helm install fall-detection ./helm/fall-detection \
  --namespace mcs-fall-detection \
  --dry-run --debug

# Install
helm install fall-detection ./helm/fall-detection \
  --namespace mcs-fall-detection \
  --set postgres.password=<real-password> \
  --set inferenceServer.apiKeys=<real-api-key> \
  --set grafana.adminPassword=<real-password> \
  --set minio.rootPassword=<real-password>     # do not forget this — was missing in earlier version

# Verify everything is running
kubectl get pods -n mcs-fall-detection
kubectl get services -n mcs-fall-detection
```

---

## Step 15 — Upgrade after code changes

```bash
# Rebuild and push new image
docker build -f inference_server/Dockerfile -t registry.example.com/inference-server:v1.1 .
docker push registry.example.com/inference-server:v1.1

# Upgrade the release with new image tag
helm upgrade fall-detection ./helm/fall-detection \
  --namespace mcs-fall-detection \
  --set images.inferenceServer.tag=v1.1
```

Alembic migrations run automatically as a Job on every `helm upgrade`.

---

## Source code protection

Running inside a Helm namespace means FOCUS installs and uses your system without seeing your source code.

**What FOCUS receives:**
- Your Docker images — contains packaged code. They can run it but cannot easily read your Python source.
- Your Helm chart (`helm/fall-detection/`) — contains YAML infrastructure templates, not application logic.

**What FOCUS never receives:** Python source files, model training code, business logic.

**Caveat:** A motivated person can `docker exec` into a running container and browse the filesystem.
This is obfuscation, not cryptographic protection. For a hospital research integration this is
standard and accepted.

**Optional — PyArmor for stronger protection** (not needed for FOCUS):
```bash
pip install pyarmor
pyarmor gen inference_server/   # outputs encrypted bytecode to dist/
# Dockerfile then copies from dist/ instead of raw source
```

---

## What happens when `helm install` runs — startup chain

Nothing starts all at once. Each step is a controller watching for its piece and reacting independently:

```
helm install fall-detection ./helm/fall-detection
  → Helm renders templates → sends YAML to K8s API Server
  → API Server validates and saves to etcd (cluster database)
  → Deployment Controller sees "want 1 pod, have 0" → creates ReplicaSet
  → ReplicaSet Controller creates Pod object (not running yet)
  → Scheduler assigns Pod to a Node
  → Kubelet on that Node pulls image from registry
  → Kubelet starts container
  → Readiness probe passes: GET /health returns 200
  → Endpoints Controller adds Pod IP to Service
  → Ingress Controller updates nginx routing rules
  → Traffic can now reach the service
```

**A Service is not a process.** It is implemented as `iptables` rules on every node by
`kube-proxy`. When a packet hits the Service ClusterIP, the kernel intercepts and redirects
it directly to a pod IP — no extra hop, no intermediary container.

**The readiness probe is the critical safety mechanism.** It tells the Service "this pod is
ready for traffic". If a pod crashes and restarts, traffic stops flowing to it until the probe
passes again. This is already configured correctly on `inference-server` (`initialDelaySeconds: 10`).

---

## Open questions — confirm with FOCUS DevOps before deploying

| Item | What to ask | Impact |
|------|-------------|--------|
| **Registry URL** | "What is your container registry URL?" | `values.yaml` → `registry:` field |
| **Namespace names** | Confirm `fall-detection` is ok; what is their namespace called? | All templates |
| **Ingress host / domain** | "What domain should our services be reachable at?" | `values.yaml` → `ingress.host` |
| **Ingress controller type** | nginx, traefik, or other? | Annotations in `ingress.yaml` |
| **NetworkPolicy enforcement** | "Do you block cross-namespace traffic by default?" | Extra config needed if yes |
| **StorageClass name** | "What is your default StorageClass?" | `postgres.storageClass` + `minio.storageClass` in `values.yaml` |
| **Where Patient Dashboard runs** | Browser app or server-side? | Whether Isa uses internal DNS or Ingress URL |
| **InfluxDB location** | FOCUS-hosted or do we bring our own? | Whether to add InfluxDB StatefulSet to chart |
