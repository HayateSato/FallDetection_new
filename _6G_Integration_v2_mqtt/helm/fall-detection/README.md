# helm/fall-detection — production Helm chart

Packages the entire `mcs-fall-detection` namespace (8 services) for Kubernetes
deployment. This is what gets shipped to FOCUS DevOps.

| | |
|---|---|
| Chart name | `fall-detection` |
| Target namespace | `mcs-fall-detection` (override in `values.yaml` → `namespaces.ours`) |
| Status | Templates written and `helm lint`-clean. **Not yet smoke-tested end-to-end** — see Step 12.5 in `REFACTOR_DOCUS/todo.md` for the dry-run plan. |

If you've only ever run this stack via `docker-compose`, this README walks
through what changes when you switch to Helm/Kubernetes — and gives you the
exact commands to install it locally on Docker Desktop.

---

## 1. Mental model — Compose vs Helm

You already know how to run the stack with Compose:

```powershell
docker-compose -f infrastructure/docker-compose.yml up
# ... then python -m fall_dashboard.main, etc.
```

Helm does the same thing in Kubernetes:

| Compose concept | Helm/K8s concept |
|----------------|------------------|
| `docker-compose.yml` | `Chart.yaml` + `templates/*.yaml` (one chart per "Compose project") |
| Service in `services:` | One `Deployment` (or `StatefulSet`) + one `Service` |
| `image:` field | `containers.image` in pod spec, looked up from a registry |
| `ports:` field | A K8s `Service` resource with a port mapping |
| `volumes:` for databases | A `PersistentVolumeClaim` (StatefulSet provides per-pod) |
| `environment:` | Pulled from a `ConfigMap` (non-secret) or `Secret` (secret) |
| `depends_on:` | initContainers + readiness probes (no implicit ordering) |
| `up`/`down` | `helm install` / `helm uninstall` |
| Local image cache | Same — Docker Desktop K8s shares your Docker daemon's images |

The Helm chart is just a **templated** version of all these manifests, so the
same chart works on your laptop AND in FOCUS's cluster — only `values.yaml`
differs between environments.

---

## 2. What's inside this chart

```
helm/fall-detection/
├── Chart.yaml                        ← chart metadata
├── values.yaml                       ← all parameters in one file (edit this per env)
├── README.md                         ← this file
│
├── templates/                        ← YAML manifests with Helm templating
│   ├── namespace.yaml                ← creates the mcs-fall-detection namespace
│   ├── configmap.yaml                ← non-secret env vars shared by all pods
│   ├── secrets.yaml                  ← Postgres password, API keys, MinIO creds
│   ├── migrate-job.yaml              ← runs `alembic upgrade head` once on install
│   │
│   ├── inference-server/             ← Deployment + Service (port 8001)
│   ├── fall-dashboard/               ← Deployment + Service + Ingress (port 8002)
│   ├── mqtt-broker/                  ← Deployment + Service (port 1883)
│   ├── postgres/                     ← StatefulSet + Service + ConfigMap
│   ├── mlflow/                       ← Deployment + Service (port 5000)
│   ├── minio/                        ← StatefulSet + Service (port 9000)
│   ├── prometheus/                   ← Deployment + Service (port 9090)
│   └── grafana/                      ← Deployment + Service (port 3000)
│
└── files/
    └── grafana/                      ← provisioned dashboards (mounted as ConfigMap)
```

Eight services, same as Compose. Each gets its own folder under `templates/`
because that scales better than one giant file.

---

## 3. Prerequisites

| Requirement | Verify with |
|-------------|-------------|
| Docker Desktop with Kubernetes enabled | Settings → Kubernetes → Enable Kubernetes (~2 min first time) |
| `kubectl` context = `docker-desktop` | `kubectl config current-context` |
| `helm` v3 installed (PATH) | `helm version` |
| Docker Desktop ≥ 8 GB RAM | Settings → Resources |
| Working dir = `_6G_Integration_v2_mqtt/` | `pwd` |

If `helm version` fails: `winget install Helm.Helm`, close + reopen PowerShell.

---

## 4. Build the local images

The chart's default `values.yaml` uses `pullPolicy: Never` so K8s expects the
images to already be in the local Docker daemon's cache. Build them once:

```powershell
# from _6G_Integration_v2_mqtt/ as cwd
docker build -f inference_server/Dockerfile -t inference-server:latest .
docker build -f fall_dashboard/Dockerfile  -t fall-dashboard:latest  .
```

Verify:

```powershell
docker images | findstr -E "inference-server|fall-dashboard"
# expect both with TAG=latest
```

These tag names MUST match `values.yaml` → `images.inferenceServer.repository`
and `images.fallDashboard.repository`. Don't rename without updating both.

> ml_dashboard and server_health are NOT in this chart yet. They run on your
> laptop only for now. Adding them is a future step (one Deployment +
> Service each, copy/paste of the inference-server template).

---

## 5. Lint the chart

Before installing, sanity-check the templates render correctly:

```powershell
helm lint .\helm\fall-detection
helm template fall-detection .\helm\fall-detection > rendered.yaml
# inspect rendered.yaml in an editor — should look like normal K8s YAML
```

`helm lint` should print `0 chart(s) failed`. If it complains about an
indented value or undefined variable, that's an error in a template, not in
your cluster.

---

## 6. Install the chart

```powershell
helm install mcs-fall-detection .\helm\fall-detection `
    --namespace mcs-fall-detection `
    --create-namespace `
    --wait --timeout 5m
```

What happens, in order:

1. `--create-namespace` creates `mcs-fall-detection`
2. `Secret` and `ConfigMap` are applied
3. `Postgres` StatefulSet starts; readiness probe waits for `pg_isready`
4. `MinIO` StatefulSet starts; readiness probe waits for `/minio/health/live`
5. `MLflow` Deployment starts; depends on Postgres + MinIO
6. `migrate-job` runs `alembic upgrade head` once against Postgres
7. `inference-server`, `fall-dashboard`, `mqtt-broker`, `Prometheus`, `Grafana` all roll out in parallel
8. `--wait` blocks until every Pod is Ready or 5 min elapses

If `--wait` times out, see [§ 9 Troubleshooting](#9-troubleshooting). The
namespace remains in whatever partial state it reached — fix the root cause,
then `helm upgrade` to retry.

---

## 7. Verify

```powershell
kubectl get pods -n mcs-fall-detection
```

Expected (all `Running` or `Completed`):

```
NAME                                READY   STATUS      RESTARTS   AGE
fall-dashboard-xxxxxxxxxx-xxxxx     1/1     Running     0          90s
grafana-xxxxxxxxxx-xxxxx            1/1     Running     0          90s
inference-server-xxxxxxxxxx-xxxxx   1/1     Running     0          90s
mcs-fall-detection-migrate-xxxxx    0/1     Completed   0          80s
minio-0                             1/1     Running     0          90s
mlflow-xxxxxxxxxx-xxxxx             1/1     Running     0          85s
mqtt-broker-xxxxxxxxxx-xxxxx        1/1     Running     0          90s
postgres-0                          1/1     Running     0          90s
prometheus-xxxxxxxxxx-xxxxx         1/1     Running     0          90s
```

Probe the services from inside the cluster (use `curl`; `wget` is not installed
in these images):

```powershell
kubectl exec -n mcs-fall-detection deploy/inference-server -- curl -s http://localhost:8001/health
# expect: {"status":"ok","model_version":"v0", ...}

kubectl exec -n mcs-fall-detection deploy/fall-dashboard -- curl -s http://localhost:8002/api/patients
# expect: {"patients":[]}
```

If `curl` is also missing, fall back to Python:

```powershell
kubectl exec -n mcs-fall-detection deploy/inference-server -- python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8001/health').read().decode())"
```

**All services are `ClusterIP` — they are only reachable inside the cluster.**
Your laptop and browser are outside the cluster, so port-forward is required
to reach any service locally. There is no other way on Docker Desktop:

```powershell
# Run each in a separate terminal; keep it running while you test
kubectl port-forward -n mcs-fall-detection svc/inference-server 8001:8001
kubectl port-forward -n mcs-fall-detection svc/fall-dashboard   8002:8002
kubectl port-forward -n mcs-fall-detection svc/mlflow           5000:5000
kubectl port-forward -n mcs-fall-detection svc/grafana          3000:3000
```

Then open `http://localhost:8002/` in a browser. The tunnel closes when you
`Ctrl+C` the port-forward process.

---

## 8. Map of common dev tasks

| Task | Compose command | Helm/K8s command |
|------|-----------------|------------------|
| Start everything | `docker-compose up -d` | `helm install ...` (above) |
| Stop everything | `docker-compose down` | `helm uninstall mcs-fall-detection -n mcs-fall-detection` |
| View logs of inference_server | `docker logs fall_inference` | `kubectl logs -n mcs-fall-detection deploy/inference-server` |
| Tail logs live | `docker logs -f ...` | `kubectl logs -f -n mcs-fall-detection deploy/inference-server` |
| Rebuild after a code change | `docker-compose up -d --build inference_server` | `docker build -t inference-server:latest .` then `kubectl rollout restart -n mcs-fall-detection deploy/inference-server` |
| Edit env var | edit `.env`, restart container | edit `values.yaml`, run `helm upgrade ...` |
| psql into Postgres | `docker exec -it fall_postgres psql ...` | `kubectl exec -it -n mcs-fall-detection postgres-0 -- psql ...` |
| Open MLflow UI | `http://localhost:5000` | `kubectl port-forward -n mcs-fall-detection svc/mlflow 5000:5000` then same URL |

The pattern is the same: replace `docker exec` with `kubectl exec`,
`docker-compose` with `helm`, and `localhost:N` for non-ingressed services
needs `kubectl port-forward`.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `helm install` times out at "waiting for pods" | Docker Desktop K8s out of resources | Increase Docker Desktop memory to ≥ 8 GB and retry |
| `ImagePullBackOff` on `inference-server` | Image not built locally | Run the `docker build` commands in [§ 4](#4-build-the-local-images) |
| `ErrImagePull` on `postgres-0` / `minio-0` | Need internet to pull official images | Confirm Docker Desktop has internet access |
| `migrate-job` stuck at `Pending` | Postgres not ready yet | This normally resolves within ~30s; if longer, `kubectl describe pod -n mcs-fall-detection postgres-0` |
| `migrate-job` `Error` status | `alembic upgrade head` failed | `kubectl logs -n mcs-fall-detection job/mcs-fall-detection-migrate` to see why |
| `postgres-0` `CrashLoopBackOff` | Wrong password or PVC issue | Delete the PVC + retry: `kubectl delete pvc -n mcs-fall-detection data-postgres-0` |
| Browser can't reach `http://localhost:8002` | No NodePort / no ingress on Docker Desktop | Use `kubectl port-forward -n mcs-fall-detection svc/fall-dashboard 8002:8002` |
| `inference-server` log shows "MLflow connection refused" | MLflow pod not ready when inference-server started | `kubectl rollout restart deploy/inference-server -n mcs-fall-detection` |

---

## 10. Cleanup

```powershell
helm uninstall mcs-fall-detection -n mcs-fall-detection
kubectl delete namespace mcs-fall-detection
```

The namespace deletion also removes any leftover PVCs (databases, MinIO data).
If you want to **keep** the data between installs, omit the namespace delete.

---

## 11. What's still TODO before this ships to FOCUS

- [ ] Smoke-test end-to-end on Docker Desktop K8s (todo.md Step 12.5 — the
      mock-focus dry-run validates cross-namespace traffic too)
- [ ] Add `ml_dashboard` and `server_health` Deployments + Services
      (currently those run only on the developer's laptop)
- [ ] Add NetworkPolicy YAML once FOCUS DevOps confirms whether their
      cluster enforces them (todo.md Step 14)
- [ ] Add an `imagePullSecret` reference in pod templates once FOCUS DevOps
      provides the secret name (Section 9 of the FOCUS DevOps Q&A)
- [ ] Replace placeholder values in `values.yaml` with FOCUS-supplied real
      values in a separate `values-overrides.yaml` (FOCUS DevOps creates
      this in their internal Git, not ours)
- [ ] Helm chart unit tests via `helm unittest` (nice-to-have, not blocking)

For the conceptual mapping of compose → Helm and the broader handover plan,
see `handover_docs/Tech_integrator.md`.
