# Isa — Local System Quickstart

**Audience:** Isa (SmarKo), porting the existing `mock_app` behaviour into the real React Native mobile app.
**Goal:** spin up the **whole fall-detection backend** on your laptop in two K8s namespaces, so your phone app can hit it during development.
**Reading order:** this doc → [`04_mobile_app_integration.md`](04_mobile_app_integration.md) for the JSON data contract.

You don't need to understand the whole stack. You need to (a) get it running, (b) point your phone app at it, (c) tear it down. Three sections cover all three.

---

## 1. Prerequisites

Install these once. All four are free.

| Tool | What for | Install link / verify |
|------|----------|------------------------|
| **Docker Desktop** with **Kubernetes enabled** | runs the whole stack as containers in a local K8s cluster | https://www.docker.com/products/docker-desktop/ → after install: **Settings → Kubernetes → Enable Kubernetes → Apply & Restart** |
| **kubectl** | talks to the K8s cluster | Bundled with Docker Desktop. Verify: `kubectl version --client` |
| **helm** v3+ | installs the chart (the YAML that defines all the services) | https://helm.sh/docs/intro/install/ — on Windows: `winget install Helm.Helm`. Verify: `helm version` |
| **git** | clones the repo | https://git-scm.com/. Verify: `git --version` |

### Resource check

Open Docker Desktop → **Settings → Resources** and confirm at least:

| Resource | Minimum |
|----------|---------|
| CPUs | 4 |
| Memory | **8 GB** (10 is safer) |
| Disk | 30 GB free |

If memory is below 8 GB, the install hangs at "waiting for pods" — the symptom is not obvious. Bump it before starting.

### One-time context check

After enabling K8s in Docker Desktop, your `kubectl` should automatically point at the local cluster. Verify:

```powershell
kubectl config current-context
# expected: docker-desktop
```

If it shows anything else (e.g. `gke-...`, `minikube`):

```powershell
kubectl config use-context docker-desktop
```

The install script also checks this and refuses to run otherwise — so you can't accidentally install into someone else's cluster.

---

## 2. Repo layout — only the parts you'll touch

Clone the repo and `cd _6G_Integration_v2_mqtt`. Everything below is relative to that folder.

```
_6G_Integration_v2_mqtt/
│
├── local_dev/
│   └── mock_app/                ← THE REFERENCE IMPLEMENTATION you're replicating in React Native.
│                                  Reads InfluxDB instead of BLE, but the steps after that are identical
│                                  to what your real phone app must do. Treat this as the spec.
│
├── helm/
│   ├── fall-detection/          ← The "real" namespace (mcs-fall-detection): inference-server,
│   │                             fall-dashboard, MQTT broker, Postgres, MLflow, MinIO, Prometheus,
│   │                             Grafana, ml-dashboard, server-health.
│   │   ├── build.ps1            Build the 5 custom images locally.
│   │   ├── install.ps1          Install ONLY the fall-detection chart.
│   │   ├── teardown.ps1         Uninstall ONLY the fall-detection chart.
│   │   └── port-forward.ps1     Open all UI tunnels on localhost:8002, 8004, 8006, 5000, 3000.
│   │
│   └── mock-focus/              ← The "FOCUS-side" mock namespace (mock-focus): mock FHIR,
│                                  mock InfluxDB, mock Patient Dashboard browser UI.
│       ├── build.ps1            Build the 2 mock images.
│       ├── install.ps1          ★ INSTALLS BOTH CHARTS — this is the one you run.
│       ├── test.ps1             Cross-namespace tests (4 of them).
│       └── teardown.ps1         ★ TEARS DOWN BOTH NAMESPACES.
│
├── inference_server/            ← FastAPI service your phone app POSTs to. (You don't edit this.)
├── fall_dashboard/              ← FastAPI service your phone app PUBLISHES to (via MQTT). (You don't edit this.)
├── ml_pipeline/                 ← shared ML code used by inference-server. (Don't touch.)
├── shared_db/                   ← shared Postgres schema. (Don't touch.)
├── model/                       ← XGBoost model files. (Don't touch.)
└── .env                         ← Local dev env file — has the INFERENCE_API_KEY your app needs.
```

### What lives in which namespace

When everything is up, you have **two namespaces** in K8s:

| Namespace | Owned by | What's in it | Why you care |
|-----------|----------|--------------|--------------|
| `mcs-fall-detection` | us (MCS) | inference-server (`:8001`), MQTT broker (`:1883` TCP / `:9001` WS), fall-dashboard (`:8002`), Postgres, MLflow, MinIO, Prometheus, Grafana, ml-dashboard, server-health | **Your phone app talks here:** POST /predict to inference-server, MQTT PUBLISH to the broker. |
| `mock-focus` | local dry-run only | mock FHIR (`:8003`), mock InfluxDB (`:8086`), mock Patient Dashboard browser UI (NodePort `30090`) | **Open `http://localhost:30090` in a browser** to see what the FOCUS Patient Dashboard will look like. When your phone app fires a fall, this dashboard goes red. That's your end-to-end visual confirmation. |

The mock-focus namespace is fake — in production it's replaced by FOCUS's real services. But for your local dev it's the closest thing to "real production cluster on a laptop".

---

## 3. Bring everything up

### 3.1 Build the images (first time + after any code change)

Two scripts, run from `_6G_Integration_v2_mqtt/`. Together they take ~5 minutes the first time, ~30 s on subsequent rebuilds (Docker layer cache).

```powershell
.\helm\fall-detection\build.ps1     # builds the 5 real-system images
.\helm\mock-focus\build.ps1         # builds the 2 mock-side images
```

Verify the images exist:

```powershell
docker images | findstr -E "inference-server|fall-dashboard|ml-dashboard|server-health|mlflow|fd-mock"
# expected: 7 lines
```

### 3.2 Install both namespaces — one script

```powershell
.\helm\mock-focus\install.ps1
```

Despite living under `mock-focus/`, this script installs **both charts** (mock-focus first, then fall-detection). It uses `helm upgrade --install --wait --timeout 5m`, so it returns only when every pod is Ready.

Expected output at the end: two pod tables, all Running (or Completed for the one-shot Jobs).

If it fails or times out: see [Troubleshooting](#5-troubleshooting).

---

## 4. Validate it's actually working

Three layers of validation — do them in this order. Each one costs you ~30 seconds.

### Layer 1 — pods are alive

```powershell
kubectl get pods -n mcs-fall-detection
kubectl get pods -n mock-focus
```

**What you want to see:** every pod in `Running` (or `Completed` for `alembic-migrate` and `create-mlflow-bucket`). Restart counts of 0 or 1 are normal; anything ≥ 3 means something is in a crash loop — check logs.

### Layer 2 — cross-namespace plumbing works

```powershell
.\helm\mock-focus\test.ps1
```

Four tests run in sequence: cross-namespace DNS, HTTP from mock-focus → fall-dashboard, HTTP from inference-server → mock FHIR, mock InfluxDB health. **You want 4/4 PASS.**

### Layer 3 — see a real fall trigger end-to-end (the one that proves the system works for *your* use case)

This is the test that mirrors what your phone app will be doing.

**Step A — open the mock Patient Dashboard in a browser:**

```
http://localhost:30090/
```

You should see two patient cards. They came from `fall-dashboard` two namespaces away. The browser also has an SSE connection open, waiting for fall alerts.

**Step B — drive a fall through `mock_app` (this is what your phone app will replace).**

Open three terminals:

```powershell
# Terminal A — port-forward inference-server (so mock_app can reach it)
kubectl port-forward -n mcs-fall-detection svc/inference-server 8001:8001
```

```powershell
# Terminal B — port-forward MQTT broker (so mock_app can publish alerts)
kubectl port-forward -n mcs-fall-detection svc/mqtt-broker 1883:1883
```

```powershell
# Terminal C — run mock_app (it reads test data, fires /predict, then publishes MQTT)
$env:INFERENCE_API_URL = "http://127.0.0.1:8001"
$env:MQTT_BROKER_HOST  = "127.0.0.1"   # IMPORTANT: use 127.0.0.1, NOT localhost
                                        # (Windows resolves localhost to ::1 first, port-forward only binds IPv4)
python -m local_dev.mock_app.main
```

Within ~10 seconds you should see a **red flag** appear on the patient card at `localhost:30090`. That confirms:
- inference-server received `/predict` and detected a fall
- mock_app published the MQTT confirmation
- fall-dashboard wrote the event to Postgres
- fall-dashboard fanned out an SSE event
- mock-patient-dashboard received the SSE and updated the UI

**This is the exact sequence your React Native app must reproduce.** Read [04_mobile_app_integration.md](04_mobile_app_integration.md) section by section while watching the mock_app behaviour — every JSON field, header, and MQTT topic is documented there.

---

## 5. Stop everything

```powershell
.\helm\mock-focus\teardown.ps1
```

This runs `helm uninstall` on both releases and deletes both namespaces. Persistent volumes (Postgres data, MinIO blobs) are wiped — your next `install.ps1` starts from a clean slate.

Verify nothing is left:

```powershell
kubectl get namespaces | findstr -E "mcs-fall-detection|mock-focus"
# (no output = clean)
```

If a namespace is stuck in `Terminating` for more than ~2 minutes, it has stuck finalizers. Force-clear with:

```powershell
kubectl get namespace <name> -o json | jq '.spec.finalizers=[]' | kubectl replace --raw "/api/v1/namespaces/<name>/finalize" -f -
```

(Substitute `<name>`. Requires `jq` — install with `winget install jqlang.jq`.)

---

## 6. Pointing your React Native app at the local backend

Once everything's up, your phone (or RN simulator) needs to reach two services running inside the cluster:

| What | Local URL (after port-forward) | What you do with it |
|------|--------------------------------|---------------------|
| inference-server | `http://<your-laptop-ip>:8001/predict` | HTTP POST with `X-API-Key` header — see [04_mobile_app_integration.md §2.2](04_mobile_app_integration.md) |
| MQTT broker | `mqtt://<your-laptop-ip>:1883` (TCP) **or** `ws://<your-laptop-ip>:9001` (WebSocket) | PUBLISH `fall/alert/<patient_id>` — see [04_mobile_app_integration.md §2.4](04_mobile_app_integration.md). React Native usually wants the WS port (9001). |

Two important quirks for phone-on-real-device dev:

1. **`localhost` on the phone ≠ your laptop.** Find your laptop's LAN IP (`ipconfig` → "IPv4 Address"), and use that in the phone app. Both your laptop and phone need to be on the same Wi-Fi network.
2. **`kubectl port-forward` binds to `127.0.0.1` only by default**, which won't accept connections from another device. Either:
   - Use `kubectl port-forward --address 0.0.0.0 svc/mqtt-broker 9001:9001` (allows LAN access — fine for dev), OR
   - Hit the cluster's NodePort directly if one is exposed.

For the inference-server you can also expose it via NodePort temporarily by editing `helm/fall-detection/templates/inference-server/service.yaml` — ask Hayate if you want to go that route.

The dev API key is in `.env` at the repo root (`INFERENCE_API_KEY`). Don't commit your local `.env`.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `install.ps1` hangs at "waiting for pods" | Docker Desktop K8s out of memory | Settings → Resources → bump RAM to ≥ 8 GB → Apply & Restart |
| Pod stuck in `ImagePullBackOff` | Forgot to run `build.ps1` (or built only one of the two) | Run both build scripts (section 3.1), then `kubectl delete pod <stuck-pod-name> -n <ns>` to retry |
| `test.ps1` test 1 fails (DNS) | CoreDNS not ready | `kubectl get pods -n kube-system` — if CoreDNS isn't Running, restart Docker Desktop K8s |
| `localhost:30090` returns "site cannot be reached" | NodePort conflict on the host | Use port-forward instead: `kubectl port-forward -n mock-focus svc/mock-patient-dashboard 30090:8090` |
| `mock_app` errors with `Connection refused` on 8001 or 1883 | port-forward isn't running, or pod restarted (port-forward dies on pod restart) | Check terminals A and B; restart them if dead |
| Patient cards render at :30090 but no alert when `mock_app` runs | MQTT connection silently failed (Windows IPv6 resolution) | Use `127.0.0.1`, not `localhost`, in `MQTT_BROKER_HOST` |
| `helm upgrade` succeeds but pod still runs old code | `pullPolicy: Never` + `latest` tag means K8s sees no spec change | `kubectl rollout restart deploy/<name> -n mcs-fall-detection` after rebuild |
| Tear down leaves "Terminating" namespaces | Stuck finalizers | See command in section 5 |

---

## 8. Where to look when you're stuck

| Question | Doc |
|----------|-----|
| **What exactly does my mobile app send and receive?** | [`04_mobile_app_integration.md`](04_mobile_app_integration.md) — read this end to end |
| What does each pod do? | [`03_fall_detection_system.md`](03_fall_detection_system.md) |
| What's the model doing? | [`02_fall_detection_algorithm.md`](02_fall_detection_algorithm.md) |
| How would FOCUS deploy this for real? | [`01_k8s.md`](01_k8s.md) (you don't need this for dev, but it explains the production picture) |
| What's the user flow on the patient side? | [`06_user_flow_patient.md`](06_user_flow_patient.md) |
| Reference Python implementation of the steps your app should do | `_6G_Integration_v2_mqtt/local_dev/mock_app/main.py` — single file, ~150 lines, easy to map to RN |

If something doesn't match what you're seeing, ping Hayate. The handover docs are kept in step with the code; if they drift it's a bug to fix.
