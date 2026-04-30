# mock-focus — Local Two-Namespace Dry Run

This Helm chart simulates the FOCUS namespace on a local Docker Desktop
Kubernetes cluster, so we can validate cross-namespace communication for
the real `fall-detection` chart **before** handing it off to FOCUS DevOps.

Implements [todo.md Step 12.5](../../REFACTOR_DOCUS/todo.md). Pairs with
[Tech_integrator.md](../../../handover_docs/Tech_integrator.md).

---

## 1. Why this exists

Docker Compose runs everything in a single namespace. Production runs across
two namespaces (`mcs-fall-detection` ↔ FOCUS namespace). A handful of things
only break when you actually have two namespaces:

- DNS resolution via `<service>.<namespace>.svc.cluster.local`
- `helm install` ordering / `helm upgrade` rollouts
- NetworkPolicy enforcement
- StatefulSets with per-pod PVCs (Postgres, MinIO)
- Cross-namespace service discovery (Patient Dashboard → fall-dashboard)
- SSE streaming through a different namespace's pod network

We want all of those to fail on a laptop, not in FOCUS's cluster on the day
of go-live.

---

## 2. What this chart contains

```
helm/mock-focus/
├── Chart.yaml
├── values.yaml                       ← image tags, ports, FQDN of fall-dashboard
├── templates/
│   ├── namespace.yaml                ← creates the mock-focus namespace
│   ├── mock-fhir.yaml                ← Deployment + Service (port 8003)
│   ├── mock-influxdb.yaml            ← StatefulSet + Service (port 8086)
│   └── mock-patient-dashboard.yaml   ← Deployment + Service (NodePort 30090)
├── dockerfiles/
│   ├── mock-fhir.Dockerfile          ← reuses local_dev/mock_focus/fhir_server.py
│   ├── mock-patient-dashboard.Dockerfile
│   ├── mock_patient_dashboard.py     ← FastAPI: serves HTML + proxies to fall_dashboard
│   └── dashboard.html                ← live patient dashboard (red flag on SSE alert)
├── extras/
│   ├── deny-all-cross-namespace.yaml ← default-deny NetworkPolicy (test it breaks)
│   └── allow-mock-focus.yaml         ← allow rule (test it recovers)
├── build.ps1                         ← builds the two mock images
├── install.ps1                       ← installs both charts + waits for ready
├── test.ps1                          ← four cross-namespace integration tests
└── teardown.ps1                      ← uninstall + delete both namespaces
```

---

## 3. The cross-namespace topology this chart sets up

```
┌────────────────────────────────────┐    ┌────────────────────────────────────────┐
│  mock-focus  (this chart)          │    │  mcs-fall-detection  (the real chart)  │
│                                    │    │                                        │
│  mock-fhir-server  :8003 ◄─────────┼────┤  inference-server :8001                │
│   (HTTP /health, /fhir/Patient)    │    │   (FHIR push if FHIR_SERVER_URL set)   │
│                                    │    │                                        │
│  mock-influxdb  :8086              │    │  fall-dashboard  :8002                 │
│   (FOCUS-side biosignals — never   │    │                                        │
│    read by our code in production) │    │  mqtt-broker  :1883 / :9001 ws         │
│                                    │    │                                        │
│  mock-patient-dashboard ───────────┼───►│  fall-dashboard /api/patients          │
│   :8090 (NodePort 30090)           │SSE │                /api/falls              │
│   ▲ HTTP+SSE proxy                 │    │                /api/stream             │
│   │                                │    │                                        │
└───┼────────────────────────────────┘    │  postgres, mlflow, prometheus,         │
    │                                     │  grafana, minio  (all in same ns)      │
    │ browser at localhost:30090          └────────────────────────────────────────┘
    │                                                       ▲
[your browser]                                              │ HTTP POST /predict
                                                            │ MQTT PUBLISH fall/alert/<pid>
                                          [mock_app on your laptop, port-forwarded]
```

The bold arrows are the connections the dry run validates. The ones to
prove are: mock-patient-dashboard → fall-dashboard (HTTP + SSE), and
inference-server → mock-fhir-server (the optional FHIR push direction).

---

## 4. Prerequisites

Before anything below works, confirm:

| Requirement | Verify with |
|-------------|-------------|
| Docker Desktop with Kubernetes enabled | Settings → Kubernetes → Enable Kubernetes → Apply & Restart |
| `kubectl` context is `docker-desktop` | `kubectl config current-context` |
| `helm` v3 installed | `helm version` |
| Docker Desktop has ≥ 8 GB RAM | Settings → Resources |
| Working dir is `_6G_Integration_v2_mqtt/` | `pwd` ends with `_6G_Integration_v2_mqtt` |

If `kubectl config current-context` is wrong, switch:

```powershell
kubectl config use-context docker-desktop
```

---

## 5. Run order — full happy path

All commands are run from `_6G_Integration_v2_mqtt/` as the working
directory. Nothing here pushes to a registry; Docker Desktop K8s shares
the local Docker daemon's image cache, which is why
`imagePullPolicy: IfNotPresent` works without a real registry.

### Step 5.1 — Build images for both charts

Each chart owns its own `build.ps1`. Run both:

```powershell
# Two mock-focus images (fd-mock-fhir, fd-mock-patient-dashboard) — no registry prefix
.\helm\mock-focus\build.ps1

# Four fall-detection images (inference-server, fall-dashboard, ml-dashboard, server-health)
# tagged with registry.example.com/ prefix because the chart's values.yaml
# references that prefix and Docker Desktop K8s with pullPolicy: Never needs
# the exact name.
.\helm\fall-detection\build.ps1
```

Verify all six are present:

```powershell
docker images | findstr -E "fd-mock|registry.example.com"
```

### Step 5.2 — Install both charts

The two charts have separate install scripts. Install the real chart first
so cross-namespace targets (`fall-dashboard.mcs-fall-detection.svc.cluster.local`)
exist when mock-focus comes up:

```powershell
.\helm\fall-detection\install.ps1
.\helm\mock-focus\install.ps1
```

Each runs `helm upgrade --install` with `--wait --timeout 5m`, so each
returns only once every pod in its namespace is Running and Ready. If a
pod fails the readiness probe within 5 minutes, the script bails — see
[Troubleshooting](#9-troubleshooting).

When both complete you should see all 10 pods in `mcs-fall-detection`
plus 3 pods in `mock-focus`, all Running (or `Completed` for the Alembic
+ MinIO bucket-creation Jobs).

### Step 5.3 — Run the cross-namespace integration tests

```powershell
.\helm\mock-focus\test.ps1
```

Four tests run sequentially. Each tests one specific
cross-namespace concern:

| # | What it validates | Why it matters |
|---|--------------------|----------------|
| 1 | DNS — `inference-server` (mcs-fall-detection) can resolve `mock-fhir-server.mock-focus.svc.cluster.local` | Cross-namespace DNS is what makes `FHIR_SERVER_URL=http://...mock-focus.svc.cluster.local:8003` work. If this fails, DNS or CoreDNS config is broken. |
| 2 | HTTP — `mock-patient-dashboard` (mock-focus) can `GET /api/patients` on `fall-dashboard` (mcs-fall-detection) | This is the live path Isa's real Patient Dashboard will use. Validates Service routing across namespaces. |
| 3 | HTTP — `inference-server` (mcs-fall-detection) can `GET /health` on `mock-fhir-server` (mock-focus) | The reverse direction. Validates the (optional) FHIR-push path that fires when `FHIR_SERVER_URL` is configured. |
| 4 | InfluxDB health | Sanity-check that the StatefulSet came up. InfluxDB itself is not part of our inference path — it's there because biosignals live in FOCUS in production. |

If all four pass, cross-namespace plumbing is fine. The script exits 0 on
success, exit 1 on any failure.

### Step 5.4 — Manual SSE end-to-end test (the one that matters most)

The integration tests above do not exercise SSE. SSE has its own failure
modes (connection upgrade, keepalives, proxy buffering) that only surface
when you actually drive an event through the system.

**Open the mock dashboard:** http://localhost:30090/

You should see patient cards rendered — that data came from
`fall-dashboard` two namespaces away via the proxy in
`mock_patient_dashboard.py`. The page also opens an SSE connection to
`/proxy/stream`, which streams `fall-dashboard`'s `/api/stream` through
this server. No alerts yet; flip to the next sub-step to fire one.

**Drive a fall through the real chart's inference-server:**

```powershell
# Terminal A — port-forward the inference server and the MQTT broker
# so mock_app on your laptop can reach the in-cluster services
kubectl port-forward -n mcs-fall-detection svc/inference-server 8001:8001
# in another window:
kubectl port-forward -n mcs-fall-detection svc/mqtt-broker 1883:1883

# Terminal B — run mock_app pointing at localhost
$env:INFERENCE_API_URL = "http://localhost:8001"
$env:MQTT_BROKER_HOST  = "localhost"
python -m local_dev.mock_app.main
```

Within ~10 seconds (the patient confirmation window) the mock dashboard
at `localhost:30090` should flash a red flag for the affected patient.
That confirms the entire cross-namespace path:

```
mock_app  →  inference-server (mcs-fall-detection)  →  /predict response
mock_app  →  MQTT broker      (mcs-fall-detection)  →  fall/alert/<pid>
                                                       ↓
                                          fall-dashboard (mcs-fall-detection)
                                          writes fall_history + emits SSE
                                                       ↓
                              mock-patient-dashboard (mock-focus)  ← cross-namespace SSE
                                                       ↓
                                     your browser at localhost:30090
```

If the cards render but no alert ever appears, SSE is the most likely
suspect — see [Troubleshooting](#9-troubleshooting).

### Step 5.5 — NetworkPolicy test (optional but recommended)

This proves what FOCUS DevOps will see if their cluster enforces
default-deny ingress.

```powershell
# Lock down the mcs-fall-detection namespace — only intra-namespace traffic allowed
kubectl apply -f .\helm\mock-focus\extras\deny-all-cross-namespace.yaml
```

Refresh `http://localhost:30090/` — patient cards should fail to load
(the proxy call to `fall-dashboard` is now denied). SSE will also drop.
This is the failure mode you want to discover here, not on FOCUS's
cluster.

```powershell
# Add the explicit allow rule for mock-focus
kubectl apply -f .\helm\mock-focus\extras\allow-mock-focus.yaml
```

The browser should recover within a few seconds.

> **Caveat:** Docker Desktop's default CNI does **not** enforce
> NetworkPolicy. The `apply` succeeds but the rule is a no-op, so your
> dashboard will keep working even after `deny-all-cross-namespace.yaml`
> is applied. To actually test enforcement, switch to a `kind` cluster
> with Calico for this step only. For most pre-handover purposes,
> applying the YAML and confirming `kubectl describe networkpolicy`
> shows the right selectors is enough.

### Step 5.6 — Tear down

Each chart owns its own teardown. Run both:

```powershell
.\helm\mock-focus\teardown.ps1
.\helm\fall-detection\teardown.ps1
```

Each runs `helm uninstall` and deletes its own namespace. PVCs are cleaned
up by namespace deletion. Verify:

```powershell
kubectl get namespaces | findstr -E "mcs-fall-detection|mock-focus"
# (no output = clean)
```

---

## 6. What each component is doing

### `mock-fhir-server` (Deployment, port 8003)

Runs `local_dev/mock_focus/fhir_server.py` — the same FastAPI mock used
in local Docker Compose dev. Serves FHIR R4 Patient + Observation
resources for two synthetic patients. In production, this gets replaced
by the real FOCUS FHIR server.

The dry run only uses it to prove the **inference-server → FHIR push
direction works across namespaces**. If the push path is never enabled
(`fhirServerUrl: ""` in the real chart), this service still gets pinged
by test 3 of `test.ps1` to confirm the route is open.

### `mock-influxdb` (StatefulSet, port 8086)

Vanilla `influxdb:2.7` with `initOrg=focus`, `initBucket=biosignals`.
Validates two things: (a) StatefulSets with per-pod PVCs work on Docker
Desktop K8s, and (b) the dependency you'd otherwise have on FOCUS-hosted
InfluxDB doesn't actually need to be wired into our code. **Our
inference path never reads InfluxDB**; this pod exists purely so the
mock-focus namespace looks like the real thing.

### `mock-patient-dashboard` (Deployment, NodePort 30090)

The most important component. It does two jobs:

1. **Serves the dashboard HTML** at `http://localhost:30090/` (NodePort
   exposure so your laptop browser can reach it).
2. **Proxies cross-namespace traffic** server-side. The browser calls
   `/proxy/patients`, `/proxy/falls`, and `/proxy/stream` on the
   dashboard pod, which then calls the in-cluster
   `fall-dashboard.mcs-fall-detection.svc.cluster.local:8002` — exactly
   the kind of cross-namespace call Isa's real dashboard will make.

The proxy pattern matters: it means the *pod* makes the cross-namespace
call (which is what NetworkPolicy gates), not the browser. This mirrors
what a server-side rendered Patient Dashboard would look like.

If Isa's real Patient Dashboard turns out to be a browser SPA, the
cross-namespace path differs: the browser would talk to the public
Ingress URL, not the internal Service DNS. That changes which auth +
NetworkPolicy concerns apply. Confirm with FOCUS DevOps which model
they're using before relying on the SPA assumption.

---

## 7. What this chart is NOT

- Not for production. Will never ship.
- Not a faithful FHIR server. Two synthetic patients, no resource
  validation. FOCUS DevOps replaces it entirely with their real FHIR.
- Not auth-aware. No JWT, no MQTT auth, no HTTPS. Production handles all
  of these via FOCUS infrastructure + the real chart's secrets.
- Not a replacement for the real test suite. It exercises wiring; unit
  tests still belong with the code.

---

## 8. Pass criteria for the whole dry run

The chart can be handed to FOCUS DevOps when **all of these are true**:

- [ ] `install.ps1` completes within 5 minutes with all pods Running.
- [ ] `test.ps1` reports 4/4 PASS.
- [ ] Manual SSE test in 5.4 shows a red flag in the browser within
      ~10 s of triggering a fall via mock_app.
- [ ] (Optional but recommended) NetworkPolicy test in 5.5 demonstrates
      the deny → allow flip in a Calico-enabled cluster.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `install.ps1` hangs at "waiting for pods" | Docker Desktop K8s out of memory | Settings → Resources → bump RAM to ≥ 8 GB and restart |
| `ImagePullBackOff` on `fd-mock-fhir` or `fd-mock-patient-dashboard` | Mock images not built | Re-run `.\helm\mock-focus\build.ps1` |
| `ImagePullBackOff` on `inference-server` or `fall-dashboard` | Real-chart images not built locally | Run the two `docker build` commands in step 5.1 |
| Test 1 fails (DNS) | CoreDNS not running | `kubectl get pods -n kube-system` — if CoreDNS is down, restart Docker Desktop K8s |
| Test 2 fails (`/api/patients` not reachable) | `fall-dashboard` not Ready, or wrong FQDN in `values.yaml → mockPatientDashboard.fallDashboardUrl` | `kubectl logs -n mcs-fall-detection deploy/fall-dashboard`, then check the FQDN matches the actual Service name + namespace |
| Test 3 fails (mock-fhir not reachable from inference-server) | Mock FHIR pod not Ready | Check `kubectl get pods -n mock-focus`; `test.ps1` already uses `python3 urllib` (no wget/curl needed) |
| NodePort 30090 unreachable in browser | Conflict on the host port | Use port-forward: `kubectl port-forward -n mock-focus svc/mock-patient-dashboard 30090:8090` |
| Patient cards render but SSE alert never appears | SSE connection died, or `proxy_stream` can't reach `/api/stream`, or mock_app not actually publishing MQTT | Open browser devtools → Network → look for `proxy/stream` (status `200 (pending)`); `kubectl logs -n mcs-fall-detection deploy/fall-dashboard` should show the MQTT alert arriving |
| `mock_app` errors with connection refused on 8001 or 1883 | `kubectl port-forward` not running, or pod restarted (port-forward dies on pod restart) | Restart the port-forwards in Terminal A |
| `kubectl apply` of NetworkPolicy seems to do nothing | Docker Desktop CNI does not enforce NetworkPolicy | Switch to `kind` with Calico for this step only — see Step 5.5 caveat |
| Tear down leaves "Terminating" namespaces | Stuck finalizers | `kubectl get namespace mcs-fall-detection -o json | jq '.spec.finalizers=[]' | kubectl replace --raw "/api/v1/namespaces/mcs-fall-detection/finalize" -f -` |

---

## 10. Related docs

| Doc | Read this when… |
|-----|-----------------|
| [Tech_integrator.md](../../../handover_docs/Tech_integrator.md) | You need the FOCUS DevOps view of the same architecture |
| [REFACTOR_DOCUS/deployment_architecture.md](../../REFACTOR_DOCUS/deployment_architecture.md) | You need the full data-source table and Postgres schema |
| [REFACTOR_DOCUS/helm_guide.md](../../REFACTOR_DOCUS/helm_guide.md) | You're touching the real chart's templates, not just the dry run |
| [REFACTOR_DOCUS/todo.md](../../REFACTOR_DOCUS/todo.md) Step 12.5 | This chart's design rationale and acceptance criteria |
