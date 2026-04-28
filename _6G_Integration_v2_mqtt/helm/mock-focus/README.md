# mock-focus — local two-namespace dry-run

This chart simulates the FOCUS namespace so we can validate cross-namespace
traffic in a real Kubernetes cluster **before** handing the real chart to FOCUS
DevOps. Implements [todo.md Step 12.5](../../../REFACTOR_DOCUS/todo.md).

## Why this exists

The local `docker-compose` setup is single-namespace. The production setup is
two-namespace (`mcs-fall-detection` ↔ FOCUS namespace). Several things only
exercise in real K8s:

- DNS resolution via `<service>.<namespace>.svc.cluster.local`
- Helm install + `helm upgrade`
- NetworkPolicy enforcement
- Per-pod PVCs (StatefulSets)
- Service discovery across namespaces

If any of those break, we want to find out on a laptop, not in FOCUS's cluster.

## What's inside

```
helm/mock-focus/
├── Chart.yaml
├── values.yaml                       ← image names, ports, FQDNs
├── templates/
│   ├── namespace.yaml                ← creates the mock-focus namespace
│   ├── mock-fhir.yaml                ← Deployment + Service (port 8003)
│   ├── mock-influxdb.yaml            ← StatefulSet + Service (port 8086)
│   └── mock-patient-dashboard.yaml   ← Deployment + Service (NodePort 30090)
├── dockerfiles/
│   ├── mock-fhir.Dockerfile          ← reuses local_dev/mock_focus/fhir_server.py
│   ├── mock-patient-dashboard.Dockerfile
│   ├── mock_patient_dashboard.py     ← FastAPI proxy + HTML server
│   └── dashboard.html                ← live patient dashboard with SSE flag
├── extras/
│   ├── deny-all-cross-namespace.yaml ← default-deny NetworkPolicy (test it breaks)
│   └── allow-mock-focus.yaml         ← allow rule (test it recovers)
├── build.ps1
├── install.ps1
├── test.ps1
└── teardown.ps1
```

The mock Patient Dashboard runs in `mock-focus` namespace and talks to
`fall-dashboard.mcs-fall-detection.svc.cluster.local:8002` — that's the
cross-namespace boundary the dry-run validates.

## Prerequisites

1. **Docker Desktop** with Kubernetes enabled (Settings → Kubernetes → Enable Kubernetes)
2. `kubectl` context set to `docker-desktop`
3. `helm` installed (`helm version` should print v3.x)
4. The real `fall-detection` chart already buildable — its images are needed by the dry-run

## Run order

From `_6G_Integration_v2_mqtt/` as the working directory:

```powershell
# 1. Build local images (mock-focus + the real chart's images)
.\helm\mock-focus\build.ps1
docker build -f inference_server/Dockerfile -t fd-inference-test:latest .
docker build -f fall_dashboard/Dockerfile  -t fd-dashboard-test:latest .
# (may need additional builds if your fall-detection chart references other images)

# 2. Install both charts
.\helm\mock-focus\install.ps1

# 3. Run the cross-namespace integration tests
.\helm\mock-focus\test.ps1

# 4. Manual SSE test:
#    Open http://localhost:30090/ in a browser. The mock Patient Dashboard
#    should show patient cards. Trigger a fall via mock_app, confirm the
#    patient card flashes red within 1-2 seconds.

# 5. (Optional) NetworkPolicy test
kubectl apply -f .\helm\mock-focus\extras\deny-all-cross-namespace.yaml
# → mock dashboard should now FAIL to reach /api/patients
kubectl apply -f .\helm\mock-focus\extras\allow-mock-focus.yaml
# → dashboard recovers within seconds

# 6. Tear down
.\helm\mock-focus\teardown.ps1
```

## Pass criteria for the dry-run

If `test.ps1` reports all green AND the manual SSE test in step 4 works
(red flag appears live in the browser), the chart is ready to hand to
FOCUS DevOps.

## What this is NOT

- Not for production. Will never ship.
- Not a replacement for the real FOCUS namespace; just enough to prove our
  side works correctly when called from another namespace.
- Not auth-aware. In the real production chart, NetworkPolicy + JWT will
  gate the same paths.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `helm install` hangs at "waiting for pods" | Docker Desktop K8s out of resources | Increase Docker Desktop memory to ≥ 8 GB |
| `ImagePullBackOff` on `fd-mock-fhir` | Image not built locally | `.\helm\mock-focus\build.ps1` |
| `ImagePullBackOff` on `inference-server` | Real chart image not built | Build it via the real chart's build script |
| NodePort 30090 unreachable | Docker Desktop NodePort range conflict | Use port-forward: `kubectl port-forward -n mock-focus svc/mock-patient-dashboard 30090:8090` |
| NetworkPolicy seems to do nothing | Docker Desktop K8s does not enforce NetworkPolicy by default | Switch to `kind` with Calico CNI for the NetworkPolicy step only |
