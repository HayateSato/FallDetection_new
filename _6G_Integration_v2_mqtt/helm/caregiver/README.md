# helm/caregiver — FOCUS Caregiver Layer (k3s)

Helm chart for the FOCUS-hosted caregiver services: MQTT broker (mosquitto) and
fall-dashboard. Targets FOCUS's existing k3s cluster with Traefik.

No Postgres. No mock-app (dev-only — replaced by the real Isa mobile app in production).

## Services deployed

| Resource | Type | Purpose |
|---|---|---|
| mosquitto | Deployment + PVC | MQTT broker — receives fall alerts from mobile app |
| mosquitto | ClusterIP Service | Internal DNS: `mosquitto.fall-dashboard.svc.cluster.local:1883` |
| mosquitto | IngressRouteTCP | Exposes port 8883 externally via Traefik TCP (mobile app connects here) |
| fall-dashboard | Deployment + PVC | SSE feed + /api/falls + /api/patients. SQLite patient store on PVC. |
| fall-dashboard | ClusterIP Service | Internal |
| fall-dashboard | IngressRoute | Flutter dashboard reaches `/api/stream`, `/api/falls`, `/api/patients` on port 443 |

## Prerequisites

### 1. One-time cluster setup (FOCUS DevOps)

Apply the Traefik TCP entrypoint so port 8883 is available for MQTT:

```powershell
kubectl apply -f helm/caregiver/extras/traefik-mqtts-entrypoint.yaml
```

This patches the existing Traefik in `kube-system`. Verify it restarted and picked up
the new entrypoint:

```powershell
kubectl rollout status deployment/traefik -n kube-system
kubectl logs -n kube-system -l app.kubernetes.io/name=traefik | Select-String "8883"
```

### 2. Create the registry pull secret

```powershell
kubectl create namespace fall-dashboard
kubectl create secret docker-registry mcs-labs `
    --docker-server=registry-smarko-health.de `
    --docker-username=<registry-user> `
    --docker-password=<registry-pass> `
    --namespace fall-dashboard
```

### 3. Fill in values.yaml

Open `helm/caregiver/values.yaml` and replace every `CHANGE_ME`:

| Field | What to set |
|---|---|
| `fallDashboard.patientIds` | Comma-separated patient IDs enrolled in the system |
| `fallDashboard.influxdb.url` | InfluxDB URL, e.g. `https://influxdb.xxx.e-healthservice.de` |
| `fallDashboard.influxdb.org` | InfluxDB organisation |
| `fallDashboard.influxdb.bucket` | Bucket for raw ACC data |
| `fallDashboard.influxdb.fallEventsBucket` | Bucket for `fall_events` measurement |
| `fallDashboard.influxdb.token` | InfluxDB auth token (stored in a k8s Secret) |
| `fallDashboard.ingress.host` | Domain where Flutter dashboard reaches fall-dashboard |

## Deployment

### Build and push image

Run from `_6G_Integration_v2_mqtt/` as working directory:

```powershell
.\helm\caregiver\build.ps1
```

### Install or upgrade

```powershell
.\helm\caregiver\install.ps1
```

### Smoke test

```powershell
.\helm\caregiver\test.ps1
# Expected: 7/7 PASS
```

### Teardown (WARNING: deletes patient data PVC)

```powershell
.\helm\caregiver\teardown.ps1
```

## Namespace

Kubernetes does not allow underscores in namespace names. The namespace is `fall-dashboard`
(hyphen), not `fall_dashboard` (underscore).

## MQTT topics

| Topic | Publisher | When |
|---|---|---|
| `fall/possible/<patient_id>` | Mobile app (Isa) | Immediately on fall detection, before patient confirms |
| `fall/alert/<patient_id>` | Mobile app (Isa) | After patient confirmation or 10s timeout |

fall-dashboard subscribes to both `fall/possible/#` and `fall/alert/#`.

## Connection addresses after deployment

| Client | Address |
|---|---|
| Mobile app (MQTT) | `mqtts://<FOCUS-server-IP>:8883` |
| Flutter dashboard (HTTP) | `https://<fallDashboard.ingress.host>/api/stream` etc. |
| fall-dashboard -> mosquitto (internal) | `mosquitto:1883` (cluster DNS) |
| fall-dashboard -> InfluxDB (external) | `https://influxdb.xxx.e-healthservice.de` |

## Files

```
helm/caregiver/
  Chart.yaml
  values.yaml                             <- fill in CHANGE_ME values here
  templates/
    _helpers.tpl
    mosquitto-configmap.yaml
    mosquitto-pvc.yaml
    mosquitto-deployment.yaml
    mosquitto-service.yaml
    mosquitto-ingressroutetcp.yaml        <- TCP route for port 8883
    fall-dashboard-secret.yaml            <- InfluxDB token, MQTT credentials
    fall-dashboard-configmap.yaml         <- non-secret env vars
    fall-dashboard-pvc.yaml               <- SQLite patient store persistence
    fall-dashboard-deployment.yaml
    fall-dashboard-service.yaml
    fall-dashboard-ingressroute.yaml      <- HTTP route for Flutter dashboard
  extras/
    traefik-mqtts-entrypoint.yaml         <- apply once before chart install
  build.ps1                               <- build + push fall-dashboard image
  install.ps1                             <- helm upgrade --install
  test.ps1                                <- 7-probe smoke test
  teardown.ps1                            <- uninstall + delete namespace
```
