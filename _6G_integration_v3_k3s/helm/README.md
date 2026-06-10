# helm/ — FOCUS Caregiver Layer (k3s)

Helm chart for the FOCUS-hosted caregiver services: MQTT broker (mosquitto) and
fall-dashboard. Targets FOCUS's existing k3s cluster with Traefik.

No Postgres. No mock-app (dev-only -- replaced by the real Isa mobile app in production).

## Services deployed

| Resource | Type | Purpose |
|---|---|---|
| mosquitto | Deployment + PVC | MQTT broker -- receives fall alerts from mobile app |
| mosquitto | ClusterIP Service | Internal: `mosquitto.fall-dashboard.svc.cluster.local:1883` (TCP) + `:9001` (WS) |
| mosquitto | IngressRoute | WSS on port 443 via Traefik -- mobile app connects here |
| fall-dashboard | Deployment + PVC | SSE feed + /api/falls + /api/patients. SQLite patient store on PVC. |
| fall-dashboard | ClusterIP Service | Internal: port 8002 |
| fall-dashboard | IngressRoute | HTTPS on port 443 via Traefik -- caregiver browser + Flutter dashboard |

## Prerequisites

### 1. One-time cluster setup (FOCUS DevOps)

```bash
# Create namespace + registry pull secret
kubectl create namespace fall-dashboard
kubectl create secret docker-registry mcs-labs \
    --docker-server=registry-smarko-health.de \
    --docker-username=<registry-user> \
    --docker-password=<registry-pass> \
    --namespace fall-dashboard
```

### 2. Fill in values_production.yaml

Open `helm/values_production.yaml` and replace every `CHANGE_ME`:

| Field | What to set |
|---|---|
| `mosquitto.ingress.host` | Subdomain for the MQTT broker, e.g. `mqtt.focus-hospital.de` |
| `fallDashboard.ingress.host` | Subdomain for the fall dashboard, e.g. `fall.focus-hospital.de` |
| `fallDashboard.influxdb.url` | InfluxDB URL |
| `fallDashboard.influxdb.org` | InfluxDB organisation |
| `fallDashboard.influxdb.fallEventsBucket` | Bucket for `fall_events` measurement |
| `fallDashboard.influxdb.token` | InfluxDB auth token (stored in a k8s Secret) |
| `fallDashboard.mqtt.username` | MQTT username (shared with Isa for the mobile app) |
| `fallDashboard.mqtt.password` | MQTT password (shared with Isa for the mobile app) |

Patients are added and deleted from the dashboard UI at runtime -- no `values_production.yaml`
edit is needed when enrolling a new patient.

## Deployment

Run from `_6G_integration_v3_k3s/` as working directory.

### Build and push image (MCS runs this)

```bash
bash helm/build.sh <tag>
```

### Install or upgrade

```bash
bash helm/install.sh
```

### Smoke test

```bash
bash helm/test.sh
# Expected: 7/7 PASS
```

### Teardown (WARNING: deletes patient data PVC)

```bash
bash helm/teardown.sh
```

PowerShell equivalents (`.ps1`) also exist for all four scripts.

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

| Client | Protocol | Address |
|---|---|---|
| Mobile app (MQTT) | MQTT over WSS | `wss://mqtt.<FOCUS-domain>` (port 443) |
| Caregiver browser / Flutter dashboard | HTTPS | `https://fall.<FOCUS-domain>` |
| fall-dashboard -> mosquitto (internal) | MQTT TCP | `mosquitto:1883` (cluster DNS) |
| fall-dashboard -> InfluxDB | HTTPS | configured via `INFLUXDB_URL` in values |

## Files

```
helm/
  Chart.yaml
  values.yaml                             local two-laptop testing values (NodePorts)
  values_production.yaml                  production values template -- fill in CHANGE_ME
  build.ps1 / build.sh                    build + push fall-dashboard image to registry
  install.ps1 / install.sh               helm upgrade --install
  test.ps1 / test.sh                      7-probe smoke test
  teardown.ps1 / teardown.sh             uninstall + delete namespace
  templates/
    _helpers.tpl
    mosquitto-configmap.yaml
    mosquitto-pvc.yaml
    mosquitto-deployment.yaml
    mosquitto-service.yaml
    mosquitto-ingressroute.yaml           WSS route for mobile app (port 443)
    fall-dashboard-secret.yaml            InfluxDB token, MQTT credentials
    fall-dashboard-configmap.yaml         non-secret env vars
    fall-dashboard-pvc.yaml               SQLite patient store persistence
    fall-dashboard-deployment.yaml
    fall-dashboard-service.yaml
    fall-dashboard-ingressroute.yaml      HTTPS route for caregiver browser

```
