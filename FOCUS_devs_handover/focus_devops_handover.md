# Fall Detection — FOCUS DevOps Handover (2026-06-10)

## Context / What You Are Deploying

The fall detection system is split into two independently hosted layers:

| Side | Owner | Runtime | Services |
|------|-------|---------|----------|
| **MCS inference layer** | MCS (Mohammed) | Docker Compose on Hetzner | 8 services — inference, ML pipeline, monitoring |
| **FOCUS caregiver layer** | FOCUS DevOps (you) | K3s inside FOCUS network | **2 services** — MQTT broker + fall dashboard |

You only run 2 pods. Everything else (fall detection AI, model training, database, metrics) runs
on MCS's own Hetzner server and is not your concern.

**What changed from the previous handover:**
Previously the plan was for FOCUS to host all 10 services in k3s. That has changed.
FOCUS now hosts only the 2 caregiver-facing services. The inference server and all ML components
have moved to MCS's Hetzner machine.

---

## Your To-Do List

In order — some steps require information from Mohammed first.

- [ ] **1. Confirm your Traefik certResolver name** — you will need to put this in `values_production.yaml`. It is the name of the Let's Encrypt resolver configured in your cluster (e.g. `le`, `letsencrypt`). Check your existing Traefik configuration.
- [ ] **2. Decide your subdomains** — two subdomains are needed:
      - One for the MQTT broker (e.g. `mqtt.focus-hospital.de`) — the mobile app connects here via WSS
      - One for the fall dashboard (e.g. `fall.focus-hospital.de`) — the Flutter caregiver dashboard opens this
      Share both domains with Mohammed and Isa as soon as decided.
- [ ] **3. Wait for Mohammed** — he must push the fall-dashboard Docker image to `registry-smarko-health.de` and deliver `_6G_integration_v3_k3s/` to you before you can install.
- [ ] **4. One-time cluster setup** — create namespace + pull secret + Traefik MQTT entrypoint (see commands below, run once only)
- [ ] **5. Fill in `values_production.yaml`** — all `CHANGE_ME` fields (see section below)
- [ ] **6. Install Helm chart** — `helm install caregiver helm/ --namespace fall-dashboard`
- [ ] **7. Verify both pods are running** — see verification commands below
- [ ] **8. Share your domains and MQTT credentials with Isa** — she needs them for the mobile app config:
      - MQTT broker domain (for WSS connection)
      - MQTT username + password (if you enabled broker auth)
      - Fall-dashboard domain (for Flutter `/api/stream` SSE calls)
- [ ] **9. Share fall-dashboard domain with Mohammed** — he sets `FALL_DASHBOARD_URL` in his `.env` so his server-health dashboard can probe your service

---

## What Each Service Does

### `mosquitto` — MQTT broker

Eclipse Mosquitto running as a K3s pod. Two listeners:

| Port | Protocol | Reachable from | Purpose |
|------|----------|---------------|---------|
| 1883 | TCP | cluster-internal only | fall-dashboard subscribes here via cluster DNS (`mosquitto:1883`) |
| 9001 | WebSocket | external via Traefik WSS :443 | mobile app (Isa) publishes fall alerts here |

The mobile app **cannot** use raw TCP — React Native requires WebSocket. Port 1883 is
never exposed outside the cluster. All external MQTT traffic goes through Traefik as WSS on port 443.

### `fall-dashboard` — caregiver alert dashboard

FastAPI web service. Reachable externally via Traefik HTTPS on port 443.

What it does:
- Subscribes to MQTT broker (`fall/possible/#` and `fall/alert/#`) for live fall events
- Fans out live alerts to the Flutter caregiver browser via Server-Sent Events (SSE)
- Queries FOCUS InfluxDB for per-patient fall history (the history tab)
- Stores the patient list in SQLite on a PVC — patients are added/deleted from the UI, no values.yaml edit needed

What it does NOT do:
- No inference — it only displays alerts that the mobile app already detected and confirmed
- No Postgres — all data comes from InfluxDB (fall history) and MQTT (live alerts)
- No connection to the MCS Hetzner server at all

---

## Step-by-Step Deployment

### Prerequisites

- K3s cluster with Traefik already running
- `kubectl` and `helm` installed
- Registry pull credentials for `registry-smarko-health.de` (obtain from Mohammed)
- `_6G_integration_v3_k3s/` folder delivered by Mohammed

### 1. One-time cluster setup (run once, never again)

```bash
# Add the MQTT WebSocket entrypoint to Traefik (required for WSS on port 443)
kubectl apply -f helm/extras/traefik-mqtts-entrypoint.yaml

# Create namespace
kubectl create namespace fall-dashboard

# Create registry pull secret (credentials from Mohammed)
kubectl create secret docker-registry mcs-labs \
    --docker-server=registry-smarko-health.de \
    --docker-username=<username from Mohammed> \
    --docker-password=<password from Mohammed> \
    --namespace fall-dashboard
```

### 2. Fill in values_production.yaml

Open `helm/values_production.yaml` and replace every `CHANGE_ME`.

**Registry (fill in after Mohammed confirms image is pushed):**

| Key | Value |
|-----|-------|
| `imagePullSecret` | `mcs-labs` |
| `fallDashboard.image` | `registry-smarko-health.de/fall-detection/fall-dashboard:latest` |
| `fallDashboard.imagePullPolicy` | `Always` |

**NodePorts — blank both (Traefik handles everything in production):**

| Key | Value |
|-----|-------|
| `mosquitto.wsNodePort` | `""` (blank) |
| `fallDashboard.httpNodePort` | `""` (blank) |

**Traefik ingress for MQTT broker:**

| Key | Value |
|-----|-------|
| `mosquitto.ingress.host` | your MQTT subdomain (e.g. `mqtt.focus-hospital.de`) |
| `mosquitto.ingress.certResolver` | your Traefik cert resolver name (e.g. `le`) |

**Traefik ingress for fall-dashboard:**

| Key | Value |
|-----|-------|
| `fallDashboard.ingress.host` | your fall-dashboard subdomain (e.g. `fall.focus-hospital.de`) |
| `fallDashboard.ingress.certResolver` | same resolver name as above |

**InfluxDB — your FOCUS InfluxDB instance:**

| Key | Value |
|-----|-------|
| `fallDashboard.influxdb.url` | your InfluxDB URL |
| `fallDashboard.influxdb.org` | your InfluxDB organisation name |
| `fallDashboard.influxdb.bucket` | bucket name where raw sensor data is stored |
| `fallDashboard.influxdb.fallEventsBucket` | bucket where `fall_events` are written (often the same bucket) |
| `fallDashboard.influxdb.token` | InfluxDB token with read access to both buckets |

**MQTT broker auth (optional — leave blank to allow anonymous connections):**

| Key | Value |
|-----|-------|
| `fallDashboard.mqtt.username` | broker username (blank = anonymous allowed) |
| `fallDashboard.mqtt.password` | broker password |

> If you set MQTT credentials here, you must share the same username and password with Isa.
> The mobile app must authenticate with the same credentials when connecting to the broker.
> A mismatch causes silent connection failure — fall alerts will not appear on the dashboard.

**Do NOT change these values:**

| Key | Value | Why |
|-----|-------|-----|
| `namespace` | `fall-dashboard` | Standard — chart uses this name |
| `mosquitto.port` | `1883` | Internal TCP, cluster-only |
| `mosquitto.wsPort` | `9001` | Internal WS, Traefik forwards to this |
| `fallDashboard.replicas` | `1` | Must stay 1 — SSE fan-out is in-process; multiple replicas break live alerts |
| `fallDashboard.port` | `8002` | Internal pod port |
| `fallDashboard.mqtt.alertTopic` | `fall/alert` | Must match MCS side |
| `fallDashboard.mqtt.possibleTopic` | `fall/possible` | Must match MCS side |

Full table with all keys and descriptions: [`config_checklist_focus_devops.md`](config_checklist_focus_devops.md).

### 3. Install the Helm chart

Run from `_6G_integration_v3_k3s/` as working directory:

```bash
# First install
helm install caregiver helm/ \
  --namespace fall-dashboard \
  --values helm/values_production.yaml

# Future updates (e.g. values change or image update)
helm upgrade caregiver helm/ \
  --namespace fall-dashboard \
  --values helm/values_production.yaml
```

> `helm upgrade` is required whenever `values_production.yaml` changes — K3s does not
> pick up values file changes automatically.

### 4. Verify

```bash
# Both pods should be Running
kubectl get pods -n fall-dashboard
# Expected:
#   mosquitto-xxxx        1/1   Running
#   fall-dashboard-xxxx   1/1   Running

# Check fall-dashboard is responding (from inside the cluster)
kubectl exec -n fall-dashboard deploy/fall-dashboard -- \
  wget -qO- http://localhost:8002/api/patients

# Check logs if a pod is not starting
kubectl logs -n fall-dashboard -l app=mosquitto
kubectl logs -n fall-dashboard -l app=fall-dashboard
```

### 5. Verify Traefik routing

After both pods are running, test the public endpoints:

```bash
# Fall dashboard (HTTPS)
curl https://fall.focus-hospital.de/api/patients

# MQTT (WebSocket — test TCP connectivity first)
curl -v --http1.1 https://mqtt.focus-hospital.de
# Then test from Isa's mobile app
```

---

## Cross-Party Agreements — What to Share and With Whom

### You → Isa (mobile app developer)

Isa needs these to configure the mobile app. Share as soon as your deployment is live:

| What | Where you set it | What Isa does with it |
|------|-----------------|----------------------|
| MQTT broker domain | `mosquitto.ingress.host` in `values_production.yaml` | Mobile app connects as `wss://<domain>:443` to publish fall alerts |
| MQTT username | `fallDashboard.mqtt.username` | Mobile app authenticates with broker |
| MQTT password | `fallDashboard.mqtt.password` | Mobile app authenticates with broker |
| InfluxDB write credentials | your FOCUS InfluxDB (URL, org, bucket, token) | Mobile app writes `fall_events` point after each patient confirmation popup |
| Fall-dashboard domain | `fallDashboard.ingress.host` | Flutter dashboard uses this to open `/api/stream` SSE for live alerts |

### You → Mohammed (MCS)

| What | Where you set it | What Mohammed does with it |
|------|-----------------|--------------------------|
| Fall-dashboard domain | `fallDashboard.ingress.host` | Sets `FALL_DASHBOARD_URL` in MCS `.env` so server-health can probe your service |
| MQTT broker domain | `mosquitto.ingress.host` | Sets `MQTT_BROKER_HOST` in MCS `.env` so server-health can probe the broker |

### Mohammed → You (before you can start)

| What | What you need it for |
|------|---------------------|
| Registry credentials for `registry-smarko-health.de` | `kubectl create secret docker-registry mcs-labs ...` (Step 1) |
| Confirmation that fall-dashboard image is pushed | Before running `helm install` |

---

## What You Do NOT Need to Worry About

- **Inference server** — runs on MCS Hetzner, not in your cluster
- **Postgres database** — runs on MCS Hetzner; your 2 pods have no database dependency
- **MLflow / MinIO / Grafana / Prometheus** — all MCS-side, not in your cluster
- **Patient IDs in `values.yaml`** — patients are now managed dynamically from the fall-dashboard UI ("+Add Patient" button). No need to edit `values.yaml` and restart the pod to add patients.
- **FHIR output** — opted out; not implemented

---

## Key Files for Reference

| File | What it contains |
|------|-----------------|
| [`_6G_integration_v3_k3s/README.md`](../_6G_integration_v3_k3s/README.md) | Quick-start, structure, connection addresses |
| [`_6G_integration_v3_k3s/helm/values_production.yaml`](../_6G_integration_v3_k3s/helm/values_production.yaml) | The file you fill in — all `CHANGE_ME` entries |
| [`config_checklist_focus_devops.md`](config_checklist_focus_devops.md) | Full table of every config key for the caregiver layer + cross-party agreements |
| [`mohammed_handover.md`](mohammed_handover.md) | Mohammed's tasks — useful to understand what he delivers to you |
