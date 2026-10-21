# Fall Detection — FOCUS DevOps Handover

## 1. Context / What FOCUS will be deploying

The fall detection system is split into two independently hosted layers:

| Side | Owner | Runtime | Services |
|------|-------|---------|----------|
| **FOCUS caregiver layer** | FOCUS | K3s | **2 services** — MQTT broker + fall dashboard |
| **MCS inference layer** | MCS  | Docker | inference API, monitoring, etc. |

Previously the plan was for FOCUS to host all 10 services in k3s. That has changed.
FOCUS now hosts only the 2 caregiver-facing services. Everything else runs on MCS's server. 


## 2. To-Do List for FOCUS side

- [ ] **1. Create two subdomains** — two subdomains are needed: Share both domains with Mohammed and Isa as soon as decided.
  1) subdomain for the MQTT broker (e.g. `mqtt.focus-hospital.de`) — the mobile app needs to connect here via WSS
  2) subdomain for the fall dashboard (e.g. `fall.focus-hospital.de`) — caregiver needs to open this to check real time fall alert or fall history.          
- [ ] **2. Pull docker images** — fall-dashboard Docker image is already pushed to `registry.smarko-health.de`.
- [ ] **3. One-time cluster setup** — create namespace + pull secret + Traefik MQTT entrypoint (see commands below, run once only)
- [ ] **5. Fill in `values_production_focus.yaml`** — all `CHANGE_ME` fields (see STEP 2 in the section 4. Step-by-Step Deployment below)
- [ ] **6. Confirm your Traefik certResolver name** — currently it is set to  `le` inside `values_production_focus.yaml`. It is the name of the Let's Encrypt resolver configured in your cluster for Patient Dashboard. Check this existing Traefik configuration (e.g. `le`, `letsencrypt`).
- [ ] **6. Install Helm chart** — `helm install caregiver helm/ --namespace fall-dashboard`
- [ ] **7. Verify both pods are running** — see verification commands below
- [ ] **8. Share your subdomains with Isa** — he needs them for the mobile app config:
  - Subdomain for MQTT broker (see 2-1 above)
  - MQTT username + password ([optional] if you enabled broker auth in `values_production_focus.yaml`)
  - Subdomain for Fall-dashboard (for Flutter `/api/stream` SSE calls)



## 3. What Each Service Does

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
- Shows live alerts on the browser when a fall is predicted and when a help is requested via Server-Sent Events (SSE)
- Queries FOCUS InfluxDB for per-patient fall history 
- Stores the patient list in SQLite on a PVC — patients are added/deleted from the UI.

What it does NOT do:
- No inference — it only displays alerts that the mobile app already detected and confirmed
- No Postgres — fall data is written into Influx DB and fall history comes from there too. 
- No connection to any external service - the running independently from MCS server.



## 4. Step-by-Step Deployment

### Prerequisites

- K3s cluster with Traefik already running
- `kubectl` and `helm` installed
- Registry pull credentials for `registry.smarko-health.de` (use the same credentials as before)
- Helm chart (`helm/` folder) and `values_production_focus.yaml` delivered by Mohammed — no source code; the Docker image is already in the registry

### STEP 0: Firewall — verify required ports are open

Both services are exposed through Traefik on standard web ports only. No custom port needs to be opened.

| Port | Protocol | Required for |
|------|----------|--------------|
| 80 | TCP | Let's Encrypt HTTP-01 ACME challenge (TLS certificate issuance) |
| 443 | TCP | Traefik HTTPS → fall-dashboard **and** Traefik WSS → MQTT broker (same port, different routes) |
| 6443 | TCP | K3s API server — needed if you run `kubectl` from a machine outside the server |

If you already have Traefik serving other services, ports 80 and 443 are almost certainly already open and **no new firewall rules are needed**. Run the check below to confirm.

**Check with ufw (Ubuntu):**

```bash
sudo ufw status
# Expected: 80/tcp and 443/tcp should show ALLOW
```

**If they are missing, open them:**

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw reload
```

**If you use firewalld (CentOS/RHEL):**

```bash
sudo firewall-cmd --list-ports           # check current open ports
sudo firewall-cmd --add-port=80/tcp --permanent
sudo firewall-cmd --add-port=443/tcp --permanent
sudo firewall-cmd --reload
```

> **k3s + ufw note:** k3s manages its own iptables rules. If you see pods unable to reach each other after enabling ufw, add `--flannel-iface=<interface>` to your k3s service args or whitelist the `flannel.1` and `cni0` interfaces. This is a k3s-level concern, not specific to this deployment.

> **Nothing extra for MQTT:** The mobile app connects to the MQTT broker via WebSocket on port 443 (WSS). Traefik routes this traffic internally to the mosquitto pod on port 9001. No separate MQTT port (1883 or 9001) needs to be opened in the OS firewall.

### STEP 1: One-time cluster setup (run once, never again)

```bash
# Add the MQTT WebSocket entrypoint to Traefik (required for WSS on port 443)
kubectl apply -f helm/extras/traefik-mqtts-entrypoint.yaml

# Create namespace
kubectl create namespace fall-dashboard

# Create registry pull secret (use the same credentials given by Mohammed before)
kubectl create secret docker.registry mcs-labs \
    --docker-server=registry.smarko-health.de \
    --docker-username=<username> \
    --docker-password=<password> \
    --namespace fall-dashboard
```

### STEP 2: Fill in values_production_focus.yaml

Open `helm/values_production_focus.yaml` and replace every `CHANGE_ME`.

**Registry (fill in after Mohammed confirms image is pushed):**

| Key | Value |
|-----|-------|
| `imagePullSecret` | `mcs-labs` |
| `fallDashboard.image` | `registry.smarko-health.de/fall-detection/fall-dashboard:latest` |
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
| `namespace` | `fall-dashboard` | Recommended name — can be changed to any valid k8s namespace name (lowercase, hyphens allowed, no underscores) |
| `mosquitto.port` | `1883` | Internal TCP, cluster-only |
| `mosquitto.wsPort` | `9001` | Internal WS, Traefik forwards to this |
| `fallDashboard.replicas` | `1` | Must stay 1 — SSE fan-out is in-process; multiple replicas break live alerts |
| `fallDashboard.port` | `8002` | Internal pod port |
| `fallDashboard.mqtt.alertTopic` | `fall/alert` | Must match MCS side |
| `fallDashboard.mqtt.possibleTopic` | `fall/possible` | Must match MCS side |

Full table with all keys and descriptions: [`config_checklist_focus_devops.md`](config_checklist_focus_devops.md).

### STEP 3: Install the Helm chart

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

> `helm upgrade` is required whenever `values_production_focus.yaml` changes — K3s does not
> pick up values file changes automatically.

### STEP 4: Verify

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

### STEP 5: Verify Traefik routing

After both pods are running, test the public endpoints:

```bash
# Fall dashboard (HTTPS)
curl https://fall.focus-hospital.de/api/patients

# MQTT (WebSocket — test TCP connectivity first)
curl -v --http1.1 https://mqtt.focus-hospital.de
# Then test from Isa's mobile app
```


## Cross-Party Agreements — What to Share and With Whom

### You → Isa (mobile app developer)

Isa needs these to configure the mobile app. Share as soon as your deployment is live:

| What | Where you set it | What Isa does with it |
|------|-----------------|----------------------|
| MQTT broker domain | `mosquitto.ingress.host` in `values_production_focus.yaml` | Mobile app connects as `wss://<domain>:443` to publish fall alerts |
| MQTT username (if configured)| `fallDashboard.mqtt.username` | Mobile app authenticates with broker |
| MQTT password (if configured)| `fallDashboard.mqtt.password` | Mobile app authenticates with broker |

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
