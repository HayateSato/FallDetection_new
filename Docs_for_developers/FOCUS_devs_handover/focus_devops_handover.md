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

- [ ] **1. Create two subdomains** 
  1) subdomain for the MQTT broker (e.g. `mqtt.focus-hospital.de`) — the mobile app needs to connect here via WSS - Isa will need to know for mobile app to send fall alert so please share it as soon as decided.
  2) subdomain for the fall dashboard (e.g. `fall.focus-hospital.de`) — caregiver needs to open this to check real time fall alert or fall history.          
- [ ] **2. Pull docker images** — fall-dashboard Docker image is already pushed to `registry.smarko-health.de`.
- [ ] **3. Follow and complete 5 steps in deployment guide below** 
  1)  Firewall setting (skip if already allowed)
  2)  Cluster set up (create namespace + pull secret + Traefik MQTT entrypoint)
  3)  Update values_production_focus.yaml (update all `CHANGE_ME` fields + name of the Let's Encrypt resolver)
  4)  Install Helm (`helm install caregiver helm/ --namespace fall-dashboard`)
  5)  Verify (Pods & Traefik routing)
- [ ] **4. Share your subdomains with Isa** — he needs them for the mobile app config:
  - Subdomain for MQTT broker (see 2-1 above) — mobile app connects here to publish fall alerts
  - MQTT username + password ([optional] if you enabled broker auth in `values_production_focus.yaml`)



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
- Helm chart (`helm/` folder) and `values_production_focus.yaml`; the Docker image is already in the registry

---

### STEP 1: Firewall — verify required ports are open

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

---

### STEP 2: One-time cluster setup (run once, never again)

  ```bash
  # Create namespace
  kubectl create namespace fall-dashboard

  # Create registry pull secret (use the same credentials given by Mohammed before)
  kubectl create secret docker-registry mcs-labs \
      --docker-server=registry.smarko-health.de \
      --docker-username=<username> \
      --docker-password=<password> \
      --namespace fall-dashboard
  ```
---

### STEP 3: Fill in values_production_focus.yaml

Open `helm/values_production_focus.yaml`. There are **3 fields you must fill in**, 4 optional fields, and the rest is pre-filled — do not change those.

#### **Required** — fill in all 3 `CHANGE_ME` fields

| Key | What to put |
|-----|-------------|
| `mosquitto.ingress.host` | MQTT broker subdomain (e.g. `mqtt.focus-hospital.de`) — mobile app connects here via WSS |
| `fallDashboard.ingress.host` | Fall-dashboard subdomain (e.g. `fall.focus-hospital.de`) — caregiver opens this in the browser |
| `fallDashboard.influxdb.token` | InfluxDB read token for the FOCUS InfluxDB instance |

#### **Optional** — check these, change only if needed

| Key | Current value | When to change |
|-----|---------------|----------------|
| `mosquitto.ingress.certResolver` | `"le"` | If your Traefik cluster uses a different cert resolver name, update it here |
| `fallDashboard.ingress.certResolver` | `"le"` | Same — must match your Traefik cert resolver name |
| `fallDashboard.mqtt.username` | `""` (blank) | Set if you want MQTT broker authentication |
| `fallDashboard.mqtt.password` | `""` (blank) | Set together with username above |

> If you set MQTT credentials, share them with Isa. The mobile app must authenticate with the same username and password when connecting to the broker. A mismatch causes silent connection failure — fall alerts will not appear on the dashboard.

#### Do NOT change

| Key | Pre-filled value | Why |
|-----|-----------------|-----|
| `imagePullSecret` | `mcs-labs` | Name of the registry pull secret — agreed with Mohammed; not a credential itself |
| `fallDashboard.image` | `registry.smarko-health.de/fall-dashboard:latest` | Points to the image already in the MCS registry |
| `fallDashboard.imagePullPolicy` | `Always` | Ensures latest image is pulled on every upgrade |
| `fallDashboard.influxdb.url` | `https://influxdb.internal.e-healthservice.de/` | FOCUS InfluxDB URL (confirmed with FOCUS) |
| `fallDashboard.influxdb.org` | `mcs-data-labs` | FOCUS InfluxDB organisation name (confirmed with FOCUS) |
| `fallDashboard.influxdb.bucket` | `6g-path` | Bucket where fall events are stored |
| `fallDashboard.influxdb.fallEventsBucket` | `6g-path` | Same bucket (fall events and sensor data share one bucket) |
| `fallDashboard.replicas` | `1` | Must stay 1 — SSE fan-out is in-process; multiple replicas silently drop live alerts |
| `fallDashboard.port` | `8002` | Internal pod port — Traefik routes to this |
| `fallDashboard.httpNodePort` | `""` (blank) | ClusterIP in production; Traefik IngressRoute handles external access |
| `mosquitto.port` | `1883` | Internal TCP, cluster-only |
| `mosquitto.wsPort` | `9001` | Internal WebSocket port; Traefik forwards WSS :443 here |
| `mosquitto.wsNodePort` | `""` (blank) | Used for local testing only; leave blank in production |
| `fallDashboard.mqtt.alertTopic` | `fall/alert` | Must match the mobile app — do not change |
| `fallDashboard.mqtt.possibleTopic` | `fall/possible` | Must match the mobile app — do not change |
| `namespace` | `fall-dashboard` | Recommended namespace name — can be any valid lowercase k3s name (hyphens allowed, no underscores), but must match the namespace used in STEP 2 commands |

---

### STEP 4: Install the Helm chart

(Run below from `_6G_integration_v3_k3s/` as working directory)

```bash
# First install
helm install caregiver helm/ \
  --namespace fall-dashboard \
  --values helm/values_production_focus.yaml

# Future updates (e.g. values change or image update)
helm upgrade caregiver helm/ \
  --namespace fall-dashboard \
  --values helm/values_production_focus.yaml
```

> `helm upgrade` is required whenever `values_production_focus.yaml` changes — K3s does not
> pick up values file changes automatically.

---

### STEP 5: Verify

#### Check if both pods are running 

(Run below from inside the cluster)

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

#### Check if pods are publicly reachable

After both pods are running, test if Traefik is handling the routing as planned.

**Test 1 — Fall dashboard (run from a terminal)**

```bash
# Expect a JSON list of registered patients
curl https://<fall-dashboard subdomain>/api/patients
```

**Test 2 — MQTT broker (use Postman)**

Terminal MQTT clients can be unreliable with Traefik TLS — use Postman for a reliable WSS test:

1. Open Postman → Open your workspace → click **New** (or **+** mark in top left) → select **MQTT** 
2. In the URL field enter `wss://<mqtt broker subdomain>` (use "V5") — Postman uses port 443 by default for `wss://`
3. Click **Connect** — the status indicator should turn green and show **Connected**
4. Go to the **Topics** tab, enter topic `#`, turn on **Subscribe** toggle.
5. Go to the **Message** tab, enter anything (e.g. Hello from Postman") in message body (while choosing Text), enter `test/ping` in the topic field on the bottom right. Then click **Send**
6. The published message should appear immediately in the **Response** section

Step 3 confirms Traefik is routing WSS traffic to the mosquitto pod. Steps 5–6 confirm the full publish/subscribe round-trip is working.


## 5. Cross-Party Agreements — What to Share and With Whom

### FOCUS → Isa (mobile app developer)

Isa needs these to configure the mobile app so that the mobile app can send fall alert. Please share them as soon as your deployment is live:

| What | Where you set it | What Isa does with it |
|------|-----------------|----------------------|
| MQTT broker domain | `mosquitto.ingress.host` in `values_production_focus.yaml` | Mobile app connects as `wss://<domain>:443` to publish fall alerts |
| MQTT username (if configured)| `fallDashboard.mqtt.username` | Mobile app authenticates with broker |
| MQTT password (if configured)| `fallDashboard.mqtt.password` | Mobile app authenticates with broker |

