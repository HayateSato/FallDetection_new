# Two-Laptop K3s Test Guide

## Setup

| Machine | Role | Services |
|---|---|---|
| Laptop 1 | FOCUS side (K3s) | mosquitto + fall-dashboard as K3s pods |
| Laptop 2 | MCS side (Docker) | inference-server, postgres, mlflow, minio + mock-app |

The mock-app on Laptop 2 simulates the Isa mobile app. It calls the inference server locally
and publishes MQTT alerts to Laptop 1's mosquitto via WebSocket port 9001.

Note: React Native cannot open raw TCP sockets in standard JS, so the mobile app uses MQTT
over WebSocket (MQTT.js). Mosquitto runs two listeners: 1883 (internal, for fall-dashboard
inside the cluster) and 9001 WebSocket (external, for the mobile app / mock-app).

```
mock-app (Laptop 2)
  |-- HTTP POST /predict --> inference-server (Laptop 2, localhost:8001)
  |-- PUBLISH fall/alert --> mosquitto (Laptop 1, ws://<L1-IP>:9001 WebSocket)
  |-- WRITE fall_events  --> InfluxDB (cloud)

fall-dashboard (Laptop 1, K3s pod)
  |-- SUBSCRIBE fall/alert/# --> mosquitto (cluster-internal, 1883 TCP)
  |-- SSE fan-out --> Flutter dashboard
```

---

## Laptop 1 — K3s caregiver layer

### Step 1 — Prerequisites

A Kubernetes cluster, kubectl, and Helm must be available. Check:

```powershell
kubectl version --client
helm version
kubectl get nodes   # should show one node Ready
```

**Option A — Docker Desktop Kubernetes (Windows local testing)**

Enable Kubernetes in Docker Desktop → Settings → Kubernetes → "Enable Kubernetes" → Apply & Restart.
This is the simplest option for local two-laptop testing on Windows.

Docker Desktop does NOT include Traefik (unlike real k3s). Install it once:

```powershell
helm repo add traefik https://traefik.github.io/charts
helm repo update
helm install traefik traefik/traefik --namespace kube-system
kubectl get pods -n kube-system | Select-String "traefik"
# Wait until: traefik-xxxx   1/1   Running
```

Traefik is needed because the caregiver Helm chart creates `IngressRoute` resources (Traefik CRDs).
Without it the chart install fails with "no matches for kind IngressRoute".

**Option B — Real k3s (WSL2 or Linux)**

Real k3s bundles Traefik automatically — no separate install needed.

Install k3s inside WSL2:

```bash
# Inside WSL2
curl -sfL https://get.k3s.io | sh -
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER ~/.kube/config
```

Then from PowerShell, point kubectl at the WSL2 kubeconfig:

```powershell
$env:KUBECONFIG = "\\wsl$\Ubuntu\home\<your-wsl-user>\.kube\config"
kubectl get nodes   # should show one node Ready
```

---

### Step 2 — Open firewall port 30901 on Laptop 1

For local testing the Helm chart exposes mosquitto's WebSocket port as NodePort 30901 on the
host NIC (see Step 5 — this requires setting `mosquitto.wsNodePort: 30901` in values.yaml).
Open the firewall so Laptop 2 (mock-app) can reach it:

```powershell
New-NetFirewallRule -DisplayName "Fall Detection - MQTT WebSocket NodePort (K3s)" -Direction Inbound -Protocol TCP -LocalPort 30901 -Action Allow
```

Also open 80 and 443 if the Flutter dashboard will be tested too:

```powershell
New-NetFirewallRule -DisplayName "Fall Detection - HTTP (K3s)" -Direction Inbound -Protocol TCP -LocalPort 80 -Action Allow
New-NetFirewallRule -DisplayName "Fall Detection - HTTPS (K3s)" -Direction Inbound -Protocol TCP -LocalPort 443 -Action Allow
```

**Linux (FOCUS production server):** On most k3s-ready Linux servers (Ubuntu Server, Debian)
`ufw` is inactive and no firewall change is needed — k3s adds the iptables NodePort rules
automatically. Check first:

```bash
sudo ufw status
```

If `ufw` is active, allow the port:

```bash
sudo ufw allow 30901/tcp      # MQTT WebSocket NodePort
sudo ufw allow 80/tcp         # HTTP (fall-dashboard via Traefik)
sudo ufw allow 443/tcp        # HTTPS (fall-dashboard + WSS via Traefik)
```

With `firewalld` (RHEL/CentOS):

```bash
sudo firewall-cmd --add-port=30901/tcp --permanent
sudo firewall-cmd --add-port=80/tcp --permanent
sudo firewall-cmd --add-port=443/tcp --permanent
sudo firewall-cmd --reload
```

Note: in the production FOCUS deployment `wsNodePort` will be blank (ClusterIP) and port 30901
will not be used — mobile app WebSocket traffic goes through Traefik on port 443 instead.
The NodePort rule is only needed for the local two-laptop test.

---

### Step 3 — Understand how external WebSocket access works

For local testing the Helm chart exposes mosquitto's WebSocket port (9001) as a NodePort on
the host NIC. This is controlled by `mosquitto.wsNodePort` in `values.yaml` — set it to
`30901` in Step 5 before running helm install.

The mock-app (and eventually the real mobile app) connects to `ws://<LAPTOP1_IP>:30901`.
The fall-dashboard pod reaches mosquitto on 1883 via cluster-internal DNS — no change needed there.

In production (FOCUS cluster) `wsNodePort` is left blank so the service is ClusterIP.
Traefik IngressRoute handles external mobile app traffic as WSS on port 443 instead.

> **Verification commands (run these AFTER Step 6 helm install):**
>
> ```powershell
> kubectl get svc mosquitto -n fall-dashboard
> # Should show: port 9001:30901/TCP (NodePort) alongside 1883 (ClusterIP internal)
>
> kubectl logs -n fall-dashboard -l app=mosquitto | Select-String "1883|9001"
> # Expected: "Opening ipv4 listen socket on port 1883" and "Opening websockets listen socket on port 9001"
> ```

---

### Step 4 — Build the fall-dashboard image locally

For local testing we build a local image and use it directly — no registry push needed.
Docker Desktop Kubernetes shares the same Docker daemon as your desktop, so any image
you build with `docker build` is immediately available to the cluster.

Build (run from `_6G_integration_v3_k3s/` as working directory):

```powershell
docker build -t fall-detection/fall-dashboard:local -f fall_dashboard/Dockerfile .
```

The `values.yaml` for local testing already points at this image tag:
```yaml
fallDashboard:
  image: fall-detection/fall-dashboard:local
  imagePullPolicy: IfNotPresent   # uses local image; never tries to pull from registry
```

`imagePullPolicy: IfNotPresent` is critical — without it Kubernetes tries to pull from the
registry (`registry-smarko-health.de`) and fails because we have not pushed there yet.

> **Note for real k3s (WSL2):** k3s uses its own containerd, which does NOT share the
> Docker daemon. You would need to either run a local registry or import the image with:
> `docker save fall-detection/fall-dashboard:local | k3s ctr images import -`

---

### Step 5 — Fill in values.yaml for local test

Open `_6G_integration_v3_k3s/helm/values.yaml` and update:

```yaml
# No registry pull needed for local testing -- image is built locally
imagePullSecret: ""        # blank: no pull secret needed

mosquitto:
  wsNodePort: 30901   # expose WS port on host NIC so Laptop 2 mock-app can reach it
  ingress:
    host: "CHANGE_ME"   # still required by template but not used in local NodePort test
    certResolver: ""

fallDashboard:
  image: fall-detection/fall-dashboard:local   # locally built image (docker build step above)
  imagePullPolicy: IfNotPresent                # do not pull from registry
  patientIds: "patient_test_1"    # must match Laptop 2 mock-app PATIENT_IDS
  macIds: "6c:1d:eb:04:a9:d9"

  mqtt:
    alertTopic: fall/alert
    possibleTopic: fall/possible
    username: ""       # leave blank — mosquitto still allow_anonymous for local test
    password: ""

  influxdb:
    url: "https://ecosystem-influxdb.smarko-health.de"
    org: "MCS Datalabs GmbH"
    bucket: "fd_test"
    fallEventsBucket: "fd_test"
    token: "<your-influxdb-token>"

  ingress:
    host: "localhost"   # local test only — no real domain needed
    certResolver: ""    # no TLS for local test
```

---

### Step 6 — Install the Helm chart

```powershell
# Run from _6G_integration_v3_k3s/ as working directory
.\helm\install.ps1
```

Check pods are Running:

```powershell
kubectl get pods -n fall-dashboard
# Expected:
#   mosquitto-xxxx      1/1   Running
#   fall-dashboard-xxxx 1/1   Running
```

---

### Step 7 — Smoke test

```powershell
.\helm\test.ps1
# Expected: 7/7 PASS
```

---

### Step 8 — Note Laptop 1's IP

```powershell
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.PrefixOrigin -eq "Dhcp" } | Select-Object IPAddress, InterfaceAlias
```

Write down the IP — you will need it for Laptop 2 step 3.

---

## Laptop 2 — Inference layer + mock-app

### Step 1 — Start inference layer

```powershell
# Run from _6G_integration_v3_docker_mcs/ as working directory
docker compose --env-file .env up -d
```

Wait for inference-server to be healthy:

```powershell
docker ps | Select-String "fall_inference"
# should show (healthy)
```

---

### Step 2 — Get Laptop 2's own IP (needed by mock-app)

```powershell
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.PrefixOrigin -eq "Dhcp" } | Select-Object IPAddress, InterfaceAlias
```

---

### Step 3 — Run mock-app pointing to Laptop 1 MQTT WebSocket

The mock-app is NOT in the K3s chart (dev-only). Run it as a standalone Docker container,
overriding the MQTT broker to point at Laptop 1's mosquitto NodePort 30901 (set via
`mosquitto.wsNodePort: 30901` in values.yaml — see Laptop 1 Steps 3 and 5).

Build the image first (if not already built):

```powershell
# Run from _6G_integration_v3_docker_focus/ as working directory
docker build -t fall-detection/mock-app:latest -f mock_app/Dockerfile .
```

Run it, substituting `<LAPTOP1_IP>` with the IP from Laptop 1 Step 8:

```powershell
docker run --rm `
  -e INFLUXDB_URL="https://ecosystem-influxdb.smarko-health.de" `
  -e INFLUXDB_TOKEN="<your-influxdb-token>" `
  -e INFLUXDB_ORG="MCS Datalabs GmbH" `
  -e INFLUXDB_BUCKET="fd_test" `
  -e INFLUXDB_FALL_EVENTS_BUCKET="fd_test" `
  -e INFERENCE_SERVER_URL="http://host.docker.internal:8001" `
  -e INFERENCE_API_KEY="<your-api-key>" `
  -e MQTT_BROKER_HOST="<LAPTOP1_IP>" `
  -e MQTT_BROKER_PORT="30901" `
  -e MQTT_TRANSPORT="websockets" `
  -e MQTT_ALERT_TOPIC="fall/alert" `
  -e MQTT_POSSIBLE_TOPIC="fall/possible" `
  -e MQTT_USERNAME="" `
  -e MQTT_PASSWORD="" `
  -e PATIENT_IDS="patient_test_1" `
  -e MAC_IDS="6c:1d:eb:04:a9:d9" `
  -e POLL_INTERVAL_SECONDS="10" `
  -e POLL_LOOKBACK_SECONDS="15" `
  -e MOCK_PATIENT_RESPONSE_TIMEOUT="1" `
  -e MOCK_PATIENT_SERVER_PORT="8005" `
  -e LOG_LEVEL="INFO" `
  -p 8005:8005 `
  fall-detection/mock-app:latest
```

> **No `--network host`** — that flag does not work on Windows Docker Desktop (containers run
> inside a Linux VM so host networking binds to the VM NIC, not the Windows NIC).
> Instead, `host.docker.internal` is Docker Desktop's special hostname that routes back
> to the Windows host where inference-server is running. Port 8005 is exposed via `-p 8005:8005`
> so the patient popup is reachable at `http://localhost:8005/` in a browser on Laptop 2.

---

## End-to-end test

### Watch fall-dashboard logs on Laptop 1

```powershell
kubectl logs -n fall-dashboard -l app=fall-dashboard -f
```

### Watch mosquitto logs on Laptop 1

```powershell
kubectl logs -n fall-dashboard -l app=mosquitto -f
```

### Trigger a fall via mock-app (Laptop 2 browser)

Open `http://localhost:8005/` in a browser on Laptop 2. This is the patient confirmation
popup page. Wait for the mock-app to detect a fall from InfluxDB data and show the popup,
then confirm or let it time out.

### Verify the alert reached Laptop 1

In the fall-dashboard logs you should see:
```
MQTT message received: fall/alert/patient_test_1
SSE: broadcasting fall event for patient_test_1
```

### Check fall-dashboard API directly (via port-forward)

```powershell
kubectl port-forward -n fall-dashboard svc/fall-dashboard 18002:8002
# In a second terminal:
curl.exe http://localhost:18002/api/falls
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| mock-app: `Connection refused` to MQTT WebSocket | Firewall on Laptop 1 blocking port 30901 | Re-run Laptop 1 Step 2 firewall rule |
| mock-app: `Connection refused` to MQTT WebSocket | NodePort service not created — `wsNodePort` blank or helm not reinstalled after values change | Verify `mosquitto.wsNodePort: 30901` in values.yaml, run `helm upgrade ...`, check `kubectl get svc -n fall-dashboard` shows `9001:30901/TCP` |
| fall-dashboard pod: `ImagePullBackOff` | Local registry not reachable from K3s | Check registries.yaml in WSL2, restart k3s |
| MQTT connected but no SSE events | fall-dashboard not subscribed to mosquitto | Check fall-dashboard logs for MQTT connect message |
| fall-dashboard log shows `MQTT broker connection failed ([Errno 111] Connection refused)` at startup | fall-dashboard started before mosquitto was ready — one-time race condition | Delete the pod: `kubectl delete pod -n fall-dashboard <fall-dashboard-pod-name>` — k8s recreates it and this time mosquitto is already running. The code retries 5 times (25s) so this should self-heal on recreation. |
| mock-app: `Connection refused` to inference-server | inference layer not healthy yet | Check `docker ps` on Laptop 2 for `(healthy)` |
| All tests pass but no falls triggered | No new sensor data in InfluxDB | Check InfluxDB `fd_test` bucket has recent ACC data |
