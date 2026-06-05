# Two-Laptop K3s Test Guide

## Setup

| Machine | Role | Services |
|---|---|---|
| Laptop 1 | FOCUS side (K3s) | mosquitto + fall-dashboard as K3s pods |
| Laptop 2 | MCS side (Docker) | inference-server, postgres, mlflow, minio + mock-app |

The mock-app on Laptop 2 simulates the Isa mobile app. It calls the inference server locally
and publishes MQTT alerts to Laptop 1's mosquitto via the Traefik TCP port 8883.

```
mock-app (Laptop 2)
  |-- HTTP POST /predict --> inference-server (Laptop 2, localhost:8001)
  |-- PUBLISH fall/alert --> mosquitto (Laptop 1, <L1-IP>:8883 via Traefik)
  |-- WRITE fall_events  --> InfluxDB (cloud)

fall-dashboard (Laptop 1, K3s pod)
  |-- SUBSCRIBE fall/alert/# --> mosquitto (cluster-internal, 1883)
  |-- SSE fan-out --> Flutter dashboard
```

---

## Laptop 1 — K3s caregiver layer

### Step 1 — Prerequisites

K3s and Helm must be installed. Check:

```powershell
kubectl version --client
helm version
```

If K3s is not installed, install it inside WSL2:

```bash
# Inside WSL2
curl -sfL https://get.k3s.io | sh -
# Copy kubeconfig to Windows-accessible location
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

### Step 2 — Open firewall port 8883 on Laptop 1

Traefik will serve MQTT on port 8883. Open it so Laptop 2 can reach it:

```powershell
New-NetFirewallRule -DisplayName "Fall Detection - MQTTS (K3s)" -Direction Inbound -Protocol TCP -LocalPort 8883 -Action Allow
```

Also open 80 and 443 if the Flutter dashboard will be tested too:

```powershell
New-NetFirewallRule -DisplayName "Fall Detection - HTTP (K3s)" -Direction Inbound -Protocol TCP -LocalPort 80 -Action Allow
New-NetFirewallRule -DisplayName "Fall Detection - HTTPS (K3s)" -Direction Inbound -Protocol TCP -LocalPort 443 -Action Allow
```

---

### Step 3 — Apply Traefik TCP entrypoint (one-time cluster change)

This patches K3s's built-in Traefik to listen on port 8883 for MQTT TCP traffic.

```powershell
# Run from _6G_integration_v3_k3s/ as working directory
kubectl apply -f helm/extras/traefik-mqtts-entrypoint.yaml
```

Verify Traefik picked it up (takes ~30 seconds to restart):

```powershell
kubectl rollout status deployment/traefik -n kube-system
kubectl logs -n kube-system -l app.kubernetes.io/name=traefik | Select-String "8883"
```

---

### Step 4 — Build the fall-dashboard image locally

For local testing, push to a local registry running in Docker rather than the FOCUS
production registry (`registry-smarko-health.de`).

Start a local registry:

```powershell
docker run -d -p 5000:5000 --name local-registry registry:2
```

Build and push (run from `_6G_integration_v3_k3s/` as working directory):

```powershell
docker build -t localhost:5000/fall-detection/fall-dashboard:latest -f fall_dashboard/Dockerfile .
docker push localhost:5000/fall-detection/fall-dashboard:latest
```

Make the local registry reachable from inside WSL2/K3s. Add it as an insecure registry
in the K3s containerd config (inside WSL2):

```bash
# Inside WSL2
sudo mkdir -p /etc/rancher/k3s
sudo tee /etc/rancher/k3s/registries.yaml <<EOF
mirrors:
  "localhost:5000":
    endpoint:
      - "http://localhost:5000"
EOF
sudo systemctl restart k3s
```

---

### Step 5 — Fill in values.yaml for local test

Open `_6G_integration_v3_k3s/helm/values.yaml` and update:

```yaml
# Use local registry instead of FOCUS production registry
registry: localhost:5000
imagePullSecret: ""        # no pull secret needed for local registry

fallDashboard:
  image: localhost:5000/fall-detection/fall-dashboard:latest
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

### Step 3 — Run mock-app pointing to Laptop 1 MQTT

The mock-app is NOT in the K3s chart (dev-only). Run it as a standalone Docker container,
overriding the MQTT broker to point at Laptop 1's Traefik port 8883.

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
  -e INFERENCE_SERVER_URL="http://localhost:8001" `
  -e INFERENCE_API_KEY="<your-api-key>" `
  -e MQTT_BROKER_HOST="<LAPTOP1_IP>" `
  -e MQTT_BROKER_PORT="8883" `
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
  --network host `
  fall-detection/mock-app:latest
```

> `--network host` lets the container reach `localhost:8001` (inference-server) directly.

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
| mock-app: `Connection refused` to MQTT | Firewall on Laptop 1 blocking 8883 | Re-run Step 2 of Laptop 1 section |
| mock-app: `Connection refused` to MQTT | Traefik not listening on 8883 | Re-run Step 3, check `traefik` pod restarted |
| fall-dashboard pod: `ImagePullBackOff` | Local registry not reachable from K3s | Check registries.yaml in WSL2, restart k3s |
| MQTT connected but no SSE events | fall-dashboard not subscribed to mosquitto | Check fall-dashboard logs for MQTT connect message |
| mock-app: `Connection refused` to inference-server | inference layer not healthy yet | Check `docker ps` on Laptop 2 for `(healthy)` |
| All tests pass but no falls triggered | No new sensor data in InfluxDB | Check InfluxDB `fd_test` bucket has recent ACC data |
