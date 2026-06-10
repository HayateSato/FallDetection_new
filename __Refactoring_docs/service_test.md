# Service Health Tests

Run all commands from the folder indicated in each section header.

---

## MCS Layer — `_6G_integration_v3_docker_mcs/` (any laptop running MCS)

Same commands regardless of whether the caregiver layer is Docker or K3s.

```powershell
# ── Python services ──────────────────────────────────────────────────────────
curl.exe http://localhost:8001/health          # inference-server
curl.exe http://localhost:8004/                # ml-dashboard
curl.exe http://localhost:8006/                # server-health

# ── MLflow ───────────────────────────────────────────────────────────────────
curl.exe http://localhost:5000/                # MLflow UI (returns HTML)

# ── Prometheus ───────────────────────────────────────────────────────────────
curl.exe http://localhost:9090/-/healthy

# ── Grafana ──────────────────────────────────────────────────────────────────
curl.exe http://localhost:3000/api/health

# ── MinIO ────────────────────────────────────────────────────────────────────
curl.exe http://localhost:9000/minio/health/live
# Console (browser): http://localhost:9002

# ── Postgres ─────────────────────────────────────────────────────────────────
docker compose --env-file .env exec postgres pg_isready -U fall_user -d fall_detection

# ── All services status ───────────────────────────────────────────────────────
docker compose --env-file .env ps
```

| Service | Expected response |
|---------|------------------|
| inference-server | `{"status":"ok","model_version":"v0",...}` |
| ml-dashboard | HTML page |
| server-health | HTML page |
| mlflow | HTML page |
| prometheus | `Prometheus Server is Healthy.` |
| grafana | `{"commit":"...","database":"ok",...}` |
| minio | HTTP 200 (empty body) |
| postgres | `localhost:5432 - accepting connections` |

---

## Caregiver Layer — Docker version (`_6G_integration_v3_docker_focus/`)

Used for the two-laptop Docker test. Run from `_6G_integration_v3_docker_focus/`.

```powershell
# ── Fall dashboard ────────────────────────────────────────────────────────────
curl.exe http://localhost:8002/api/patients    # patient list
curl.exe http://localhost:8002/api/falls       # fall history (from InfluxDB)

# ── Mock app patient popup (open in browser) ──────────────────────────────────
# http://localhost:8005/

# ── MQTT — publish a test message (broker internal TCP port) ──────────────────
docker compose --env-file .env exec mqtt mosquitto_pub -h localhost -p 1883 -t test -m ping

# ── Mock app logs ─────────────────────────────────────────────────────────────
docker compose --env-file .env logs -f mock-app

# ── Fall dashboard logs ───────────────────────────────────────────────────────
docker compose --env-file .env logs -f fall-dashboard

# ── All services status ────────────────────────────────────────────────────────
docker compose --env-file .env ps
```

| Service | Expected response |
|---------|------------------|
| fall-dashboard `/api/patients` | `{"patients":[...]}` (empty list is fine on first run) |
| fall-dashboard `/api/falls` | `{"falls":[...]}` (empty until a fall fires) |
| mock-app popup | Browser UI at http://localhost:8005/ |
| mqtt test publish | no output = success |

---

## Caregiver Layer — K3s version (`_6G_integration_v3_k3s/`)

Used for the two-laptop K3s test. No Docker Compose — use kubectl.
Mock-app is NOT in the K3s chart; it runs as a standalone container on the MCS laptop.

```powershell
# ── Pod status ────────────────────────────────────────────────────────────────
kubectl get pods -n fall-dashboard
# Expected: mosquitto-xxxx 1/1 Running, fall-dashboard-xxxx 1/1 Running

# ── Services and NodePorts ────────────────────────────────────────────────────
kubectl get svc -n fall-dashboard
# Expected: mosquitto 9001:30901/TCP, fall-dashboard 8002:30802/TCP

# ── Fall dashboard API (via NodePort 30802) ───────────────────────────────────
curl.exe http://localhost:30802/api/patients
curl.exe http://localhost:30802/api/falls

# ── Fall dashboard API (via port-forward alternative) ─────────────────────────
kubectl port-forward -n fall-dashboard svc/fall-dashboard 18002:8002
# then in a second terminal:
curl.exe http://localhost:18002/api/patients

# ── Fall dashboard logs ───────────────────────────────────────────────────────
kubectl logs -n fall-dashboard -l app=fall-dashboard -f

# ── Mosquitto logs ────────────────────────────────────────────────────────────
kubectl logs -n fall-dashboard -l app=mosquitto -f
# Expect: "Opening ipv4 listen socket on port 1883" and "Opening websockets listen socket on port 9001"

# ── MQTT test publish (cluster-internal TCP port) ─────────────────────────────
kubectl exec -n fall-dashboard -l app=mosquitto -- mosquitto_pub -h localhost -p 1883 -t test -m ping
# no output = success

# ── Run smoke test suite ──────────────────────────────────────────────────────
.\helm\test.ps1
# Expected: 7/7 PASS
```

| Service | Expected response |
|---------|------------------|
| fall-dashboard `/api/patients` | `{"patients":[...]}` (empty list is fine on first run) |
| fall-dashboard `/api/falls` | `{"falls":[...]}` (empty until a fall fires) |
| mosquitto test publish | no output = success |
| smoke test | 7/7 PASS |

---

## Cross-machine checks

### Docker caregiver version

Run on **MCS laptop** — verify it can reach the caregiver laptop:

```powershell
curl.exe http://<CAREGIVER_LAPTOP_IP>:8002/api/patients
```

Run on **caregiver laptop** — verify it can reach the MCS laptop:

```powershell
curl.exe http://<MCS_LAPTOP_IP>:8001/health
```

### K3s caregiver version

Run on **MCS laptop** — verify it can reach the K3s caregiver (via NodePort):

```powershell
curl.exe http://<K3S_LAPTOP_IP>:30802/api/patients
```

Run on **K3s laptop** — verify it can reach the MCS laptop:

```powershell
curl.exe http://<MCS_LAPTOP_IP>:8001/health
```

Both should return JSON. If not, check firewall rules on the target machine.
