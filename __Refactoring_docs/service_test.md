# Service Health Tests

Run all commands from `_6G_Integration_v2_mqtt/` as working directory.

---

## MCS Layer — `inference_posttraining_layer` (Laptop 1)

```powershell
# ── Python services ──────────────────────────────────────────────────────
curl.exe http://localhost:8001/health          # inference-server
curl.exe http://localhost:8004/                # ml-dashboard
curl.exe http://localhost:8006/                # server-health

# ── MLflow ───────────────────────────────────────────────────────────────
curl.exe http://localhost:5000/                # MLflow UI (returns HTML)

# ── Prometheus ───────────────────────────────────────────────────────────
curl.exe http://localhost:9090/-/healthy

# ── Grafana ──────────────────────────────────────────────────────────────
curl.exe http://localhost:3000/api/health

# ── MinIO ────────────────────────────────────────────────────────────────
curl.exe http://localhost:9000/minio/health/live
# Console (browser): http://localhost:9002

# ── Postgres ─────────────────────────────────────────────────────────────
docker compose -f inference_posttraining_layer/docker-compose.yml exec postgres pg_isready -U fall_user -d fall_detection

# ── All services status ───────────────────────────────────────────────────
docker compose -f inference_posttraining_layer/docker-compose.yml ps
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

## Caregiver Layer — `caregiver_layer` (Laptop 2)

```powershell
# ── Fall dashboard ────────────────────────────────────────────────────────
curl.exe http://localhost:8002/api/patients    # patient list (from postgres)
curl.exe http://localhost:8002/api/falls       # fall history (from InfluxDB)

# ── Mock app patient popup (open in browser) ──────────────────────────────
# http://localhost:8005/

# ── Postgres ──────────────────────────────────────────────────────────────
docker compose -f caregiver_layer/docker-compose.yml exec postgres pg_isready -U fall_user -d fall_detection

# ── MQTT ──────────────────────────────────────────────────────────────────
docker compose -f caregiver_layer/docker-compose.yml exec mqtt mosquitto_pub -h localhost -t test -m ping

# ── Mock app logs (check it is polling InfluxDB and calling /predict) ──────
docker compose -f caregiver_layer/docker-compose.yml logs -f mock-app

# ── All services status ────────────────────────────────────────────────────
docker compose -f caregiver_layer/docker-compose.yml ps
```

| Service | Expected response |
|---------|------------------|
| fall-dashboard `/api/patients` | `{"patients":[...]}` (empty list is fine) |
| fall-dashboard `/api/falls` | `{"falls":[...]}` (empty until a fall fires) |
| mock-app popup | Browser UI at http://localhost:8005/ |
| postgres | `localhost:5432 - accepting connections` |
| mqtt | no output = success (message published) |

---

## Cross-machine check (two-laptop test)

Run on **Laptop 2** — verify it can reach Laptop 1:

```powershell
curl.exe http://<LAPTOP1_IP>:8001/health
```

Run on **Laptop 1** — verify it can reach Laptop 2:

```powershell
curl.exe http://<LAPTOP2_IP>:8002/api/patients
```

Both should return JSON. If not, check the firewall rules (see firewall notes).
