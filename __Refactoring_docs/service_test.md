`# ── Python services ─────────────────────────────────────────────────────
curl.exe http://localhost:8001/health          # inference-server
curl.exe http://localhost:8002/api/patients    # fall-dashboard
curl.exe http://localhost:8004/                # ml-dashboard
curl.exe http://localhost:8006/                # server-health

# ── MLflow ──────────────────────────────────────────────────────────────
curl.exe http://localhost:5000/                # MLflow UI (returns HTML)

# ── Prometheus ──────────────────────────────────────────────────────────
curl.exe http://localhost:9090/-/healthy       # Prometheus

# ── Grafana ─────────────────────────────────────────────────────────────
curl.exe http://localhost:3000/api/health      # Grafana

# ── MinIO ───────────────────────────────────────────────────────────────
curl.exe http://localhost:9000/minio/health/live   # MinIO S3 API
# Console (browser): http://localhost:9002

# ── Postgres (not HTTP — use docker exec) ───────────────────────────────
docker compose -f deploy/docker-compose.yml exec postgres pg_isready -U fall_user -d fall_detection

# ── MQTT (not HTTP — use docker exec) ───────────────────────────────────
docker compose -f deploy/docker-compose.yml exec mqtt mosquitto_pub -h localhost -t test -m ping`

**Expected results:**

| Service | Expected response |
| --- | --- |
| inference-server | `{"status":"ok","model_version":"v0",...}` |
| fall-dashboard | `{"patients":[...]}` (empty list is fine) |
| ml-dashboard | HTML page |
| server-health | HTML page |
| mlflow | HTML page |
| prometheus | `Prometheus Server is Healthy.` |
| grafana | `{"commit":"...","database":"ok",...}` |
| minio | HTTP 200 (empty body) |
| postgres | `localhost:5432 - accepting connections` |
| mqtt | no output = success (message published) |