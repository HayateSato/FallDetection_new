# Production Config Checklist — Mohammed (MCS Inference Layer)

**File to fill in:** `_6G_integration_v3_docker_mcs/.env`
Copy `.env.example` to `.env`, then change every row marked REQUIRED below.

---

## 1. Secrets — all REQUIRED

| Variable | Production value | Notes |
|----------|-----------------|-------|
| `POSTGRES_PASSWORD` | Strong random password | Used by inference-server, mlflow, db-migrate |
| `API_KEYS` | Generated secret key | Mobile app (Isa) sends this in `X-API-Key` header. Generate: `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Share with Isa after deployment. |
| `MINIO_USER` | Any username | MinIO model artifact store admin user |
| `MINIO_PASSWORD` | Strong random password | MinIO admin password |
| `GF_SECURITY_ADMIN_PASSWORD` | Strong random password | Grafana admin login |

---

## 2. Caregiver layer URLs — fill in after FOCUS DevOps deploys

These come from FOCUS DevOps. Set them once they confirm their domains.

| Variable | Production value | Notes |
|----------|-----------------|-------|
| `FALL_DASHBOARD_URL` | `https://<domain FOCUS gives you>` | Used by `server-health` to health-probe the fall-dashboard. Matches `fallDashboard.ingress.host` in FOCUS `values.yaml`. |
| `MQTT_BROKER_HOST` | `<MQTT domain FOCUS gives you>` | Used by `server-health` to probe the MQTT broker. Matches `mosquitto.ingress.host` in FOCUS `values.yaml`. |
| `MQTT_BROKER_PORT` | `443` | Production always goes through Traefik on 443. |

> If FOCUS DevOps has not deployed yet, leave these blank. The server-health dashboard will
> show the two caregiver-layer probe cards as "down" — that is acceptable. The 4 MCS-side
> probes (inference-server, postgres, mlflow, minio) will still be healthy.

---

## 3. Security hardening — recommended for production

| Variable | Current value | Recommended |
|----------|--------------|-------------|
| `CORS_ALLOWED_ORIGINS` | `*` | Exact FOCUS dashboard origin (e.g. `https://fall.focus-hospital.de`) |
| `RATE_LIMIT_PER_MINUTE` | `60` | Keep or increase — 60 req/min per IP is fine for one patient |

---

## 4. Values that do NOT need to change

| Variable | Value | Why unchanged |
|----------|-------|---------------|
| `MODEL_VERSION` | `v0` | Default model. Change only after retraining via MLflow hot-swap. |
| `ACC_SENSOR_TYPE` | `bosch` | SmarKo hardware is Bosch ACC |
| `HARDWARE_ACC_SAMPLE_RATE` | `50` | InfluxDB data confirmed at 50 Hz |
| `RESAMPLING_METHOD` | `linear` | Standard |
| `MLFLOW_REGISTERED_MODEL` | `fall-detection-xgboost` | Model registry name |
| `MQTT_ALERT_TOPIC` | `fall/alert` | Must match FOCUS k3s `fallDashboard.mqtt.alertTopic` |
| `PUBLIC_ENDPOINT_ENABLED` | `true` | Inference endpoint must be public |

---

## 5. How to run on Hetzner

```bash
cd _6G_integration_v3_docker_mcs/
cp .env.example .env
# fill in .env
docker compose up -d
```

Check inference server is up:

```bash
curl https://<your-domain>/health
# expected: {"status":"ok","model_version":"v0", ...}
```

Full deployment guide: [`_6G_integration_v3_docker_mcs/README.md`](../_6G_integration_v3_docker_mcs/README.md)

---

## 6. Cross-reference: values shared with FOCUS side

These are already set to the correct defaults. **Leave them as-is unless explicitly asked to change.**
If you change one of these, FOCUS DevOps must change their matching value at the same time — and vice versa.

| Your `.env` variable | Matching FOCUS `values.yaml` key | Action |
|---------------------|----------------------------------|--------|
| `MQTT_ALERT_TOPIC=fall/alert` | `fallDashboard.mqtt.alertTopic: fall/alert` | Leave as default |

---

## 7. What to share with others

### You → Isa (after Hetzner is live)

| What | How to get it | What Isa does with it |
|------|--------------|----------------------|
| Inference server URL | Your Hetzner domain (e.g. `https://fall-api.mcs-labs.de`) | Sets as the `/predict` endpoint in the mobile app |
| API key | `API_KEYS` value in your `.env` | Sends as `X-API-Key` header on every `/predict` call |

### You receive from FOCUS DevOps

| What | Where FOCUS sets it | You put it in |
|------|--------------------|--------------:|
| Fall-dashboard domain | `fallDashboard.ingress.host` | `FALL_DASHBOARD_URL` in your `.env` |
| MQTT broker domain | `mosquitto.ingress.host` | `MQTT_BROKER_HOST` in your `.env` |

---

## 8. All three parties must agree on

| What | Current value | Note |
|------|--------------|-------|
| MQTT alert topic | `fall/alert` | Hardcoded default — only change if all parties change together |
| MQTT possible topic | `fall/possible` | Same as above |
