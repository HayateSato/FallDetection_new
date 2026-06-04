# Fall Detection — FOCUS Caregiver Layer (k3s)

Self-contained repository for the FOCUS-hosted caregiver services.
Targets FOCUS's existing k3s cluster with Traefik.

Two services deployed: **mosquitto** (MQTT broker) + **fall-dashboard** (SSE API).
No Postgres. No mock-app (replaced by the real Isa mobile app in production).

---

## Structure

```
_6G_integration_v3_k3s/
  fall_dashboard/       Python service — MQTT subscriber + SSE feed + /api/* REST
    Dockerfile          builds the fall-dashboard image
    main.py             entry point + MQTT callback
    web.py              FastAPI routes (/api/patients, /api/falls, /api/stream SSE)
    db.py               InfluxDB queries (fall history + fall counts)
    mqtt_listener.py    FallEventBroker — paho -> asyncio SSE bridge
    patient_store.py    SQLite patient store (patient_id, MAC address)
    dashboard/          local-test HTML/JS (index.html, app.js, style.css)
    requirements.txt
  ml_pipeline/          needed by fall_dashboard for InfluxDB client
  config/               hardware config (needed by ml_pipeline)
  helm/
    caregiver/          k3s Helm chart — all k8s manifests + scripts
      values.yaml       <- fill in CHANGE_ME values before deploying
      templates/        Deployments, Services, PVCs, IngressRoutes, Secrets
      extras/
        traefik-mqtts-entrypoint.yaml  <- apply once before chart install
      build.ps1         build + push fall-dashboard image
      install.ps1       helm upgrade --install
      test.ps1          7-probe smoke test
      teardown.ps1      uninstall + delete namespace
      README.md         full deployment guide
```

---

## Quick start

### 1. One-time cluster setup (FOCUS DevOps, done once)

```powershell
# Add MQTT port 8883 to Traefik
kubectl apply -f helm/extras/traefik-mqtts-entrypoint.yaml

# Create namespace + registry pull secret
kubectl create namespace fall-dashboard
kubectl create secret docker-registry mcs-labs `
    --docker-server=registry-smarko-health.de `
    --docker-username=<user> `
    --docker-password=<pass> `
    --namespace fall-dashboard
```

### 2. Fill in values.yaml

Open [helm/values.yaml](helm/values.yaml) and replace every `CHANGE_ME`.

### 3. Build image and deploy

Run from this directory as working directory:

```powershell
.\helm\caregiver\build.ps1      # build + push fall-dashboard image
.\helm\caregiver\install.ps1    # helm upgrade --install
.\helm\caregiver\test.ps1       # smoke test (expected: 7/7 PASS)
```

---

## Connection addresses after deployment

| Client | Protocol | Address |
|---|---|---|
| Mobile app (Isa) | MQTTS | `mqtts://<FOCUS-server-IP>:8883` |
| Flutter dashboard | HTTPS | `https://<ingress.host>/api/stream` |
| fall-dashboard -> mosquitto | MQTT (internal) | `mosquitto:1883` |
| fall-dashboard -> InfluxDB | HTTPS | `https://influxdb.xxx.e-healthservice.de` |

## MQTT topics

| Topic | Publisher | When |
|---|---|---|
| `fall/possible/<patient_id>` | Mobile app | Immediately on fall detection |
| `fall/alert/<patient_id>` | Mobile app | After patient confirmation or 10s timeout |

---

## Related

- `_6G_Integration_v2_mqtt/` — full development repo (inference layer, Docker Compose, local dev tools)
- `_6G_Integration_v2_mqtt/__Refactoring_docs/k3s_flow_diagram.md` — architecture flow diagram
