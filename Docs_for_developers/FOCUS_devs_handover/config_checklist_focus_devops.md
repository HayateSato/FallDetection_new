# Production Config Checklist — FOCUS DevOps (Caregiver Layer)

**File to fill in:** `_6G_integration_v3_k3s/helm/values_production.yaml`
Edit this file, then run `helm install` / `helm upgrade`.

---

## 1. Registry and image pull — REQUIRED

| Key | Production value | Notes |
|-----|-----------------|-------|
| `imagePullSecret` | `mcs-labs` | Pull secret name for `registry-smarko-health.de`. Create with `kubectl create secret docker-registry mcs-labs ...` — get credentials from Mohammed. |
| `fallDashboard.image` | `registry-smarko-health.de/fall-detection/fall-dashboard:latest` | Production image pushed by Mohammed |
| `fallDashboard.imagePullPolicy` | `Always` | Always pull from registry in production |

---

## 2. NodePorts — blank both for production

NodePorts were only needed for local two-laptop testing. Traefik handles all external access in production.

| Key | Production value | Notes |
|-----|-----------------|-------|
| `mosquitto.wsNodePort` | `""` (blank) | Blank = ClusterIP only; Traefik IngressRoute handles external WSS |
| `fallDashboard.httpNodePort` | `""` (blank) | Blank = ClusterIP only; Traefik IngressRoute handles external HTTPS |

---

## 3. Mosquitto ingress (Traefik) — REQUIRED

| Key | Production value | Notes |
|-----|-----------------|-------|
| `mosquitto.ingress.host` | e.g. `mqtt.focus-hospital.de` | Domain the real mobile app connects to via WSS. Share this with Isa. |
| `mosquitto.ingress.certResolver` | e.g. `le` | Traefik Let's Encrypt resolver name used in your cluster. Check your existing Traefik config for the resolver name. |

---

## 4. Fall-dashboard ingress (Traefik) — REQUIRED

| Key | Production value | Notes |
|-----|-----------------|-------|
| `fallDashboard.ingress.host` | e.g. `fall.focus-hospital.de` | Domain Flutter dashboard calls for `/api/falls`, `/api/stream` SSE. Share this with Mohammed and Isa. |
| `fallDashboard.ingress.certResolver` | e.g. `le` | Same resolver name as above |

---

## 5. InfluxDB — REQUIRED (your FOCUS InfluxDB instance)

The fall-dashboard reads fall history from your InfluxDB. These must match what the mobile app writes to.

| Key | Production value |
|-----|-----------------|
| `fallDashboard.influxdb.url` | Your InfluxDB URL |
| `fallDashboard.influxdb.org` | Your InfluxDB organisation name |
| `fallDashboard.influxdb.bucket` | Bucket name where raw sensor data is stored |
| `fallDashboard.influxdb.fallEventsBucket` | Bucket where `fall_events` are written (often the same as `bucket`) |
| `fallDashboard.influxdb.token` | InfluxDB token with read access to both buckets |

Share the InfluxDB write credentials (URL, token, org, bucket) with Isa — the mobile app
writes `fall_events` points directly to your InfluxDB after each patient confirmation popup.

---

## 6. MQTT auth — set if broker requires authentication

| Key | Production value | Notes |
|-----|-----------------|-------|
| `fallDashboard.mqtt.username` | broker username | Leave blank if `allow_anonymous true` stays in mosquitto config |
| `fallDashboard.mqtt.password` | broker password | Leave blank if anonymous allowed |

> **Important:** If you set MQTT credentials here, you must share the same username and password
> with Isa. The mobile app connects to the broker from outside the cluster and must use the
> **exact same** credentials. If these change, inform Isa immediately — a mismatch causes silent
> connection failure (falls never appear on the dashboard).

---

## 7. Patients — initial seed values (optional)

Patient management is now **dynamic from the UI** — caregivers can add and delete patients
directly from the fall-dashboard browser without editing `values.yaml` or restarting the pod.
The patient list is stored in SQLite on a PVC and survives pod restarts.

These values are only used to pre-seed patients on first install. Leave blank if you prefer
to add patients manually from the UI after deployment.

| Key | Notes |
|-----|-------|
| `fallDashboard.patientIds` | Comma-separated patient IDs. Must match the patient IDs in the mobile app and InfluxDB tag `patient_id`. |
| `fallDashboard.macIds` | Comma-separated MAC addresses. Positional 1:1 mapping to `patientIds`. |

---

## 8. Values that do NOT need to change

| Key | Value | Why unchanged |
|-----|-------|---------------|
| `namespace` | `fall-dashboard` | Standard namespace name |
| `mosquitto.port` | `1883` | Internal TCP port (cluster-only, never changes) |
| `mosquitto.wsPort` | `9001` | Internal WebSocket port (Traefik forwards to this) |
| `mosquitto.image` | `eclipse-mosquitto:2` | Standard image, no custom build |
| `fallDashboard.replicas` | `1` | **Must stay 1** — SSE fan-out is in-process; multiple replicas break live alerts |
| `fallDashboard.port` | `8002` | Internal pod port |
| `fallDashboard.mqtt.alertTopic` | `fall/alert` | Must match MCS `.env` `MQTT_ALERT_TOPIC` |
| `fallDashboard.mqtt.possibleTopic` | `fall/possible` | Pre-confirmation alert topic |
| `mosquitto.persistence.size` | `500Mi` | Sufficient for MQTT broker storage |
| `fallDashboard.persistence.size` | `500Mi` | Sufficient for SQLite patient store |
| `mosquitto.persistence.storageClass` | `""` | Blank = k3s default (`local-path`) |
| `fallDashboard.persistence.storageClass` | `""` | Blank = k3s default (`local-path`) |
| `registry` | `registry-smarko-health.de` | Already set to production registry |

---

## 9. How to install on FOCUS k3s

```bash
# from _6G_integration_v3_k3s/ as working directory

# First time
helm install caregiver helm/ \
  --namespace fall-dashboard \
  --create-namespace \
  --values helm/values_production.yaml

# Updates (after values change or new image)
helm upgrade caregiver helm/ \
  --namespace fall-dashboard \
  --values helm/values_production.yaml
```

Check pods are running:

```bash
kubectl get pods -n fall-dashboard
# Expected:
#   mosquitto-xxxx        1/1   Running
#   fall-dashboard-xxxx   1/1   Running
```

Check fall-dashboard is reachable (from inside the cluster):

```bash
kubectl exec -n fall-dashboard deploy/fall-dashboard -- \
  wget -qO- http://localhost:8002/api/patients
```

Full deployment guide: [`_6G_integration_v3_k3s/README.md`](../_6G_integration_v3_k3s/README.md)

---

## 10. Cross-reference: values shared with MCS side

These are already set to the correct defaults. **Leave them as-is unless explicitly asked to change.**
If you change one of these, Mohammed must change his matching value at the same time — and vice versa.

| Your `values.yaml` key | Matching MCS `.env` variable | Action |
|------------------------|------------------------------|--------|
| `fallDashboard.mqtt.alertTopic: fall/alert` | `MQTT_ALERT_TOPIC=fall/alert` | Leave as default |

---

## 11. What to share with others

### You → Isa (mobile app) — share after deployment is live

| What | Where you set it | What Isa does with it |
|------|-----------------|----------------------|
| MQTT broker domain | `mosquitto.ingress.host` | Mobile app connects as `wss://<host>:443` to publish fall alerts |
| MQTT username | `fallDashboard.mqtt.username` | Mobile app authenticates with the broker |
| MQTT password | `fallDashboard.mqtt.password` | Mobile app authenticates with the broker |
| InfluxDB write credentials | Your FOCUS InfluxDB (URL, token, org, bucket) | Mobile app writes `fall_events` point after each confirmation popup |
| Fall-dashboard domain | `fallDashboard.ingress.host` | Flutter dashboard uses this for `/api/stream` SSE and `/api/falls` |

### You → Mohammed (MCS) — share after deployment is live

| What | Where you set it | What Mohammed does with it |
|------|-----------------|--------------------------|
| Fall-dashboard domain | `fallDashboard.ingress.host` | Sets `FALL_DASHBOARD_URL` in MCS `.env` so server-health can probe your service |
| MQTT broker domain | `mosquitto.ingress.host` | Sets `MQTT_BROKER_HOST` in MCS `.env` so server-health can probe the broker |

### You receive from Mohammed (before you can install)

| What | What you need it for |
|------|---------------------|
| Registry credentials for `registry-smarko-health.de` | `kubectl create secret docker-registry mcs-labs ...` |
| Confirmation that fall-dashboard image is pushed | Before running `helm install` |

---

## 12. All three parties must agree on

| What | Current value | Note |
|------|--------------|-------|
| MQTT alert topic | `fall/alert` | Hardcoded default — only change if all parties change together |
| MQTT possible topic | `fall/possible` | Same as above |
