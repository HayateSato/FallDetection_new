# helm/ — FOCUS Caregiver Layer (k3s)

Helm chart for the FOCUS-hosted caregiver services: MQTT broker (mosquitto) and
fall-dashboard. Targets FOCUS's existing k3s cluster with Traefik.

For deployment steps, configuration guide, and verification commands, see:
**`focus_devops_handover.md`** (included in this package).

---

## What this chart deploys

| Resource | Type | Purpose |
|---|---|---|
| mosquitto | Deployment + PVC | MQTT broker — receives fall alerts from mobile app |
| mosquitto | ClusterIP Service | Internal: TCP :1883 (fall-dashboard) + WS :9001 (Traefik forwards here) |
| mosquitto | IngressRoute | WSS on port 443 via Traefik — mobile app connects here |
| fall-dashboard | Deployment + PVC | Live alert UI + fall history + patient management. SQLite patient store on PVC. |
| fall-dashboard | ClusterIP Service | Internal: port 8002 |
| fall-dashboard | IngressRoute | HTTPS on port 443 via Traefik — caregiver opens this in the browser |

## MQTT topics

| Topic | Publisher | When |
|---|---|---|
| `fall/possible/<patient_id>` | Mobile app | Immediately on fall detection, before patient confirms |
| `fall/alert/<patient_id>` | Mobile app | After patient confirmation or 10s timeout |

fall-dashboard subscribes to both `fall/possible/#` and `fall/alert/#`.

## Files in this package

```
helm/
  Chart.yaml
  values.yaml                           default values (reference only)
  values_production_focus.yaml          fill in 3 CHANGE_ME fields before deploying
  templates/
    _helpers.tpl
    mosquitto-configmap.yaml
    mosquitto-pvc.yaml
    mosquitto-deployment.yaml
    mosquitto-service.yaml
    mosquitto-ingressroute.yaml         WSS route for mobile app (port 443)
    fall-dashboard-secret.yaml          InfluxDB token, MQTT credentials
    fall-dashboard-configmap.yaml       non-secret env vars
    fall-dashboard-pvc.yaml             SQLite patient store persistence
    fall-dashboard-deployment.yaml
    fall-dashboard-service.yaml
    fall-dashboard-ingressroute.yaml    HTTPS route for caregiver browser
focus_devops_handover.md               deployment guide (start here)
caregiver_user_guide.docx              end-user guide for caregiver staff
```
