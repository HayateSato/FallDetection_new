# Debug Guide

## Quick Checks — Start Here

```bash
# Are both pods running?
kubectl get pods -n fall-dashboard

# Any recent errors?
kubectl logs -n fall-dashboard -l app=fall-dashboard --tail=50
kubectl logs -n fall-dashboard -l app=mosquitto --tail=50

# Run the smoke test (7 probes)
bash helm/test.sh
```

---

## Issue: Pod stuck in CrashLoopBackOff or Error

```bash
# See why the pod is failing
kubectl describe pod -n fall-dashboard -l app=fall-dashboard
kubectl logs -n fall-dashboard -l app=fall-dashboard --previous
```

Common causes:

| Symptom in logs | Cause | Fix |
|-----------------|-------|-----|
| `INFLUXDB_URL not set` or InfluxDB connection refused at startup | InfluxDB env vars missing or wrong | Check `fallDashboard.influxdb.*` in `values_production.yaml`, then `helm upgrade` |
| `MQTT_BROKER_HOST not set` | MQTT env var missing | Set `MQTT_BROKER_HOST=mosquitto` in values (already default), then `helm upgrade` |
| `ModuleNotFoundError` | Image is outdated or build failed | Ask Mohammed to rebuild and push the image, then `helm upgrade` |
| `OOMKilled` | Pod hit memory limit | Increase `resources.limits.memory` in values |

---

## Issue: Live stream shows "Disconnected" in the browser

The browser SSE connection to `/api/stream` is broken. This is usually a network
or Traefik routing issue, not a fall-dashboard pod issue.

```bash
# 1. Confirm the pod itself is running and healthy
kubectl get pods -n fall-dashboard -l app=fall-dashboard

# 2. Test /api/stream from inside the cluster via port-forward
kubectl port-forward -n fall-dashboard svc/fall-dashboard 18002:8002 &
curl -N http://localhost:18002/api/stream
# Expected first line: event: connected
# Kill with Ctrl+C when done

# 3. Test the public endpoint
curl -N https://fall.focus-hospital.de/api/stream
# Expected first line: event: connected
# If you get a 502/504: Traefik is not reaching the pod (check IngressRoute)
# If connection refused: TLS/DNS issue
```

Check IngressRoute is present:

```bash
kubectl get ingressroute fall-dashboard -n fall-dashboard -o yaml
# Confirm spec.routes[0].match has the correct Host value
```

---

## Issue: No fall alerts appearing on the dashboard

The dashboard shows "Connected" but cards never turn red when a fall happens.
Work through this in order:

**Step 1 — Is the MQTT broker receiving messages from the mobile app?**

```bash
kubectl logs -n fall-dashboard -l app=mosquitto | grep "PUBLISH"
# If the mobile app is sending: you should see "Received PUBLISH from ..."
# If nothing: the mobile app is not connecting or not publishing
```

**Step 2 — Is fall-dashboard subscribed to the broker?**

```bash
kubectl logs -n fall-dashboard -l app=mosquitto | grep "SUBSCRIBE"
# Expected:
#   SUBSCRIBE from fall-detection-caregiver: fall/alert/#
#   SUBSCRIBE from fall-detection-caregiver: fall/possible/#
# If missing: fall-dashboard is not connected to the broker
```

**Step 3 — If fall-dashboard is not subscribed, check its MQTT connection**

```bash
kubectl logs -n fall-dashboard -l app=fall-dashboard | grep -i mqtt
# Look for: "FallEventBroker connecting to MQTT mosquitto:1883"
# Bad: "MQTT broker unreachable after 5 attempts"
```

If the broker is unreachable from fall-dashboard, verify:

```bash
# Check MQTT_BROKER_HOST is set to "mosquitto" (not an IP)
kubectl exec -n fall-dashboard deploy/fall-dashboard -- env | grep MQTT
# Expected: MQTT_BROKER_HOST=mosquitto

# Test TCP connectivity to the broker from inside fall-dashboard pod
kubectl exec -n fall-dashboard deploy/fall-dashboard -- \
  wget -qO- http://mosquitto:1883 2>&1 | head -5
# Connection refused = broker is up but port is TCP (expected for MQTT)
# Network error / timeout = DNS resolution or network policy issue
```

**Step 4 — Check the alert conditions**

Not every fall event triggers a visible alert. The dashboard only turns red when:
- `patient_confirmed = not_answered` (patient did not respond in 10s), **or**
- `patient_confirmed = yes` AND `needs_help = true`

If the patient responds "no" or "yes but I'm fine", the fall is stored in InfluxDB
but the caregiver card does **not** activate. Verify by checking fall-dashboard logs:

```bash
kubectl logs -n fall-dashboard -l app=fall-dashboard | grep "Fall recorded"
# "Fall recorded (no caregiver alert)" = event received but alert suppressed by design
# "Fall ALERT -> caregiver" = alert was sent to the SSE stream
```

---

## Issue: Fall history tab is empty

The history tab reads from InfluxDB. No data means one of:

1. **Mobile app is not writing to InfluxDB** — Open InfluxDB and check if the
   `fall_events` measurement exists. 

2. **Wrong bucket configured** — check `fallDashboard.influxdb.fallEventsBucket` in
   `values_production.yaml`. Must match the bucket the mobile app writes to.

3. **InfluxDB credentials wrong** — the fall-dashboard uses a read token. Verify it
   has read access to the `fall_events` measurement:

```bash
kubectl logs -n fall-dashboard -l app=fall-dashboard | grep -i influx
# Look for: "InfluxDB fall count query failed"
```

4. **No falls have happened yet** — the history is empty by design if no fall events
   have been written.

---

## Issue: Patient card not appearing after adding via UI

```bash
# Check the patient was saved to SQLite
kubectl exec -n fall-dashboard deploy/fall-dashboard -- \
  wget -qO- http://localhost:8002/api/patients
# The patient should appear in the JSON response
```

If the patient is in the API response but not in the browser, do a hard refresh
(`Ctrl+Shift+R`) — the browser may be caching the old page.

---

## Issue: ErrImagePull / ImagePullBackOff

```bash
kubectl describe pod -n fall-dashboard -l app=fall-dashboard | grep -A 10 "Events:"
```


---

## Issue: helm upgrade has no effect

K3s does not watch `values_production.yaml` for changes. Every change — including
image updates — requires an explicit `helm upgrade`:

```bash
bash helm/install.sh
# or directly:
helm upgrade caregiver helm/ \
    --namespace fall-dashboard \
    --values helm/values_production.yaml
```

If pods still do not restart after upgrade (e.g. image tag is `latest` and you want
to force a fresh pull):

```bash
kubectl rollout restart deployment/fall-dashboard -n fall-dashboard
kubectl rollout restart deployment/mosquitto -n fall-dashboard
```

---

## Issue: fall-dashboard starts but MQTT reconnects repeatedly

```bash
kubectl logs -n fall-dashboard -l app=fall-dashboard | grep -i "attempt\|reconnect\|disconnect"
```

Repeated reconnect cycles usually mean the broker is up but authentication is wrong.
Check credentials match on both sides:

```bash
# What credentials is fall-dashboard using?
kubectl exec -n fall-dashboard deploy/fall-dashboard -- env | grep MQTT
# MQTT_USERNAME and MQTT_PASSWORD must match what's in mosquitto.conf password_file
```

If you recently added authentication to the broker, remember to share the same
credentials with Isa (mobile app) — see [03_k3s_values_and_secrets.md](03_k3s_values_and_secrets.md).

---

## Useful Commands Reference

```bash
# Pod status
kubectl get pods -n fall-dashboard
kubectl get pods -n fall-dashboard -o wide           # includes node + IP

# Logs
kubectl logs -n fall-dashboard -l app=fall-dashboard -f          # follow live
kubectl logs -n fall-dashboard -l app=fall-dashboard --tail=100
kubectl logs -n fall-dashboard -l app=fall-dashboard --previous  # last crashed container
kubectl logs -n fall-dashboard -l app=mosquitto -f

# Describe (events, image, resource limits)
kubectl describe pod -n fall-dashboard -l app=fall-dashboard
kubectl describe pod -n fall-dashboard -l app=mosquitto

# Port-forward (access pod directly without Traefik)
kubectl port-forward -n fall-dashboard svc/fall-dashboard 18002:8002
# Then open: http://localhost:18002

# Exec into pod
kubectl exec -it -n fall-dashboard deploy/fall-dashboard -- sh
kubectl exec -it -n fall-dashboard deploy/mosquitto -- sh

# ConfigMap (verify mosquitto.conf was applied)
kubectl get configmap mosquitto-config -n fall-dashboard -o yaml

# Secret (verify InfluxDB token + MQTT credentials are present)
kubectl get secret fall-dashboard-secret -n fall-dashboard -o yaml
# Values are base64 encoded: echo "<value>" | base64 -d

# Restart pods
kubectl rollout restart deployment/fall-dashboard -n fall-dashboard
kubectl rollout restart deployment/mosquitto -n fall-dashboard

# Apply values change
helm upgrade caregiver helm/ --namespace fall-dashboard --values helm/values_production.yaml
```
