# Hetzner MQTT Fix — Action Guide for Mohammed

> **STATUS: RESOLVED (2026-06-23)**
>
> MQTT WebSocket connectivity is confirmed working on port 443 via Postman.
> No firewall changes or port 8883 needed — the current cluster state is correct.
>
> **What was the actual problem:** `mosquitto_sub` on Windows is incompatible with
> Traefik v3's TLS renegotiation behavior. It was giving false "protocol error" results.
> Postman (and React Native MQTT.js) handle TLS renegotiation correctly and connect fine.
>
> **Confirmed working:**
> - Postman connects to `wss://fall-mqtt.smarko-health.de` (port 443) ✓
> - Publish/subscribe round-trip works in Postman ✓
> - mosquitto pod logs show `postman-mqtt-client` connected and doing PINGREQ/PINGRESP ✓
> - `fall-detection-caregiver` (fall-dashboard) connected and healthy ✓
>
> **For future testing:** use Postman or MQTT Explorer — NOT `mosquitto_sub` on Windows.
> Full debug log: `_6G_integration_v3_k3s/ai_docs/hetzner_mqtt_debug_log.md`

---

# STEP 0 — Open firewall port (Hetzner Cloud console)

```
Add firewall rule on server 5.75.255.114:
  Protocol: TCP
  Port:      8883
  Direction: Inbound
```

Do this BEFORE running the kubectl commands below.

---

# STEP 1 — Add port 8883 entrypoint to Traefik

Run in Portainer kubectl shell:

```bash
kubectl apply -f - <<'EOF'
apiVersion: helm.cattle.io/v1
kind: HelmChartConfig
metadata:
  name: traefik
  namespace: kube-system
spec:
  valuesContent: |-
    persistence:
      enabled: true
      size: 128Mi
      path: /data
    additionalArguments:
      - "--certificatesresolvers.le.acme.email=mcsinfomail@gmail.com"
      - "--certificatesresolvers.le.acme.storage=/data/acme.json"
      - "--certificatesresolvers.le.acme.httpchallenge=true"
      - "--certificatesresolvers.le.acme.httpchallenge.entrypoint=web"
    ports:
      mqttwss:
        port: 8883
        expose:
          default: true
        exposedPort: 8883
        protocol: TCP
EOF
```

Wait for Traefik to restart before continuing:

```bash
kubectl rollout status deployment/traefik -n kube-system
```

---

# STEP 2 — Point MQTT routing to the new entrypoint

```bash
kubectl patch ingressroutetcp mosquitto-ws-tcp -n fall-dashboard --type=json \
  -p '[{"op":"replace","path":"/spec/entryPoints/0","value":"mqttwss"}]'
```

---

# STEP 3 — Verify

```bash
kubectl get ingressroutetcp mosquitto-ws-tcp -n fall-dashboard -o yaml
```

Look for this in the output:
```yaml
entryPoints:
  - mqttwss
```

---

# STEP 4 — Test

```bash
mosquitto_sub -L "wss://fall-mqtt.smarko-health.de:8883/#" --insecure -v
```

Expected: command hangs waiting for messages = **connected successfully**.

`--insecure` is needed for the first few minutes while Let's Encrypt issues the cert for
port 8883. Try without it after a few minutes — if it connects, the cert is ready.

End-to-end test (run in a second terminal while subscribe is running):

```bash
mosquitto_pub -L "wss://fall-mqtt.smarko-health.de:8883/fall/alert/test" --insecure -m "hello"
```

The subscribe terminal should print the message.

---

## Notes

- The `IngressRouteTCP mosquitto-ws-tcp` resource already exists in the cluster (created
  during debugging). Step 2 patches it in place — no need to create or delete anything.
- The old HTTP IngressRoute for mosquitto (`mosquitto-ws`) has already been deleted.
- Port 443 is untouched — fall-dashboard HTTPS continues to work as before.
