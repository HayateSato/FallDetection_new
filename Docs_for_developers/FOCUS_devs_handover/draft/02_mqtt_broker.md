# MQTT Broker Setup

## What the Broker Is and Why It Is Here

The fall detection system needs a way for the **mobile app** to send a fall alert into your
network in real time, so that the fall dashboard/dashboard for caregiver can display it immediately.

That channel is the MQTT broker. It is a lightweight message relay — one client publishes a
message to a topic, and all clients subscribed to that topic receive it instantly.

In this system there are exactly two MQTT clients:

| Client | What it is | Role |
|--------|-----------|------|
| **MQTT Client A** | Mobile app (React Native, runs on patient's phone) | **Publishes** predicted/confirmed fall events |
| **MQTT Client B** | fall-dashboard pod (runs in your cluster) | **Subscribes** to both predicted/confirmed fall events |

```
Mobile app  --PUBLISH fall/alert/<pid>-->  [ mosquitto pod ]  --forward-->  fall-dashboard pod
(outside your cluster, over HTTPS/WSS)     (inside your cluster)           (inside your cluster)
```

The broker is deployed by the Helm chart — you do not build or configure it separately.
It runs as a standard `eclipse-mosquitto:2` container.

---

## How It Fits in Your Existing Stack

The mosquitto pod sits alongside fall-dashboard inside the `fall-dashboard` namespace.
It has no connection to your InfluxDB, patient dashboard, or any other existing service.

```
Your k3s cluster (namespace: fall-dashboard)
 ┌──────────────────────────────────────────────────────────┐
 │                                                          │
 │  [ mosquitto pod ]  <──── WebSocket (WSS :443 Traefik)── │──── Mobile app (outside)
 │        │                                                 │
 │        │ TCP :1883 (cluster-internal only)               │
 │        ▼                                                 │
 │  [ fall-dashboard pod ] ──── SSE ──────────────────────── │──── Caregiver browser (outside)
 │        │                                                 │
 │        │ HTTPS read                                      │
 │        ▼                                                 │
 │  [ your InfluxDB (F1) ] (fall_events measurement)        │
 │                                                          │
 └──────────────────────────────────────────────────────────┘
```

---

## Port Design — Two Listeners

Mosquitto runs two listeners simultaneously. This is the most important thing to understand
about the broker configuration.

```
                         mosquitto pod
                  ┌───────────────────────────┐
 fall-dashboard   │                           │
 pod              │  listener :1883  (TCP)    │  <- cluster-internal only
 (same cluster) ──► ClusterIP Service         │     fall-dashboard reaches it via
                  │                           │     DNS: mosquitto:1883
                  │  listener :9001  (WS)     │  <- WebSocket, routed through Traefik
 Mobile app ──────► Traefik :443 (WSS)        │     as WSS on port 443
 (internet,       │       ↓ strips TLS        │
  outside cluster)│  forwards to pod :9001    │
                  └───────────────────────────┘
```

| Port | Protocol | Exposed to | Used by |
|------|----------|-----------|---------|
| **1883** | TCP | Cluster-internal only (ClusterIP Service) | fall-dashboard pod — connects via `mosquitto:1883` |
| **9001** | WebSocket | External via Traefik WSS on port **443** | Mobile app — connects as `wss://<your-domain>` |

**Why two listeners?**
The mobile app is built with React Native. React Native cannot open raw TCP sockets in
standard JavaScript — port 1883 is not usable from a phone. WebSocket on port 9001 is the
only approach that works. Port 1883 remains for fall-dashboard, which is inside the cluster
and CAN use plain TCP.

**Why WSS and not plain WS?**
Production traffic goes through Traefik which terminates TLS. The mobile app connects to
`wss://<domain>:443` — Traefik decrypts it and forwards plain WebSocket to the pod on port 9001.

**Port 1883 is never exposed outside the cluster.** The OS firewall needs no extra rule.
All external traffic (mobile app) goes through Traefik on port 443, which is almost certainly
already open if you are serving other services through Traefik.

---

## MQTT Topics

Two topics flow through the broker. Both are published by the mobile app.

| Topic | When | Dashboard reaction |
|-------|------|-------------------|
| `fall/possible/<patient_id>` | Immediately on fall detection, before patient answers | Card turns pale red, shows "Possible fall" notice |
| `fall/alert/<patient_id>` | After patient answers popup — or after 10s timeout | Full caregiver alert with confirmation status |

The fall-dashboard subscribes to both `fall/possible/#` and `fall/alert/#`.
`#` is an MQTT wildcard — it matches any patient ID.

Example topics:
```
fall/possible/patient_001
fall/alert/patient_001
```

### Alert payload format (`fall/alert`)

The mobile app publishes this JSON on `fall/alert/<patient_id>`:

```json
{
  "patient_id":        "patient_001",
  "observation_id":    "550e8400-e29b-41d4-a716-446655440000",
  "fall_detected":     true,
  "patient_confirmed": "yes",
  "needs_help":        true,
  "confidence":        0.998,
  "timestamp":         "2026-06-10T14:32:00Z",
  "alert_time":        "2026-06-10T14:32:11Z",
  "model_version":     "v3"
}
```

`patient_confirmed` values: `"yes"` / `"no"` / `"not_answered"` (string on MQTT).

---

## Configuration in values_production.yaml

The broker is fully configured via `values_production.yaml`. The relevant section:

```yaml
mosquitto:
  image: eclipse-mosquitto:2        # no change needed
  port: 1883                        # internal TCP -- do not change
  wsPort: 9001                      # internal WebSocket -- do not change
  wsNodePort: ""                    # leave blank in production (Traefik handles external access)

  ingress:
    host: "CHANGE_ME"               # your MQTT subdomain, e.g. mqtt.focus-hospital.de
    certResolver: ""                # your Traefik cert resolver, e.g. "le"

  config: |
    per_listener_settings false
    allow_anonymous true            # change to false and add auth if you want credentials
    listener 1883
    listener 9001
    protocol websockets
    persistence true
    persistence_location /mosquitto/data/
    log_dest stdout
    log_type all
```

### Enabling MQTT authentication (optional but recommended for production)

By default `allow_anonymous true` is set — any client can connect without credentials.
To enable authentication:

**Step 1 — change the config block in values_production.yaml:**

```yaml
  config: |
    per_listener_settings false
    allow_anonymous false           # <-- change it to false for more security
    password_file /mosquitto/config/passwd
    listener 1883
    listener 9001
    protocol websockets
    persistence true
    persistence_location /mosquitto/data/
    log_dest stdout
    log_type all
```

**Step 2 — set credentials in values_production.yaml:**

```yaml
fallDashboard:
  mqtt:
    username: "focus_caregiver"     # shared with Isa so that he can let mobile app talk to the broker
    password: "CHANGE_ME"
```

**Step 3 — generate the password file and add it to the ConfigMap:**

The mosquitto password file must be created using the `mosquitto_passwd` utility.
You can do this from inside the running pod:

```bash
kubectl exec -n fall-dashboard deploy/mosquitto -- \
  mosquitto_passwd -c /mosquitto/config/passwd focus_caregiver
# you will be prompted for the password
```

Then restart the pod so the config is re-read:

```bash
kubectl rollout restart deployment/mosquitto -n fall-dashboard
```

> **Important:** if you set credentials here, share the same `username` and `password` with Isa
> (mobile app developer). A mismatch causes a silent connection failure — the mobile app will
> appear connected but fall alerts will never arrive. There is no error on the dashboard side.

---

## Commands — By Situation

### Check broker pod is running

```bash
kubectl get pods -n fall-dashboard -l app=mosquitto
# Expected: STATUS = Running, READY = 1/1
```

### Check broker logs (live)

```bash
kubectl logs -n fall-dashboard -l app=mosquitto -f
```

### Check last 100 lines of broker log

```bash
kubectl logs -n fall-dashboard -l app=mosquitto --tail=100
```

### Filter logs for a specific patient

```bash
kubectl logs -n fall-dashboard -l app=mosquitto | grep "patient_001"
```

### Verify fall-dashboard is connected to the broker

```bash
# Look for the dashboard's SUBSCRIBE lines in broker log
kubectl logs -n fall-dashboard -l app=mosquitto | grep "SUBSCRIBE"
# Expected:
#   SUBSCRIBE from fall-detection-caregiver: fall/alert/#  (QoS 0)
#   SUBSCRIBE from fall-detection-caregiver: fall/possible/#  (QoS 0)
```

If you see no SUBSCRIBE lines, the fall-dashboard has not connected to the broker yet.
Check fall-dashboard logs: `kubectl logs -n fall-dashboard -l app=fall-dashboard`.

---

## Reading the Broker Logs

Every log line is written from the **broker's point of view**:

- `Received PUBLISH from X` — the broker received a message from client X
- `Sending PUBLISH to Y` — the broker forwarded it to subscriber Y

One fall alert produces two lines (Received + Sending). If you see `Received` with no
matching `Sending`, no subscriber was connected at that moment — the alert was dropped.

```
# Normal fall alert flow in the log:
Received PUBLISH from mobile-app-client 'fall/possible/patient_001' (N bytes)
Sending  PUBLISH to   fall-detection-caregiver 'fall/possible/patient_001' (N bytes)

Received PUBLISH from mobile-app-client 'fall/alert/patient_001' (N bytes)
Sending  PUBLISH to   fall-detection-caregiver 'fall/alert/patient_001' (N bytes)
```

### Lines you can ignore (normal noise)

| Line | Meaning |
|------|---------|
| `PINGREQ` / `PINGRESP` | Keepalive heartbeat every ~60s — connection alive only, no data |
| `CONNECT` / `CONNACK (0,0)` | A client connected successfully |
| `DISCONNECT` / `connection closed by client` | Normal reconnect cycle |
| `SUBSCRIBE` / `SUBACK` | Client registered for a topic |
| `No will message specified` | Client set no "last will" message — harmless |

### Timestamps in the log

Log lines start with a Unix epoch timestamp (seconds since 1970), not a line number.
To convert one:

```bash
date -d @1780574956   # Linux
```

---

## QoS — Brief Note

The current deployment uses QoS 0 (fire-and-forget) on both topics. This means:

- The broker does not retry if the fall-dashboard is momentarily disconnected.
- An alert published during a brief reconnect gap is silently dropped.

`fall/possible` alerts are pre-confirmation and transient — QoS 0 is acceptable.
For `fall/alert` (confirmed falls), upgrading to **QoS 1** is recommended for production.
QoS is set by the clients (mobile app + fall-dashboard code), not by the broker config —
the broker already supports all QoS levels with no change needed.

If you observe dropped confirmed alerts, contact MCS to upgrade the client QoS settings.

---

## Summary — Port Reference

| Port | Where | Protocol | Reachable from | Purpose |
|------|-------|----------|---------------|---------|
| 1883 | mosquitto pod | TCP | fall-dashboard pod only (ClusterIP) | fall-dashboard ↔ broker |
| 9001 | mosquitto pod | WebSocket | Traefik (internal forwarding) | receives WS traffic from Traefik |
| 443 | Traefik (your cluster) | HTTPS **and** WSS | mobile app (internet) | broker public entry point — same port as your existing HTTPS services |

**OS firewall: no new rules needed.**
If port 443 is already open for your existing services (patient dashboard, caregiver webapp), WSS traffic for the MQTT broker flows through the same port automatically — nothing extra to open.
WSS (WebSocket Secure) is not a separate protocol at the TCP level — it is a standard HTTPS
connection that Traefik upgrades to a WebSocket once the `Upgrade: websocket` handshake is
complete. The OS firewall only sees TCP port 443 in both cases. 

Traefik distinguishes between your existing HTTPS routes and the new WSS broker route by
**hostname** (`Host` header), not by port. Each service gets its own subdomain:

```
https://fall.focus-hospital.de     → fall-dashboard pod   (HTTPS, your new service)
wss://mqtt.focus-hospital.de       → mosquitto pod :9001  (WSS, same TCP port 443)
https://patient.focus-hospital.de  → patient dashboard    (HTTPS, your existing service)
```

All three share the same Traefik entrypoint on port 443. Only the hostname differs.
