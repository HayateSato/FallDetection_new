# K3s Flow Diagram — FOCUS Production Architecture

Two separate entry channels into the FOCUS clusters: one TCP tunnel for MQTT (port 8883),
one HTTPS tunnel for everything else (port 443). InfluxDB lives in FOCUS's existing cluster
and is reached by both the mobile app and fall-dashboard over standard HTTPS.

---

## Full flow diagram

```
                                      MCS / Hetzner
                                    ┌───────────────────┐
                              :443  │  inference-server  │
                         ┌─────────►  ml-dashboard       │
                         │         │  postgres           │
                         │         └───────────────────--┘
                         │ HTTPS POST /predict
                         │ HTTPS POST /inference/{id}/confirm
                         │
Mobile app               │
(phone, on any network)  │
         │               │
         │               │         FOCUS — new k3s cluster (caregiver layer)
         │               │        ┌─────────────────────────────────────────┐
         │               │        │  Traefik                                │
         │  mqtts://:8883│        │  ┌──────────┐                           │
         ├───────────────┼───────►│  │TCP :8883 ├──► mosquitto pod          │
         │               │        │  └──────────┘         │  push           │
         │               │        │                        ▼  (internal DNS) │
         │               │        │               fall-dashboard pod ◄───────┼────┐
         │               │        │                        │                 │    │
         │               │        │  ┌──────────┐         │  HTTPS GET      │    │ cross-cluster
         │               │        │  │HTTP :443 │◄────────┘  (read falls,   │    │ HTTPS :443
         │               │        │  └──────────┘            fall counts)   │    │ /api/stream
         │               │        └──────────────────────────────────────────┘    │ /api/falls
         │               │                                  │                     │ /api/patients
         │               │                                  │ HTTPS GET           │
         │               │                                  │ /api/v2/query       │
         │               │                                  │ (cross-cluster)     │
         │  HTTPS POST   │                                  ▼                     │
         │  /api/v2/write│        FOCUS — existing k3s cluster (InfluxDB + Flutter dashboard)
         └───────────────┼───────►┌──────────────────────────────────────────────────────────┐
                         │        │  Traefik :443 (already exists)                           │
     fall_events point   │        │  ┌──────────────────────┐   ┌─────────────────────────┐ │
     (line protocol)     │        │  │ influxdb.xxx....de   │   │  Flutter dashboard pod  │ │
                         │        │  │                      │   │  (caregiver-facing UI)  ├─┘
                         │        │  │  InfluxDB pod        │   │  FOCUS DevOps team      │  │
                         │        │  │  fd_test bucket      │   └─────────────────────────┘  │
                         │        │  └──────────────────────┘                                │
                         │        └──────────────────────────────────────────────────────────┘
```

---

## Channel 1 — MQTT (TCP tunnel, port 8883)

MQTT is a raw TCP protocol, not HTTP. It does not use URL paths.
The mobile app connection string is:

```
mqtts://xxx.healthservice.de:8883        (TLS, production)
mqtt://xxx.healthservice.de:1883         (plain, internal testing only)
```

**Flow:**

1. Mobile app opens a persistent TCP connection to `mqtts://xxx.healthservice.de:8883`
2. Traefik receives raw TCP on port 8883 — this is a separate TCP IngressRouteTCP,
   completely independent from the HTTP routing table
3. Traefik forwards to the mosquitto pod inside the new k3s cluster on port 1883
4. Mobile app publishes `fall/possible/<patient_id>` (immediately on fall detection)
   and `fall/alert/<patient_id>` (after patient confirms or 10s timeout)
5. mosquitto delivers the message to fall-dashboard, which has an active subscription
   (`fall/possible/#` and `fall/alert/#`) opened at startup via internal cluster DNS:
   `mosquitto.namespace.svc.cluster.local:1883`

**Key point — asymmetric path:**

| Direction | Path | Goes through Traefik? |
|---|---|---|
| Mobile app -> broker | External IP:8883 -> Traefik TCP -> mosquitto pod | Yes |
| Broker -> fall-dashboard | Internal k8s DNS (mosquitto:1883) | No — internal only |

fall-dashboard connects to the broker using the internal cluster DNS address, never the
external hostname or port 8883. This gives a fast, local connection with no TLS overhead
for the subscriber, while the publisher (mobile app) still reaches it securely from outside.

**Traefik config needed (one-time, in k3s Traefik Helm values):**

```yaml
# Add a new entrypoint so Traefik also listens on port 8883
additionalArguments:
  - "--entrypoints.mqtts.address=:8883"
```

```yaml
# IngressRouteTCP — forward port 8883 to the mosquitto service
apiVersion: traefik.io/v1alpha1
kind: IngressRouteTCP
metadata:
  name: mqtt-broker
spec:
  entryPoints: [mqtts]
  routes:
    - match: HostSNI(`*`)
      services:
        - name: mosquitto
          port: 1883
```

---

## Channel 2 — HTTPS (port 443, two directions for InfluxDB)

InfluxDB exposes a standard HTTP REST API. Both the mobile app (write) and fall-dashboard
(read) reach it over normal HTTPS on port 443 — the same entry point FOCUS already uses
for all their existing services. No new ports, no TCP tunnel, no changes to the existing cluster.

### Mobile app writing fall_events to InfluxDB

After patient confirmation, the mobile app sends one HTTP POST per detected fall:

```
POST https://influxdb.xxx.e-healthservice.de/api/v2/write
     ?org=FOCUS&bucket=fd_test&precision=ns
Authorization: Token <influx_token>
Content-Type: text/plain

fall_events,patient_id=p1,device_id=aa:bb fall_detected=true,patient_confirmed=1i,
needs_help=true,confidence=0.91,observation_id="<uuid>" <timestamp_ns>
```

This travels through FOCUS's existing Traefik on port 443. No new infrastructure needed.

### fall-dashboard reading from InfluxDB

fall-dashboard (inside the new cluster) queries InfluxDB (inside the existing cluster)
for fall history and fall counts. From Kubernetes' perspective this is just an outbound
HTTPS call to an external URL — no different from calling any third-party API.

```
POST https://influxdb.xxx.e-healthservice.de/api/v2/query
Authorization: Token <influx_token>
Content-Type: application/vnd.flux

from(bucket: "fd_test")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "fall_events")
  ...
```

Called by:
- `/api/patients` — returns per-patient fall counts (GROUP BY patient_id, COUNT)
- `/api/falls` — returns fall history rows with confirmation status

### patient_confirmed int encoding in InfluxDB

InfluxDB stores `patient_confirmed` as an integer (field type locks on first write):

```
 1  = patient confirmed it was a fall  ("yes")
 0  = patient denied, false positive   ("no")
-1  = no response within timeout       ("not_answered")
```

The mobile app must write the integer form. Mixing string and int for the same field
causes silent write failures in InfluxDB.

---

## Summary — what each component connects to

| Component | Location | Connects to | Protocol | Port | Comm type |
|---|---|---|---|---|---|
| Mobile app | phone (any network) | MCS inference-server | HTTPS | 443 | Type 9 |
| Mobile app | phone (any network) | mosquitto (new cluster) | MQTTS | 8883 | Type 6 |
| Mobile app | phone (any network) | InfluxDB (existing cluster) | HTTPS | 443 | Type 1 |
| fall-dashboard | new k3s cluster | mosquitto (same cluster) | MQTT (internal DNS) | 1883 | Type 10 |
| fall-dashboard | new k3s cluster | InfluxDB (existing cluster) | HTTPS | 443 | Type 2 |
| Flutter dashboard | existing k3s cluster | fall-dashboard (new cluster) | HTTPS + SSE | 443 | Type 2 |

---

## All communication types — full classification

| Type | Description | Use case in our system | Example |
|---|---|---|---|
| 1 | External client -> pod inside k3s cluster (HTTPS) | Mobile app writes fall timestamps to InfluxDB | `POST https://influxdb.xxx.e-healthservice.de/api/v2/write` |
| 2 | Pod inside cluster A -> pod inside cluster B (HTTPS, cross-cluster) | fall_dashboard reads fall history from InfluxDB | `POST https://influxdb.../api/v2/query` (Flux) |
| 2 | Pod inside cluster A -> pod inside cluster B (HTTPS, cross-cluster) | Flutter dashboard reads SSE feed + fall data from fall_dashboard | `GET https://<fall-dashboard-host>/api/stream` |
| 3 | Pod inside k3s cluster -> external client (HTTPS) | **None** | — |
| 4 | External service (MCS Hetzner) -> pod inside k3s cluster (HTTPS) | **None** | — |
| 5 | Pod inside k3s cluster -> external service (MCS Hetzner) (HTTPS) | **None** | — |
| 6 | External client -> pod inside k3s cluster (MQTTS) | Mobile app publishes fall alert to mosquitto broker | `PUBLISH mqtts://xxx:8883 fall/alert/<patient_id>` |
| 7 | Pod -> pod within k3s cluster (MQTT, between different services via broker) | **None** | — |
| 8 | Pod inside k3s cluster -> external client (MQTTS) | **None** | — |
| 9 | External client -> service on MCS Hetzner (HTTPS) | Mobile app sends sensor data for inference | `POST https://mcs-server/predict` and `POST .../confirm` |
| 10 | Pod -> pod within the same k3s cluster (any protocol, internal DNS) | mosquitto delivers subscribed MQTT message to fall_dashboard | `mosquitto:1883` (internal cluster DNS, plain TCP) |

**Types with no use case in this system: 3, 4, 5, 7, 8**

---

## What needs to be configured on the new k3s cluster

| Item | What | Why |
|---|---|---|
| Traefik entrypoint | Add `--entrypoints.mqtts.address=:8883` | Traefik does not listen on 8883 by default |
| IngressRouteTCP | Route port 8883 -> mosquitto:1883 | Expose broker to external clients |
| mosquitto Deployment + PVC | Persistent message broker pod | Standard k8s workload |
| fall-dashboard Deployment | Stateless pod, HTTP Ingress on :443 | Standard k8s workload |
| fall-dashboard env vars | `INFLUXDB_URL`, `INFLUXDB_TOKEN`, `INFLUXDB_ORG`, `INFLUXDB_FALL_EVENTS_BUCKET`, `PATIENT_IDS`, `MQTT_BROKER_HOST` (internal DNS), `MQTT_POSSIBLE_TOPIC`, `MQTT_ALERT_TOPIC` | All runtime config via ConfigMap/Secret |

## What does NOT need to change on the existing cluster

Nothing. The existing Traefik and InfluxDB deployment are unchanged.
The mobile app and fall-dashboard both connect to InfluxDB as standard HTTP clients.
