# MQTT Architecture — Fall Detection System

## Overview

MQTT is used for **one purpose only**: delivering a confirmed fall alert from the mobile app to the caregiver dashboard.

The inference server is **not an MQTT participant** — it communicates with the mobile app over HTTP only.

---

## MQTT Broker

| | |
|-|-|
| **What it is** | Central message router. All clients connect to it. |
| **Where it runs** | Separate process / container (not inside any app component) |
| **Implementation** | eclipse-mosquitto Docker container |
| **Local test** | `docker run -d --name mqtt-local -p 1883:1883 eclipse-mosquitto` |
| **Config** | `MQTT_BROKER_HOST`, `MQTT_BROKER_PORT` in `.env` |

The broker does not run inside any of our Python components. It is infrastructure, like a database server.

---

## MQTT Client 1 — mock_app (Publisher)

| | |
|-|-|
| **Lives in** | `_6G_Integration_v2_mqtt/mock_app/client.py` |
| **Role** | Publisher |
| **Publishes to** | `fall/alert/<patient_id>` |
| **When it publishes** | After the patient confirmation window closes (10s timeout) |
| **paho client ID** | `mock-app-publisher` |
| **Created by** | `_create_mqtt_publisher()` in `mock_app/client.py` |

### What happens before the MQTT publish

The mock_app is not a passive listener — it drives the full detection cycle:

```
1. Fetch ACC window from InfluxDB        (mock_app/influx_fetcher.py)
2. POST to inference server /predict     (mock_app/api_caller.py)
3. Receive HTTP response: fall_detected  (inference result returned directly)
4. Open patient confirmation window      (mock_app/poller.py — 10s sleep in daemon thread)
5. PUBLISH fall/alert/<patient_id>       (paho client.publish)
```

The inference server is **not involved in MQTT** — the fall result arrives in the HTTP response at step 3.

### Alert payload (published to fall/alert/<patient_id>)

```json
{
  "patient_id":        "test_patient-001",
  "device_id":         "6c:1d:eb:04:a9:e6",
  "timestamp":         "2026-04-14T10:00:00+00:00",
  "fall_detected":     true,
  "confidence":        0.912,
  "model_version":     "v0",
  "fhir_observation":  { ... },
  "alert_time":        "2026-04-14T10:00:10+00:00",
  "patient_confirmed": "not_answered",
  "needs_help":        true
}
```

---

## MQTT Client 2 — fall_dashboard (Subscriber)

| | |
|-|-|
| **Lives in** | `_6G_Integration_v2_mqtt/fall_dashboard/mqtt_listener.py` |
| **Role** | Subscriber |
| **Subscribes to** | `fall/alert/#`  (wildcard — receives alerts for all patients) |
| **paho client ID** | `fall-detection-caregiver` |
| **Class** | `FallEventBroker` |
| **Started by** | `fall_dashboard/client.py` startup hook (after `on_fall` callback is set) |

### What happens on message received

```
MQTT message arrives on fall/alert/<patient_id>
    → FallEventBroker._on_message() (paho background thread)
    → calls on_fall(event)           (set by client.py before broker.start())
        → cdb.record_fall(...)       (DB write — SQLite)
        → broker.publish_local(...)  (SSE fan-out to dashboard browsers)
```

The `on_fall` callback is wired in `client.py` before `broker.start()` is called — this prevents a race condition where a message could arrive before the callback is set.

---

## What does NOT have an MQTT client

| Component | Why not |
|-----------|---------|
| `inference_server/server.py` | Returns fall result in HTTP response — no need to publish separately. The mobile app already has the result. |
| `influx_marker_writer.py` | Deleted — colleague writes InfluxDB markers directly from their side. |

---

## Topic Summary

| Topic | Publisher | Subscriber |
|-------|-----------|------------|
| `fall/alert/<patient_id>` | mock_app | fall_dashboard |

Only one active topic. The previous `fall/events/<patient_id>` topic (used when inference server published directly) has been removed.

---

## Full Message Flow

```
[InfluxDB]
    │
    │  fetch ACC window
    ▼
[mock_app]  ──── HTTP POST /predict ────▶  [inference_server :8001]
    ◀─────────── HTTP response ──────────  fall_detected=True, confidence=0.91
    │
    │  10s patient confirmation wait
    │  (simulated — real app shows popup)
    │
    │  MQTT PUBLISH fall/alert/test_patient-001
    ▼
[MQTT broker :1883]
    │
    │  route to subscribers of fall/alert/#
    ▼
[fall_dashboard :8002]
    │  DB write (SQLite)
    │  SSE fan-out
    ▼
[caregiver dashboard browser]
    alert: "test_patient-001 has fallen at 10:00:10 and needs help"
```

---

## Configuration (.env)

```env
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883
MQTT_ALERT_TOPIC=fall/alert
MQTT_USERNAME=
MQTT_PASSWORD=
MOCK_PATIENT_RESPONSE_TIMEOUT=10
```

`MQTT_ALERT_TOPIC` is the base topic. Both mock_app and fall_dashboard read this value:
- mock_app publishes to `{MQTT_ALERT_TOPIC}/{patient_id}`
- fall_dashboard subscribes to `{MQTT_ALERT_TOPIC}/#`
