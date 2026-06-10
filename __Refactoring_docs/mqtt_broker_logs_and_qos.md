# Reading the MQTT Broker Logs & When to Change QoS

**Audience:** FOCUS engineers operating the Fall Detection caregiver layer.
**Scope:** how to read the `focus_mqtt` (Mosquitto) broker log, and when/why/where
to change MQTT QoS for the fall-alert path.

---

## 1. Who writes this log

The log comes from the **broker** (`focus_mqtt`, an Eclipse Mosquitto container).
The single most important rule:

> **Every "Received" and "Sending" line is written from the BROKER's point of view.**

- **"Received … from X"** = the broker got a message **from** client X.
- **"Sending … to X"** = the broker is pushing a message **out to** client X.

Clients never talk to each other directly — all traffic goes through the broker.
So one published message produces **two** log lines: a "Received from publisher"
and a "Sending to subscriber".

```
mock-app-publisher ──PUBLISH──► [ focus_mqtt broker ] ──PUBLISH──► fall-detection-caregiver
        (mock app)                  (writes this log)                    (dashboard)
                          "Received from"          "Sending to"
```

### The two clients

| Client name in log | What it is | Container | Role |
|--------------------|------------|-----------|------|
| `mock-app-publisher` | Mock mobile app | `focus_mock_app` | **Publishes** fall events |
| `fall-detection-caregiver` | Fall dashboard | `focus_fall_dashboard` | **Subscribes** to alerts |

---

## 2. How to view the log

```powershell
docker logs -f focus_mqtt           # follow live
docker logs --tail 100 focus_mqtt   # last 100 lines
```

### Timestamps are Unix epoch seconds
The number on the left (e.g. `1780574956:`) is seconds since 1970, **not** a line
number. Convert one:

```powershell
[DateTimeOffset]::FromUnixTimeSeconds(1780574956).ToLocalTime()
```

---

## 3. Reading a real fall flow

```
SUBSCRIBE from fall-detection-caregiver: fall/alert/#  (QoS 0)     ← dashboard asks for all alerts
SUBSCRIBE from fall-detection-caregiver: fall/possible/# (QoS 0)   ← and pre-confirmation alerts

Received PUBLISH from mock-app-publisher    'fall/possible/patient_test_2' (1419 bytes)  ← mock app sent a "possible fall"
Sending  PUBLISH to   fall-detection-caregiver 'fall/possible/patient_test_2' (1419 bytes)  ← broker forwarded it

Received PUBLISH from mock-app-publisher    'fall/alert/patient_test_2' (1497 bytes)     ← mock app sent the CONFIRMED alert
Sending  PUBLISH to   fall-detection-caregiver 'fall/alert/patient_test_2' (1497 bytes)     ← broker forwarded it
```

Each **Received + Sending pair with the same topic and byte count is the same
message** passing through. Same timestamp = forwarded instantly.

### The diagnostic rule
- **"Received from mock-app" present** → the broker definitely got the message.
- **No matching "Sending to …"** → **no subscriber was connected**, so the broker
  dropped it (at QoS 0 there is no queue for offline clients). **That fall was lost.**
- **"Sending to …" present** → broker *attempted* delivery (see QoS caveat below).

### Lines you can ignore (normal noise)
| Line | Meaning |
|------|---------|
| `PINGREQ` / `PINGRESP` | Keepalive heartbeats (every ~60 s, `k60`). Connection-alive only, no data. |
| `CONNECT` / `CONNACK (0,0)` | A client connected successfully. |
| `DISCONNECT` / `disconnected` | A client dropped. `connection closed by client` = a normal reconnect. |
| `SUBSCRIBE` / `SUBACK` | A client registered for a topic; broker acknowledged. |
| `No will message specified` | Client set no "last will" message. Harmless. |

---

## 4. Understanding QoS (the `q0` in the log)

Each PUBLISH line shows flags like `(d0, q0, r0, m0)`:
- `q0` = **QoS 0** ← the important one
- `d0` = not duplicate, `r0` = not retained, `m0` = message id 0

### QoS levels

| QoS | Guarantee | Sender gets confirmation? |
|-----|-----------|---------------------------|
| **0** (current) | "fire and forget" — send once, no ack, no retry | **No** |
| **1** | "at least once" — receiver sends `PUBACK`, retried until ack'd (may duplicate) | Yes |
| **2** | "exactly once" — 4-step handshake | Yes |

### What QoS 0 means for these logs (key point)
At QoS 0, **the broker does NOT know whether the dashboard actually received the
message.** A `Sending PUBLISH to fall-detection-caregiver` line means only
*"handed to the TCP socket"* — **not** *"the dashboard app processed it."* There
is no acknowledgement at QoS 0, which is why you see only two lines per message
(no third "ack" line).

To **prove** the dashboard processed a fall, check the dashboard's own log:
```powershell
docker logs focus_fall_dashboard | Select-String "patient_test_2"
```

---

## 5. QoS is set by the CLIENTS, not the broker

A common misconception: QoS is **not** a broker config switch. It is chosen by
the clients, per message / per subscription. There are two independent choices:

1. **Publish QoS** — chosen by the **publisher** on each `publish(..., qos=)` call.
   (This is the `q0` in "Received PUBLISH from mock-app-publisher".)
2. **Subscribe QoS** — chosen by the **subscriber** as a *maximum* on `subscribe(..., qos=)`.
   (This is the `(QoS 0)` in the SUBSCRIBE line.)

The broker enforces:

> **effective delivery QoS = min(publish QoS, subscribe QoS)**

So to get QoS 1 end-to-end, **BOTH** the publisher and the subscriber must use
QoS 1. The Mosquitto broker needs **no change** — it already supports all levels.

---

## 6. When / why / where to change QoS

### When & why to consider QoS 1
The fall-alert path is safety-critical. At QoS 0 a fall alert can **silently
vanish** with no record of the loss, if any of these happen:
- the dashboard is momentarily disconnected (you'll see `disconnected` /
  `New connection` pairs in the log — these happen routinely);
- a dropped packet on the network;
- the subscriber's internal queue is full.

**QoS 1** makes the broker retry until the subscriber acknowledges, and combined
with a persistent session (`clean_session=false`) it will even queue alerts for a
briefly-disconnected dashboard. The trade-offs: possible **duplicate** deliveries
(the receiver must tolerate/deduplicate them) and slightly more overhead.

Recommendation: keep `fall/possible/#` at QoS 0 (transient pre-alerts), but
**upgrade `fall/alert/#` (the confirmed alerts) to QoS 1.**

### Where to change it (exact locations)

Both sides currently pass **no `qos=` argument**, so paho defaults to QoS 0.

| Hop | File | Lines | Current | Change to |
|-----|------|-------|---------|-----------|
| **Publish** (mock app) | `local_dev/mock_app/poller.py` | ~132 and ~230 | `self._mqtt.publish(topic, payload)` | `self._mqtt.publish(topic, payload, qos=1)` *(for the alert publish)* |
| **Subscribe** (dashboard) | `fall_dashboard/mqtt_listener.py` | 124–125 | `client.subscribe(f"{MQTT_ALERT_TOPIC}/#")` | `client.subscribe(f"{MQTT_ALERT_TOPIC}/#", qos=1)` |

For full reliability across reconnects, also set a **persistent session** on the
dashboard client (in `mqtt_listener.py` where `mqtt.Client(...)` is created):
`clean_session=False` with a stable `client_id` — so the broker holds undelivered
QoS 1 alerts while the dashboard is briefly offline.

### After changing
Both the mock app and dashboard are rebuilt/restarted like any code change:
```powershell
# from caregiver_layer/
docker compose up -d --build mock-app fall-dashboard
```
The broker (`focus_mqtt`) is not modified.

### How to confirm the change worked
In the broker log, the PUBLISH flags will show `q1` instead of `q0`, and you'll
see acknowledgement traffic (`PUBACK`) for the alert messages.

---

## 7. Quick cheat-sheet

| Symptom in broker log | Meaning |
|-----------------------|---------|
| `Received PUBLISH from mock-app-publisher 'fall/alert/...'` + matching `Sending to fall-detection-caregiver` | Alert flowed through broker to dashboard (delivery attempted) |
| `Received ...` with **no** matching `Sending ...` | No subscriber connected → **alert lost** |
| `q0` in PUBLISH flags | Fire-and-forget; no delivery confirmation |
| `q1` in PUBLISH flags + `PUBACK` | Acknowledged delivery (after the QoS 1 change) |
| Repeated `disconnected` / `New connection` | Client reconnects — normal, but at QoS 0 any alert sent during the gap is lost |
| `PINGREQ` / `PINGRESP` | Keepalive only — ignore |

To prove the dashboard actually *processed* an alert (not just that the broker
sent it), always cross-check `docker logs focus_fall_dashboard`.
