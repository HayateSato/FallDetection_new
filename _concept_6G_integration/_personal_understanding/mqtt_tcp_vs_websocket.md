# MQTT — TCP vs WebSocket: Why We Need Both Ports

## The short answer

| Port | Protocol | Used by |
|------|----------|---------|
| 1883 | Raw TCP | `mock_app`, `fall_dashboard` (Python services) |
| 9001 | WebSocket | React Native mobile app, browser-based clients |

Both carry MQTT messages. The difference is the **transport layer** underneath, not the MQTT protocol itself.

---

## What is raw TCP (port 1883)?

TCP is a direct, low-level network connection. When Python's `paho-mqtt` library connects to the broker on port 1883, it opens a plain TCP socket and speaks the MQTT protocol directly over it.

```
paho-mqtt (Python)
    │
    │  raw TCP socket
    ▼
MQTT broker :1883
```

This is fast, simple, and works great for server-to-server or process-to-process communication — which is exactly what `mock_app` (publisher) and `fall_dashboard` (subscriber) are doing.

---

## What is WebSocket (port 9001)?

WebSocket is a protocol that starts as an HTTP request and then upgrades to a persistent, full-duplex connection. It was designed specifically so that **browsers and mobile apps** could maintain long-lived connections to a server.

```
React Native app
    │
    │  HTTP upgrade → WebSocket
    ▼
MQTT broker :9001  (WebSocket listener)
    │
    │  MQTT messages travel inside WebSocket frames
    ▼
same broker, same topics, same messages
```

The MQTT message content is identical — it just travels inside WebSocket frames instead of raw TCP packets.

---

## Why can't React Native just use TCP (port 1883)?

React Native (and browsers) **do not have access to raw TCP sockets** for security reasons. The JavaScript/mobile runtime only allows:
- HTTP/HTTPS requests
- WebSocket connections

Raw TCP is a lower-level primitive that the browser/mobile sandbox intentionally blocks. So `mqtt.connect('tcp://host:1883')` fails silently or throws an error in React Native — it is not a bug, it is a platform restriction.

The workaround is MQTT over WebSocket: the broker listens on port 9001 for WebSocket connections, wraps MQTT inside those frames, and everything else (topics, QoS, subscriptions) works exactly the same.

---

## Does this mean there are two separate brokers?

No. It is the **same Eclipse Mosquitto broker** with two listeners configured in `mosquitto.conf`:

```
listener 1883
protocol mqtt

listener 9001
protocol websockets
```

A message published on `fall/alert/patient_001` by `mock_app` (via TCP 1883) is received by React Native (via WebSocket 9001) just like any other subscription. The broker handles the translation internally.

---

## In our system

```
mock_app (Python, paho-mqtt)
    │  publish fall/alert/<patient_id>
    │  TCP → port 1883
    ▼
Eclipse Mosquitto broker
    ▲                    ▲
    │ TCP :1883          │ WebSocket :9001
    │                    │
fall_dashboard      React Native app
(Python, paho-mqtt)  (Isa's mobile app)
subscriber           subscriber
```

Both `fall_dashboard` and the React Native app receive the same MQTT message when `mock_app` publishes a fall alert. The only difference is the transport they use to connect to the broker.

---

## Port conflict with MinIO

MinIO's web console was originally configured on port 9001, which caused a conflict when added to `docker-compose.yml`. It was moved to **port 9002**. Port 9001 must stay reserved for MQTT WebSocket.

| Port | Service |
|------|---------|
| 1883 | MQTT TCP |
| 9001 | MQTT WebSocket (React Native) |
| 9000 | MinIO S3 API |
| 9002 | MinIO web console |
