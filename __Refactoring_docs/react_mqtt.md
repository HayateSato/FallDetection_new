For **React Native** (mobile), JavaScript can't open raw TCP sockets directly, so MQTT must go over **WebSockets** — port `1883` raw TCP does not work. The architecture is now settled on **Option A (MQTT.js + port 9001 WebSocket)**.

The two options were:

---

## Option A — MQTT.js over WebSocket (recommended for React Native)

The most popular library. But it needs WebSocket support on the broker.

**Mobile app side:**

`import mqtt from 'mqtt';

const client = mqtt.connect('ws://192.168.8.126:9001');`

**Your mosquitto.conf needs a second listener** (currently only has 1883):

`listener 1883
allow_anonymous true

listener 9001
protocol websockets
allow_anonymous true`

Then rebuild/restart the mqtt container.

---

## Option B — react-native-mqtt (native TCP, port 1883 as-is)

This wraps native MQTT libraries (Paho under the hood) so it can use plain TCP — no WebSocket needed, port 1883 works directly.

`import MQTT from 'react-native-mqtt';

MQTT.createClient({
  uri: 'mqtt://192.168.8.126:1883',
  clientId: 'mobile-app'
}).then(client => { client.connect(); });`

No mosquitto config change needed.

---

## Decision: Option A — MQTT.js over WebSocket (confirmed)

|  | Option A (MQTT.js) — CHOSEN | Option B (react-native-mqtt) |
| --- | --- | --- |
| Port | 9001 (WebSocket) | 1883 (TCP) |
| Mosquitto change | Yes — add WS listener | No |
| Library maturity | Very high | Moderate |
| Setup effort | Easy | Requires native build config |

**Option A is the chosen architecture.** React Native cannot open raw TCP sockets in standard JS, so port 1883 TCP is not viable. MQTT.js over WebSocket (port 9001) is the only approach that works without native module dependencies.

Mosquitto must be configured with both listeners:

```
listener 1883
allow_anonymous true

listener 9001
protocol websockets
allow_anonymous true
```

The 1883 listener remains for internal service-to-service connections (fall_dashboard → broker inside the container network). Port 9001 WebSocket is what the mobile app uses externally.