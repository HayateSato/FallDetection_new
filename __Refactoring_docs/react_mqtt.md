For **React Native** (mobile), there's a catch: JavaScript can't open raw TCP sockets directly, so most JS MQTT libraries use **WebSockets** instead of plain TCP. That means port `1883` won't work out of the box.

You have two options:

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

## Recommendation for Isa

|  | Option A (MQTT.js) | Option B (react-native-mqtt) |
| --- | --- | --- |
| Port | 9001 (WebSocket) | 1883 (TCP) |
| Mosquitto change | Yes — add WS listener | No |
| Library maturity | Very high | Moderate |
| Setup effort | Easy | Requires native build config |

**Go with Option A** if Isa is already using MQTT.js or wants the simplest JS setup — just requires adding 4 lines to `mosquitto.conf`. Pass her the WS listener change and the `ws://192.168.8.126:9001` address.