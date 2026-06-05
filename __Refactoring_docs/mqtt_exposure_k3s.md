## How MQTT exposure changes in K3s

Context: The caregiver layer (mqtt broker, fall-dashboard, mock-app) is hosted by **FOCUS on their own physical machine**, not Hetzner. K3s + Traefik runs on that machine. The mobile app (Isa) connects from the hospital/clinic WiFi network to FOCUS's machine over LAN or their internal network.

In Docker Compose, exposure is trivial — `ports: "1883:1883"` maps straight to the host NIC. In K3s, MQTT is a raw TCP protocol which means **Traefik's normal HTTP ingress won't handle it** — you need one of three approaches:

---

### Option A — LoadBalancer Service (simplest, works immediately)

K3s ships with a built-in load balancer (ServiceLB/Klipper) that can expose raw TCP ports directly on the physical machine's NIC. No Traefik config needed.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mqtt-broker
  namespace: caregiver
spec:
  type: LoadBalancer
  selector:
    app: mqtt
  ports:
    - name: mqtt
      port: 1883
      targetPort: 1883
```

Mobile app connects to `<FOCUS-machine-ip>:1883` — same as the two-laptop test but with FOCUS's physical machine IP. Simple, but exposes a raw unencrypted port on their internal network.

Good fit here because FOCUS's machine is inside their own network (not a public server), so no TLS is strictly required for the initial integration.

---

### Option B — Traefik TCP IngressRouteTCP

Traefik supports TCP routing via a `TCPRoute` CRD, but requires adding a custom **entrypoint** (Traefik doesn't listen on 1883 by default, only 80/443).

```yaml
# 1. Patch Traefik to listen on port 1883 (add to HelmChartConfig or values)
additionalArguments:
  - "--entrypoints.mqtt.address=:1883"

# 2. Expose 1883 on the Traefik service
ports:
  mqtt:
    port: 1883
    exposedPort: 1883
    protocol: TCP
```

```yaml
# 3. Route TCP traffic to the mqtt pod
apiVersion: traefik.io/v1alpha1
kind: IngressRouteTCP
metadata:
  name: mqtt-tcp
  namespace: caregiver
spec:
  entryPoints:
    - mqtt
  routes:
    - match: HostSNI(`*`)   # no TLS = wildcard match
      services:
        - name: mqtt-broker
          port: 1883
```

More config than Option A but keeps everything routed through Traefik.

---

### Option C — MQTT over WebSocket via HTTPS (best if FOCUS requires TLS internally)

If FOCUS has a hostname/cert for their machine (e.g. `focus-server.hospital.de`), this is the cleanest path. Instead of raw TCP 1883, mosquitto also listens on 9001 (WebSocket), Traefik routes it as a normal HTTP/WebSocket upgrade at port 443, TLS terminated at Traefik.

```
Mobile app  →  wss://focus-server.hospital.de  →  Traefik:443  →  mqtt pod:9001
```

Mosquitto config gains one listener:

```
listener 9001
protocol websockets
allow_anonymous true
```

Traefik Ingress (WebSocket upgrades work automatically):

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mqtt-ws
  annotations:
    traefik.ingress.kubernetes.io/router.entrypoints: websecure
spec:
  rules:
    - host: focus-server.hospital.de
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: mqtt-broker
                port:
                  number: 9001
```

Mobile app (MQTT.js / React Native):

```js
mqtt.connect('wss://focus-server.hospital.de')  // port 443, TLS at Traefik
```

---

## Recommendation for your setup

|  | Option A | Option B | Option C |
|---|---|---|---|
| Config effort | Minimal | Medium | Medium |
| TLS support | No (raw TCP) | Needs cert on broker | Yes (Traefik terminates) |
| Works with port 1883 | Yes | Yes | No (uses 443/WebSocket) |
| Works with MQTT.js (React Native) | No (needs WebSocket) | No | Yes |
| Good for FOCUS physical machine | Yes — simple LAN exposure | Yes | Only if FOCUS has a hostname + cert |

**For local K3s testing on FOCUS machine:** Option A — ServiceLB exposes port 1883 on the physical NIC directly, Isa's app connects to that IP. Matches the two-laptop test setup exactly.

**For production on FOCUS machine:** depends on whether FOCUS requires TLS and whether Isa's app uses plain TCP MQTT or WebSocket MQTT.
- If Isa's app uses `react-native-mqtt` (plain TCP) → Option A or B, port 1883.
- If Isa's app uses `MQTT.js` (WebSocket) → Option C, port 9001 behind Traefik HTTPS.

**Coordinate with FOCUS DevOps** (Mohammed's counterpart there) on:
1. Whether their machine has a hostname or just an IP.
2. Whether their internal network policy requires TLS.
3. Which MQTT library Isa's app will use — that determines Option A/B vs C.
