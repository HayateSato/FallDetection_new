# Hetzner MQTT WSS Debug Log
**Date:** 2026-06-22 — 2026-06-23
**Goal:** Test MQTT WebSocket connectivity from laptop to k3s caregiver layer on Hetzner (`5.75.255.114`) via `wss://fall-mqtt.smarko-health.de`

---

## FINAL STATUS: RESOLVED (2026-06-23)

MQTT broker is confirmed working. Full end-to-end test passed via Postman:
- Connected to `wss://fall-mqtt.smarko-health.de` (port 443, WSS) ✓
- Publish/subscribe round-trip confirmed (subscriber received published message) ✓
- mosquitto pod logs confirmed: `postman-mqtt-client` connected + PINGREQ/PINGRESP ✓
- `fall-detection-caregiver` (fall-dashboard) connected on port 1883 ✓

**Root cause of the long debugging session:** `mosquitto_sub` on Windows was the broken
tool, not the infrastructure. mosquitto_sub uses OpenSSL on Windows and cannot handle
Traefik v3's TLS renegotiation behavior. It produced false "protocol error" results on
every test, leading to a long investigation into Traefik routing that was ultimately
unnecessary. The IngressRouteTCP on `websecure` (port 443) was working correctly the
entire time after mosquitto was restarted (Issue 2 fix).

**For future MQTT testing:** use Postman or MQTT Explorer. Do NOT use `mosquitto_sub`
on Windows to test WSS connections through Traefik v3.

---

## Current cluster state (final, confirmed working)

| Resource | State | Notes |
|---|---|---|
| mosquitto Deployment | Running, healthy | Port 9001 open (restarted during debug) |
| mosquitto Service | ClusterIP, ports 1883 + 9001 | Unchanged from original |
| IngressRoute `mosquitto-ws` | DELETED | Removed during debug — not needed |
| IngressRoute `fall-dashboard` | `Host("5.75.255.114")`, no TLS cert | Unchanged |
| IngressRouteTCP `mosquitto-ws-tcp` | `websecure` entrypoint, `certResolver: le`, TLSOption `mqtt-tls-13` | Working ✓ |
| TLSOption `mqtt-tls-13` | `minVersion: VersionTLS13` | Applied during debug — harmless, kept |

---

## Issue 1 — Routing conflict (IP address as IngressRoute host)

**Status: FIXED**

Both IngressRoutes had `Host("5.75.255.114")` — Traefik routed WebSocket to fall-dashboard
instead of mosquitto. Fixed by DNS A record `fall-mqtt.smarko-health.de → 5.75.255.114` +
helm upgrade.

---

## Issue 2 — mosquitto not listening on port 9001

**Status: FIXED**

mosquitto pod started before WebSocket listener was in ConfigMap. Fixed by:
```bash
kubectl rollout restart deployment/mosquitto -n fall-dashboard
```
Startup logs confirmed: `Opening ipv4 listen socket on port 9001` ✓

---

## Issue 3 — "protocol error" on all mosquitto_sub tests

**Status: RESOLVED — was a test tool problem, not an infrastructure problem**

### What we thought was happening

We believed Traefik v3's `websecure` entrypoint was broken for MQTT WebSocket because
`mosquitto_sub` returned "protocol error" on every test attempt. We tried:

| Attempt | Change | Result |
|---|---|---|
| 1 | Removed certResolver from IngressRoute | Still "protocol error" |
| 2 | Replaced IngressRoute with IngressRouteTCP | Still "protocol error" |
| 3 | Added TLSOption (TLS 1.3 minimum) | Still "protocol error" |

Traefik logs consistently showed `tls: bad record MAC` on each attempt.

### What was actually happening

`mosquitto_sub` on Windows uses OpenSSL and cannot handle TLS renegotiation.
Traefik v3 requests TLS renegotiation during the WebSocket handshake (visible in curl
as `schannel: remote party requests renegotiation`). mosquitto_sub fails when this
happens, producing "protocol error". The server-side connection was working correctly.

### Proof

Postman (Electron/Node.js TLS stack — handles renegotiation transparently):
- Connected to `wss://fall-mqtt.smarko-health.de` on port 443 immediately ✓
- mosquitto pod logs showed the client connected and doing PINGREQ/PINGRESP ✓
- Publish/subscribe round-trip confirmed end-to-end ✓

### Changes made during debugging that are still in the cluster

These were applied during debugging and are harmless to leave in place:
- `IngressRoute mosquitto-ws` → DELETED (was the old HTTP route, not needed)
- `IngressRouteTCP mosquitto-ws-tcp` → CREATED on `websecure` with `certResolver: le`
- `TLSOption mqtt-tls-13` → CREATED with `minVersion: VersionTLS13`, attached to IngressRouteTCP

The IngressRouteTCP + TLSOption combination is a reasonable production configuration.

---

## Lesson learned

When testing WSS MQTT through Traefik v3 on Windows:
- `mosquitto_sub` → incompatible with Traefik v3 TLS renegotiation, gives false errors
- `curl` with WebSocket headers → shows the renegotiation + close_notify, misleading
- **Postman / MQTT Explorer** → correct tools, handle TLS transparently

---

## Note on Option A (port 8883) — NOT needed

During debugging we planned a dedicated port 8883 entrypoint as a fix. This is NOT
needed — port 443 works correctly. Do not apply the HelmChartConfig patch from the
v2 doc's "COMMANDS TO RUN" section. The broker is working on port 443 as designed.
