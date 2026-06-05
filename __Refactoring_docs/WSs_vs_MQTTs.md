## Your understanding is correct, with one nuance

`Docker / K3s internal:
  fall_dashboard  -->  mqtt:1883  (plain TCP MQTT, never leaves the container/pod network)

External (K3s with WebSocket):
  mobile app  -->  wss://host:443  -->  Traefik (TLS termination)  -->  mosquitto:9001  (plain WS inside cluster)`

The `9001` hop between Traefik and mosquitto is plain WebSocket — no TLS — but that's fine because it never leaves the cluster network. TLS only needs to exist on the outside edge, and Traefik handles that.

In the **Docker Compose version** there is no Traefik, so external is also plain `host:1883` — same port inside and out. The split only exists in the K3s version.

---

## Security: WSS (port 443) vs raw TCP (port 1883)

|  | Raw TCP 1883 | MQTTS 8883 | **WSS 443** |
| --- | --- | --- | --- |
| Encrypted in transit | No | Yes (TLS) | Yes (TLS) |
| Credentials protected | No — sniffable | Yes | Yes |
| Works through hospital firewalls | Often blocked | Often blocked | Always open |
| Works with MQTT.js (React Native) | No (needs native module) | No (needs native module) | Yes (standard JS) |
| TLS managed by | broker | broker | Traefik (cert-manager / Let's Encrypt) |

**WSS on 443 is the best option for a hospital environment** for two reasons:

1. **Security is equivalent to MQTTS** — both encrypt with TLS. The difference is who terminates TLS (Traefik vs mosquitto itself). Letting Traefik handle it is actually better because cert rotation is automatic via cert-manager.
2. **Port 443 is universally open** in hospital networks. Ports 1883 and 8883 are often blocked by the hospital IT firewall without a special request. Port 443 (HTTPS) never is — hospital systems depend on it. This is a practical deployment advantage, not just a theoretical one.

The only trade-off is a tiny WebSocket framing overhead per message (~10 bytes header) — completely irrelevant for fall alerts which are sent once per event.

## Protocol choice doesn't control external access — the network does

Whether connections from outside the hospital network are possible has nothing to do with MQTT vs WebSocket. It depends entirely on:

1. **Does the FOCUS machine have a public IP?** If it's behind hospital NAT with no port forwarding, nothing reaches it from outside — port 1883, 8883, or 443, doesn't matter.
2. **Does the hospital's external firewall forward inbound connections to that machine?** If not configured, nothing gets through.

Switching from WSS 443 to MQTT 1883 doesn't add isolation. It just changes the port. An attacker outside the network who can reach port 443 can equally reach port 1883 if both are exposed.

---

## The real control: bind to internal interface only

If you want to guarantee the broker is unreachable from outside, the correct fix is at the network level — not the protocol:

`Option 1 — Hospital IT simply doesn't port-forward to the FOCUS machine
           (most likely already the case — internal servers are behind NAT)

Option 2 — Traefik IP allowlist middleware (K3s)
           Only accept connections from the hospital internal IP range

Option 3 — Bind Traefik / mosquitto to the internal NIC IP only
           Service never answers on a public-facing interface`

---

## Where your intuition is partially right

There is one real difference: **port 443 is more likely to be accidentally exposed** by hospital IT because it's the standard HTTPS port and they may open it liberally for web services. Port 1883 or 8883 is obviously an MQTT port — a hospital IT team is less likely to forward it without a specific reason.

So the argument is:

|  | WSS 443 | MQTT 1883/8883 |
| --- | --- | --- |
| Accidentally exposed by hospital IT | Higher risk — IT may open 443 broadly | Lower risk — unusual port, unlikely to be forwarded without reason |
| Works through internal hospital WiFi firewall | Always | Sometimes blocked |
| Actual protocol security | Same (TLS on both) | Same |
| Correct control for external isolation | Network/firewall | Network/firewall |

---

## Recommendation

Use **WSS 443** but ask FOCUS DevOps to explicitly **not** port-forward it to the public internet. The machine should sit behind hospital NAT with no inbound forwarding. That gives you:

- Works reliably on hospital WiFi (port 443 never blocked internally)
- Encrypted (TLS)
- Not reachable from outside because it's never exposed — not because of the port number

If you use MQTT 1883 for isolation, you're relying on "obscurity" (unusual port) rather than an actual network control. That's not a security guarantee.

## Two different directions of traffic

"Universally open" and "blocked by firewall" refer to **outbound** traffic from devices **inside** the hospital network — not inbound from the internet.

`OUTSIDE INTERNET
      │
      │  ← inbound: controlled by hospital external firewall / NAT
      │
┌─────▼──────────────────────────────────┐
│  HOSPITAL NETWORK                       │
│                                         │
│   Phone (WiFi)  ──outbound──►  :443  ──►  FOCUS machine  │
│   Phone (WiFi)  ──outbound──►  :1883 ──►  FOCUS machine  │
│                      ▲                  │
│                      │                  │
│              "universally open"         │
│              means THIS direction       │
└─────────────────────────────────────────┘`

When I said **"port 443 is universally open"** I meant: a phone on hospital WiFi trying to make an **outbound** connection to port 443 will never be blocked by the internal WiFi firewall. Port 1883 on the other hand may be blocked — many hospital WiFi policies only allow outbound 80 and 443.

When I said **"behind NAT, nothing reaches it from outside"** I meant the **inbound** direction — traffic originating from the internet trying to reach the FOCUS machine. That is controlled by whether the hospital router is configured to forward inbound connections to that machine. By default it is not, so the machine is unreachable from outside regardless of port.

---

## Concrete summary

| Scenario | Port 443 | Port 1883 |
| --- | --- | --- |
| Phone on hospital WiFi → FOCUS machine | Always works | May be blocked by WiFi policy |
| Internet → FOCUS machine (no port forwarding) | Blocked | Blocked |
| Internet → FOCUS machine (IT accidentally forwards 443) | Reachable | Not reachable |
| Internet → FOCUS machine (IT explicitly forwards 1883) | Not reachable | Reachable |

The only scenario where the port choice matters for external isolation is the third row — accidental exposure. That's the risk I flagged, not a fundamental protocol difference.