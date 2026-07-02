# k3s — Internal Port vs Machine OS Port

## Why Traefik needs two separate subdomains (not two ports)

External clients never connect to port 9001 or 8002. Those are internal ports
that Traefik forwards *to*. From the outside, everything arrives on port 443:

```
Mobile app  →  5.75.255.114:443  →  Traefik
Browser     →  5.75.255.114:443  →  Traefik
```

Traefik receives both on the same port. At this point it has to decide: does
this go to mosquitto:9001 or fall-dashboard:8002?

**The only information it has is the HTTP Host header** — the domain name from
the URL the client used:

```
wss://mqtt.smarkohealth.de   → HTTP header: Host: mqtt.smarkohealth.de  → mosquitto:9001
https://fall.smarkohealth.de → HTTP header: Host: fall.smarkohealth.de  → fall-dashboard:8002
```

With an IP address there is no hostname — both requests carry the same header:

```
wss://5.75.255.114   → HTTP header: Host: 5.75.255.114  ← identical
https://5.75.255.114 → HTTP header: Host: 5.75.255.114  ← identical
```

Traefik has two IngressRoutes both saying `Host("5.75.255.114")` — it cannot
tell them apart.

The destination ports (9001, 8002) are what Traefik forwards TO inside the
cluster. External clients never specify those ports. They just connect to 443.

---

## NodePort — Traefik is bypassed entirely

With NodePort, routing happens at the OS networking layer (kube-proxy), not
Traefik. The request never reaches Traefik:

```
mobile app → ws://5.75.255.114:30901 → kube-proxy → mosquitto pod
                                        (OS level, no Traefik)
```

Port number distinguishes the traffic, but Traefik doesn't read it. Traefik
is simply bypassed.

---

## Mixed setup (NodePort for MQTT + Traefik for fall-dashboard)

```
mobile app → ws://5.75.255.114:30901  → kube-proxy → mosquitto   (Traefik not involved)
browser    → https://5.75.255.114:443 → Traefik → fall-dashboard  (only one IngressRoute matches)
```

Once mosquitto is on NodePort, there is only one IngressRoute left on port 443
— the fall-dashboard one. No conflict. Traefik sees a request for
`Host("5.75.255.114")` on 443 and there is only one match.

---

## What routes what — full table

| Traffic | Goes through | Routed by |
|---|---|---|
| `ws://5.75.255.114:30901` | NodePort | kube-proxy (OS) — port number |
| `http://5.75.255.114:8002` | LoadBalancer | kube-proxy (OS) — port number |
| `https://5.75.255.114:443` | Traefik | Host header — hostname |

The OS routes by **port**. Traefik routes by **hostname**. They operate at
different levels and do not interfere with each other.

---

## Where ServiceLB (LoadBalancer) fits in k3s architecture

k3s does not have a cloud provider (AWS ELB, GCP LB). Instead it uses its own
built-in **ServiceLB** (klipper-lb): when a Service of type LoadBalancer is
created, k3s runs a DaemonSet pod (`svclb-*`) that binds a real port on the
host's NIC using `hostPort` in the container spec.

```
External traffic hits 5.75.255.114
              │
              ▼
     Host NIC -- OS sees the port
              │
    ┌─────────┼──────────────────────────┐
    │         │                          │
   :443      :8002                     :30901
    │         │                          │
    ▼         ▼                          ▼
svclb-traefik  svclb-fall-dashboard-lb   kube-proxy
(ServiceLB     (ServiceLB pod,           (NodePort,
 pod, binds     binds hostPort 8002)      iptables rule)
 hostPort 443)  │                         │
    │           ▼                         ▼
    ▼     fall-dashboard pod         mosquitto pod
 Traefik pod
    │
    │  reads Host header
    ├── Host("mqtt.x")   → mosquitto:9001
    └── Host("fall.x")   → fall-dashboard:8002
```

- **For port 443**: ServiceLB → Traefik → pod. ServiceLB is before Traefik.
- **For ports 8002, 8086**: ServiceLB → pod directly. Traefik never involved.
- **For NodePort (30901)**: no ServiceLB — kube-proxy iptables rules only.

There are three completely separate paths from the internet to a pod. None of
them go through each other.

---

## Summary

| Concept | Who handles it | Based on |
|---|---|---|
| Traefik ingress routing | Traefik | HTTP Host header (hostname) |
| LoadBalancer port binding | k3s ServiceLB (klipper-lb) | Port number on host NIC |
| NodePort routing | kube-proxy (iptables) | Port number on host NIC |
| Internal service discovery | Kubernetes DNS | Service name (e.g. `mosquitto`) |
