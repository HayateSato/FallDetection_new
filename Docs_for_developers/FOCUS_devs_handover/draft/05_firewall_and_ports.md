# Firewall and Port Management

## Short Answer

If Traefik is already serving other HTTPS services in your cluster, **no new firewall
rules are needed**. Both new services (MQTT broker + fall-dashboard) go through the
same Traefik entrypoint on port 443 that your existing services already use.

---

## What Ports Are Used and Why

### Ports that must be open (OS firewall / cloud security group)

| Port | Protocol | Direction | Required for |
|------|----------|-----------|--------------|
| **80** | TCP | Inbound | Let's Encrypt HTTP-01 ACME challenge — needed for TLS certificate issuance. If your cluster already has HTTPS services with valid certs, this is already open. |
| **443** | TCP | Inbound | All external traffic: HTTPS to fall-dashboard + WSS to MQTT broker. Both go through the same Traefik entrypoint. If patient-dashboard already runs on 443, nothing extra to open. |

### Ports that do NOT need firewall rules

| Port | Where | Why no rule needed |
|------|-------|--------------------|
| **1883** | mosquitto pod | TCP, ClusterIP only — never leaves the cluster. Used internally by fall-dashboard to reach the broker via `mosquitto:1883`. |
| **9001** | mosquitto pod | WebSocket, internal only — Traefik receives WSS on 443 and forwards it to this port inside the cluster. Never exposed on the host NIC. |
| **8002** | fall-dashboard pod | ClusterIP only — Traefik forwards HTTPS on 443 to this port inside the cluster. |

---

## Why MQTT Does Not Need Its Own Port

A common question: MQTT usually runs on port 1883 — why is no firewall rule needed here?

The mobile app connects to the broker using **WebSocket over HTTPS (WSS)**, not raw TCP.
At the OS firewall level, WSS is indistinguishable from regular HTTPS — both are TCP:443.
Traefik receives the connection on port 443 and, based on the `Host` header (your MQTT
subdomain), routes it to the mosquitto pod on port 9001 internally.

```
Mobile app
    │  wss://mqtt.focus-hospital.de:443  (looks like HTTPS to the firewall)
    ▼
OS firewall — sees TCP:443 ALLOW — lets it through
    ▼
Traefik — Host: mqtt.focus-hospital.de → route to mosquitto:9001
    ▼
mosquitto pod :9001 (WebSocket listener, inside cluster)
```

Port 1883 (raw TCP MQTT) is never exposed. The mobile app is built with React Native
which cannot open raw TCP sockets — WebSocket is the only viable approach.

---

## Checking What Is Already Open

### ufw (Ubuntu / Debian)

```bash
sudo ufw status verbose
# Look for 80/tcp and 443/tcp in the ALLOW list
```

### firewalld (CentOS / RHEL / Fedora)

```bash
sudo firewall-cmd --list-all
# Look for ports: 80/tcp 443/tcp in the ports or services line
```

### iptables (direct)

```bash
sudo iptables -L INPUT -n -v | grep -E "80|443"
```

---

## Opening Ports If Missing

### ufw

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw reload
sudo ufw status   # confirm
```

### firewalld

```bash
sudo firewall-cmd --add-port=80/tcp --permanent
sudo firewall-cmd --add-port=443/tcp --permanent
sudo firewall-cmd --reload
sudo firewall-cmd --list-ports   # confirm
```

---

## K3s + ufw Compatibility Note

K3s manages its own iptables rules for pod-to-pod traffic (via Flannel CNI).
Enabling ufw **after** k3s is running can break inter-pod communication because
ufw's default DROP policy may block Flannel's internal traffic.

If you enable ufw and pods stop reaching each other, run:

```bash
# Allow Flannel overlay network interface
sudo ufw allow in on flannel.1
sudo ufw allow in on cni0

# Or, allow the pod CIDR explicitly (k3s default is 10.42.0.0/16)
sudo ufw allow from 10.42.0.0/16
sudo ufw allow to 10.42.0.0/16
sudo ufw reload
```

This is a k3s/ufw general concern, not specific to this deployment.

---

## Full Port Reference

| Port | Host exposure | Protocol | Traffic direction | Used by |
|------|:-------------:|----------|:-----------------:|---------|
| 80 | Yes | TCP | Inbound | Traefik (Let's Encrypt ACME) |
| 443 | Yes | TCP (HTTPS + WSS) | Inbound | Traefik → fall-dashboard + mosquitto |
| 1883 | No | TCP | Internal | fall-dashboard pod → mosquitto pod |
| 9001 | No | WebSocket | Internal | Traefik → mosquitto pod |
| 8002 | No | HTTP | Internal | Traefik → fall-dashboard pod |
| 6443 | Optional | TCP | Inbound | K3s API server (only if `kubectl` run from outside the server) |
