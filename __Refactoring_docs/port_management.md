# Port Management Guide

Covers four environments:
1. Single laptop — dev mode (everything local)
2. Two laptops — all Docker
3. Two laptops — K3s on Laptop 1 + Docker on Laptop 2 (current local test)
4. Real production deployment

---

## Why ports differ between environments

Two rules drive all the differences:

**Rule 1 — MQTT transport:**
Within Docker's bridge network, services communicate via Docker DNS (e.g. `mqtt:1883`) using raw TCP.
Outside Docker/K8s (mobile app, mock-app on a different machine), MQTT must use **WebSocket** because
React Native cannot open raw TCP sockets in standard JS. So:

| Caller location | Reach method | Protocol | Port |
|---|---|---|---|
| Same Docker network | Docker DNS name | TCP | 1883 |
| Different machine / outside cluster | IP or domain | WebSocket | 9001 (local) or 443 (production TLS) |

**Rule 2 — Port-forward vs NodePort vs direct:**
`kubectl port-forward` creates a local tunnel on your laptop. The left number is YOUR local port;
the right number is the pod's internal port. NodePort is a completely separate mechanism that the
cluster exposes on the host NIC. They exist independently — you use whichever is convenient:

```
kubectl port-forward svc/fall-dashboard 18002:8002
                                        ^^^^^  ^^^^
                                        your   pod's internal port (always 8002)
                                        local
                                        port
```

---

## Scenario 1 — Single laptop (dev mode)

**Folder:** `_6G_Integration_v2_mqtt/`
Everything runs on one machine (Python processes in terminals + infra Docker Compose).

```
Browser / curl                Your laptop
      |
      |-- localhost:8001  -->  inference-server   (uvicorn, terminal)
      |-- localhost:8002  -->  fall-dashboard      (python -m fall_dashboard.main, terminal)
      |-- localhost:8005  -->  mock-app popup      (patient confirmation browser page)
      |-- localhost:3000  -->  Grafana             (Docker Compose)
      |-- localhost:5000  -->  MLflow UI           (Docker Compose)
      |-- localhost:9090  -->  Prometheus          (Docker Compose)
      |
      mock-app (terminal)
        |-- localhost:1883  -->  mosquitto  (Docker Compose, TCP)
        |-- localhost:8001  -->  inference-server
        |-- cloud URL      -->  InfluxDB
```

### Port table

| Service | Port | Exposed to | Note |
|---------|------|-----------|------|
| inference-server | 8001 | localhost | |
| fall-dashboard | 8002 | localhost | |
| mock-app popup | 8005 | localhost | browser confirmation UI |
| mosquitto TCP | 1883 | localhost | used by mock-app and fall-dashboard |
| mosquitto WS | 9001 | localhost | not used in this mode |
| postgres | 5432 | localhost (Docker) | |
| MLflow | 5000 | localhost | |
| MinIO API | 9000 | localhost | |
| MinIO console | 9002 | localhost | 9001 taken by MQTT WS |
| Prometheus | 9090 | localhost | |
| Grafana | 3000 | localhost | |
| ml-dashboard | 8004 | localhost | |
| server-health | 8006 | localhost | |

---

## Scenario 2 — Two laptops, all Docker

**Laptop 1 (MCS):** `_6G_integration_v3_docker_mcs/`
**Laptop 2 (FOCUS):** `_6G_integration_v3_docker_focus/`

The mock-app is included in the docker-compose on Laptop 2. It reaches the MQTT broker via
Docker's internal network (service name `mqtt`, plain TCP port 1883). The only cross-machine
traffic is mock-app → inference-server over the LAN.

```
Laptop 2 (FOCUS / Docker)                   Laptop 1 (MCS / Docker)
+---------------------------------------+   +-------------------------------+
|                                       |   |                               |
|  mock-app (focus_mock_app)            |   |  inference-server             |
|    localhost:8005  <-- browser popup  |   |    <Laptop1-IP>:8001          |
|    |                                  |   |                               |
|    |-- mqtt:1883 (TCP, Docker DNS) -->|-->| postgres (internal 5432)      |
|    |-- <Laptop1-IP>:8001 (HTTP)  ----|-->| mlflow   (internal 5000)      |
|    |-- InfluxDB cloud URL            |   | minio    (internal 9000)      |
|    |                                  |   | prometheus (internal 9090)    |
|  mqtt (focus_mqtt)                    |   | grafana   (internal 3000)     |
|    1883 -- internal Docker only       |   | ml-dashboard (internal 8004)  |
|    9001:9001 -- host NIC (WS)         |   | server-health (internal 8006) |
|    (for real mobile app / Isa)        |   |                               |
|                                       |   +-------------------------------+
|  fall-dashboard (focus_fall_dashboard)|
|    8002:8002 -- host NIC              |
|    |-- mqtt:1883 (TCP, Docker DNS)    |
|    |-- InfluxDB cloud URL             |
|                                       |
+---------------------------------------+
```

### Port table — Laptop 1

| Service | Container port | Host port | Reachable from |
|---------|---------------|-----------|----------------|
| inference-server | 8001 | **8001** | Laptop 2 (LAN) |
| postgres | 5432 | internal | fall-net only |
| mlflow | 5000 | internal | fall-net only |
| minio | 9000 | internal | fall-net only |
| prometheus | 9090 | internal | fall-net only |
| grafana | 3000 | internal | fall-net only |
| ml-dashboard | 8004 | internal | fall-net only |
| server-health | 8006 | internal | fall-net only |

### Port table — Laptop 2

| Service | Container port | Host port | Reachable from |
|---------|---------------|-----------|----------------|
| mqtt TCP | 1883 | **not exposed** | caregiver-net only (Docker DNS) |
| mqtt WebSocket | 9001 | **9001** | real mobile app / LAN |
| fall-dashboard | 8002 | **8002** | LAN / Flutter dashboard |
| mock-app popup | 8005 | **8005** | localhost browser on Laptop 2 |

### Key: why 1883 is NOT exposed on Laptop 2 host

Mock-app and fall-dashboard both live inside the same Docker Compose network (`caregiver-net`).
They reach mosquitto via Docker DNS (`mqtt:1883`, plain TCP). There is no reason to open 1883
on the host NIC — only port 9001 (WebSocket) is opened for the real mobile app.

---

## Scenario 3 — Two laptops: K3s on Laptop 1 + Docker on Laptop 2

**Laptop 1 (FOCUS / K3s):** `_6G_integration_v3_k3s/` — mosquitto + fall-dashboard as K3s pods
**Laptop 2 (MCS / Docker):** `_6G_integration_v3_docker_mcs/` — inference layer
**Mock-app:** runs as a standalone `docker run` on Laptop 2 (NOT in the docker-compose)

```
Laptop 2 (MCS)                               Laptop 1 (K3s / Docker Desktop K8s)
+------------------------------------------+  +---------------------------------------------+
|                                          |  |  Kubernetes cluster (namespace: fall-dashboard) |
|  inference-server (Docker, :8001)        |  |                                               |
|    <-- mock-app calls here               |  |  mosquitto pod                                |
|    8001:8001 on host NIC                 |  |    :1883  <-- fall-dashboard (cluster-internal)|
|                                          |  |    :9001  <-- NodePort 30901 on host NIC      |
|  mock-app (docker run, standalone)       |  |                                               |
|    -p 8005:8005  -->  popup at :8005     |  |  fall-dashboard pod                           |
|    |                                     |  |    :8002  <-- Traefik IngressRoute             |
|    |-- host.docker.internal:8001  -------|->|    :8002  <-- NodePort 30802 on host NIC      |
|    |   (routes to inference-server       |  |    :8002  <-- port-forward 18002:8002 (debug) |
|    |    running on Windows host)         |  |                                               |
|    |                                     |  |  Traefik (kube-system)                        |
|    |-- ws://<Laptop1-IP>:30901  ---------|->|    :80 / :443 (standard HTTP/HTTPS)           |
|        (MQTT WebSocket, NodePort)        |  |    /falls --> fall-dashboard:8002             |
|                                          |  |                                               |
+------------------------------------------+  +---------------------------------------------+
```

### Why `host.docker.internal` instead of `localhost`

When mock-app runs inside a Docker container on Laptop 2, `localhost` refers to the container
itself — not to Windows. `host.docker.internal` is Docker Desktop's special hostname that
routes from inside the container back to the Windows host where inference-server is running.

### Three ways to reach fall-dashboard (Laptop 1)

| Access method | Port | From | When to use |
|---|---|---|---|
| **NodePort** `<Laptop1-IP>:30802` | 30802 (local) → 8002 (pod) | Laptop 2 / LAN | server-health probe, API calls from outside (used to run a kbectl command in a terminal)  |
| **Port-forward** `localhost:18002` | 18002 (local) → 8002 (pod) | Laptop 1 terminal | quick curl / debug without opening firewall |
| **Traefik IngressRoute** | 80 / 443 | LAN | Flutter dashboard path `/falls` |

### Port table — Laptop 1 (K3s)

| Service | Pod internal port | Exposed as | Port | Accessible from |
|---------|:-----------------:|-----------|:----:|----------------|
| mosquitto TCP | 1883 | ClusterIP | 1883 | fall-dashboard pod only (cluster DNS) |
| mosquitto WS | 9001 | **NodePort** | **30901** | LAN (mock-app, real mobile app) |
| fall-dashboard | 8002 | **NodePort** | **30802** | LAN (optional, for debugging) |
| fall-dashboard | 8002 | **Port-forward** | **18002** (local) | localhost only |
| fall-dashboard | 8002 | **Traefik** `/falls` | 80/443 | Flutter dashboard |

### Port table — Laptop 2 (Docker)

| Service | Host port | Reachable from | Note |
|---------|-----------|----------------|------|
| inference-server | **8001** | Laptop 1 LAN | opened by firewall rule |
| mock-app popup | **8005** | localhost browser | `-p 8005:8005` on docker run |

### Firewall rules needed

```powershell
# Laptop 1 — open NodePorts
New-NetFirewallRule -DisplayName "MQTT WS NodePort" -Direction Inbound -Protocol TCP -LocalPort 30901 -Action Allow
New-NetFirewallRule -DisplayName "Fall Dashboard NodePort" -Direction Inbound -Protocol TCP -LocalPort 30802 -Action Allow

# Laptop 2 — open inference-server
New-NetFirewallRule -DisplayName "Inference Server" -Direction Inbound -Protocol TCP -LocalPort 8001 -Action Allow
```

---

## Scenario 4 — Real production

**MCS (Hetzner):** `_6G_integration_v3_docker_mcs/` behind a reverse proxy (nginx/Traefik)
**FOCUS k3s cluster:** `_6G_integration_v3_k3s/` installed as a Helm release

No NodePorts in production. All external traffic goes through Traefik on port 443.

```
Internet / mobile app (Isa)                   FOCUS k3s cluster
                                              +----------------------------------+
  HTTPS POST /predict  ----> MCS Hetzner      |  Traefik (port 443)             |
  (inference-server)         :443 via proxy   |    wss://<domain>  --> :9001    |
                                              |    (WSS terminates TLS,         |
  MQTT (after fall)   ----------------------->|     forwards WS to mosquitto)   |
  wss://<domain>:443                          |                                 |
                                              |  mosquitto pod                  |
                                              |    :1883 -- internal TCP only   |
                                              |    :9001 -- receives WS from    |
                                              |             Traefik             |
                                              |                                 |
                                              |  fall-dashboard pod             |
                                              |    :8002 -- internal only       |
                                              |                                 |
  Flutter dashboard ----------------------->  |  Traefik /falls --> :8002       |
  (inside FOCUS cluster,                      |  (cluster-internal HTTP call,   |
   not external)                              |   no TLS needed inside cluster) |
                                              +----------------------------------+

MCS Hetzner
+----------------------------------+
|  reverse proxy (port 443)        |
|    /predict --> inference:8001   |
|    /confirm --> inference:8001   |
|                                  |
|  inference-server  (internal)    |
|  postgres          (internal)    |
|  mlflow            (internal)    |
|  minio             (internal)    |
|  prometheus        (internal)    |
|  grafana           (internal)    |
|  ml-dashboard      (internal)    |
|  server-health     (internal)    |
+----------------------------------+
```

### Port table — production

| Service | Internal port | External port | Protocol | Who reaches it |
|---------|:-------------:|:-------------:|----------|----------------|
| inference-server (MCS) | 8001 | **443** | HTTPS | real mobile app |
| mosquitto TCP (FOCUS) | 1883 | none | TCP | fall-dashboard pod (internal) |
| mosquitto WS (FOCUS) | 9001 | **443** | WSS via Traefik | real mobile app |
| fall-dashboard (FOCUS) | 8002 | **443** `/falls` | HTTPS via Traefik | Flutter dashboard (inside cluster) |
| all others (MCS) | various | none | internal | docker-net only |

### Key differences from local testing

| | Local (K3s test) | Production |
|--|---|---|
| MQTT mobile app port | `ws://<IP>:30901` NodePort | `wss://<domain>:443` via Traefik |
| fall-dashboard external | NodePort 30802 or port-forward | HTTPS :443 via Traefik IngressRoute |
| mock-app | runs on Laptop 2, simulates mobile app | replaced by Isa's real React Native app |
| TLS | none | Traefik + Let's Encrypt (certResolver) |
| `imagePullPolicy` | `IfNotPresent` (local image) | `Always` (registry pull) |
| `imagePullSecret` | blank | `mcs-labs` |
| `wsNodePort` in values.yaml | `30901` | blank (ClusterIP, Traefik handles external) |
| `httpNodePort` in values.yaml | `30802` | blank (ClusterIP, Traefik handles external) |

---

## Summary — MQTT port by environment

| Environment | Mock-app/mobile connects to | Transport | Port |
|---|---|---|---|
| Single laptop | `localhost:1883` | TCP | 1883 |
| Two laptops, all Docker | `mqtt:1883` (Docker DNS) | TCP | 1883 |
| Two laptops, K3s | `ws://<Laptop1-IP>:30901` | WebSocket | 30901 (NodePort) |
| Production | `wss://<domain>:443` | WSS (TLS) | 443 (Traefik) |

> React Native cannot open raw TCP sockets in standard JS. All external (non-Docker-internal)
> MQTT connections MUST use WebSocket — only the internal Docker-to-Docker case can use TCP 1883.
