**Port-forward**

The command is:

`kubectl port-forward svc/fall-dashboard 18002:8002`

Read it as: **"forward MY laptop's port 18002 to the pod's port 8002"**

`Your browser types: localhost:18002
                         |
                    your laptop
                    (127.0.0.1:18002)
                         |
                    kubectl acts as a middleman/tunnel
                         |
                    pod inside the cluster
                    (pod:8002 — where FastAPI listens)`

So 18002 is the door you knock on from your browser. kubectl hears that knock and passes it through to 8002 inside the pod. The pod only ever sees traffic arriving on 8002 — it has no idea 18002 existed.

The 18002 number is completely arbitrary. You could use 9999:8002 and open `localhost:9999` instead. It just needs to be a free port on your laptop.

---

**NodePort**

Port-forward requires you to run a kubectl command in a terminal — it only works while that terminal is open, and only on your own laptop.

NodePort is a permanent version that works from other machines too.

When you set `wsNodePort: 30802` in values.yaml, Kubernetes tells every node (machine) in the cluster:

> "Open port 30802 on your real network card, and forward anything arriving there into pod:8002."
> 

`Laptop 2 on same Wi-Fi
    types: 192.168.1.45:30802   (Laptop 1's real LAN IP)
                |
           Laptop 1's NIC
           (port 30802 permanently open)
                |
           Kubernetes forwards it
                |
           pod:8002 (FastAPI)`

Laptop 2 never hears about port 8002. It only knows about 30802.

---

**Side by side**

`PORT-FORWARD                          NODEPORT
────────────────────────────────      ────────────────────────────────
your laptop                           your laptop (the K8s node)

127.0.0.1:18002  ──tunnel──► pod:8002     192.168.1.45:30802  ──► pod:8002
     ▲                                          ▲
only reachable                          reachable from
from your own browser                   any machine on the LAN

only works while                        permanently open
kubectl command runs                    while cluster runs

18002 = arbitrary free port             30802 = must be in 30000-32767 range
        you chose                               (Kubernetes rule)`

---

**The relationship in one sentence each**

- Port-forward: your laptop's port 18002 is a temporary personal tunnel that leads to pod:8002.
- NodePort: your laptop's network card permanently exposes port 30802 to the LAN, which leads to pod:8002.

Both lead to the same place (pod:8002). The difference is who can use them and how long they last.

---
**Port 8002 — inside the pod (always exists)**

This is the port FastAPI listens on inside the container. It exists regardless of whether anything else is set up. Nothing outside the cluster can reach it directly.

`[pod: fall-dashboard container]
    FastAPI listening on 8002
    only reachable by other pods in the same cluster via ClusterIP`

---

**NodePort 30802 — opens a port on the host NIC**

Kubernetes takes port 8002 inside the pod and maps it to port 30802 on the laptop's actual network card. Traffic arriving at `192.168.1.45:30802` (your laptop's LAN IP) gets forwarded into the pod's port 8002.

`Laptop 2 on same Wi-Fi
    → 192.168.1.45:30802  (host NIC)
        → pod:8002`

This is how Laptop 2's server-health probe reaches fall-dashboard across the LAN.

---

**Port-forward 18002 — a tunnel on localhost only**

`kubectl port-forward svc/fall-dashboard 18002:8002` creates a tunnel that lives only on `127.0.0.1`. It does NOT open anything on the real NIC. Laptop 2 cannot reach this — only your own browser on the same laptop can.

`Your browser on Laptop 1
    → 127.0.0.1:18002  (loopback only, not on NIC)
        → kubectl tunnel
            → pod:8002`

---

**So to directly answer your questions:**

The two rows exist because NodePort and port-forward are two completely different mechanisms that happen to both lead to the same destination (pod:8002):

|  | NodePort 30802 | Port-forward 18002 |
| --- | --- | --- |
| Opens on | Host NIC (real IP) | Loopback only (127.0.0.1) |
| Reachable from | Other machines on LAN | Only your own browser |
| Requires | `wsNodePort: 30802` in values.yaml | Running `kubectl port-forward` command in terminal |
| Persistent | Yes, stays open while cluster runs | No, dies when you close the terminal |

**Can you open `127.0.0.1:8002` in the browser?**

No — that would only work if FastAPI was running directly on your laptop (like in single-laptop dev mode). In K3s, FastAPI is inside a pod. `127.0.0.1:8002` on your laptop points at nothing. You need either `127.0.0.1:18002` (port-forward) or `<your-LAN-IP>:30802` (NodePort) to reach it.