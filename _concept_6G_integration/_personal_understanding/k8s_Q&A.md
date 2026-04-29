`OUR NAMESPACE
│
├── Pod: inference-server-7d4f2  ← only inference_server runs here
├── Pod: inference-server-8c1a3  ← identical copy (replica for HA/load balancing)
├── Pod: inference-server-9b2e1  ← identical copy
│
├── Pod: caregiver-client-3f9d1  ← only fall_dashboard runs here (1 replica — stateful timer)
│
├── Pod: mqtt-broker-1a2b3       ← only mosquitto runs here (1 replica)
│
├── Pod: postgres-0              ← only Postgres runs here (StatefulSet, 1 replica)
│
├── Pod: mlflow-5c3d2            ← only MLflow tracking server runs here
├── Pod: prometheus-6e4f1        ← only Prometheus runs here
├── Pod: grafana-7a5b2           ← only Grafana runs here
└── Pod: minio-8d6c3             ← only MinIO runs here`

So the inference server gets 3 pods because it needs fault tolerance and can handle load balancing. Everything else gets 1 pod because:

- **Postgres, MQTT broker, MinIO** — stateful. Running two copies without special clustering logic would cause data conflicts. One pod, managed as a `StatefulSet` with a persistent volume.
- **fall_dashboard** — also 1 pod for now. It holds MQTT subscription state and writes to Postgres. You *could* scale it but you'd need to handle duplicate DB writes.
- **MLflow, Prometheus, Grafana** — monitoring/tracking tools that manage their own internal state. Typically run as 1 pod in this kind of setup.

The thing that connects them is the Kubernetes `Service` object — it gives each pod a stable internal hostname so other pods can find it:

`inference_server  →  "postgres:5432"       (resolves to the Postgres pod)
inference_server  →  "mqtt-broker:1883"    (resolves to the MQTT pod)
fall_dashboard  →  "postgres:5432"       (same Postgres pod)`

No pod knows or cares about the IP address of another pod — they just use the service name, which is why the config in `.env` uses hostnames like `localhost` in dev and will use service names like `postgres` in the Helm chart.


In Kubernetes the hierarchy is:

`Deployment  (desired state: "I want 3 replicas of this container")
    └── ReplicaSet  (keeps the count at 3)
            └── Pod  (one running instance of the container)
            └── Pod
            └── Pod`

A **pod is the smallest deployable unit** — one running instance of your application (usually one container, sometimes a few tightly-coupled containers that share a network). So yes, "the InfluxDB instance running in the FOCUS namespace" is a pod (specifically a `StatefulSet` pod since it has persistent storage).

The distinction that matters in practice:

- **One Deployment = potentially many pods** (replicas for high availability). The inference server might run as 3 pods — they're all the same "instance" of the app from a service perspective, but each is a separate pod.
- **One StatefulSet pod = one DB instance** — for Postgres and InfluxDB you normally run 1 pod per instance (primary), with replicas handled by the DB itself, not by K8s replicas.

So "pod = instance" is a good mental model for stateful things like databases. For stateless services like the inference server, it's more accurate to say "a pod is one replica of the instance".


Both, and they work together. Let me explain each role.

**Fault tolerance (one goes down):**
If you run 3 pods and one crashes, Kubernetes detects it within seconds and the other 2 keep serving requests while K8s restarts the failed pod. With only 1 pod, a crash means downtime until the pod restarts — which could be 30–60 seconds. For a medical alert system that's unacceptable.

**Load balancing (traffic distribution):**
A Kubernetes `Service` sits in front of the pods and acts as a load balancer. Incoming requests are distributed across all healthy pods so no single pod gets overwhelmed.

`Mobile App
    │  POST /predict
    ▼
 Service  (ClusterIP — load balancer)
    │
    ├──► inference_server pod 1  (handling request A)
    ├──► inference_server pod 2  (handling request B)
    └──► inference_server pod 3  (idle)`

**Why both matter for your system specifically:**

The inference server is stateless — each `/predict` call is independent, no shared memory between requests. This makes it safe to run multiple pods. If pod 1 is busy running XGBoost inference (which takes 10–50ms), pod 2 can immediately handle the next request. Without multiple pods, requests queue up behind each other.

**The important constraint for your system:**

Your inference server currently runs `--workers 1` (single uvicorn worker). This was required in the full system because the auto-confirm timer was per-process. That constraint no longer applies in the MQTT version — confirmation is handled by the mobile app. So in production you could safely run 3 pods, each with a single worker, and the K8s Service load-balances across them.

The one thing that does need care is the `POST /model/switch` endpoint — switching a model on pod 1 does not affect pod 2 or pod 3. If you hot-swap, you'd need to call the endpoint on each pod, or use a shared signal (e.g. a config map watch). This is worth noting for Step 7.3.

----

## The metaphor

| Term | Recipe analogy | Software analogy |
| --- | --- | --- |
| **Chart** | a cake recipe (in a book) | a Python package on PyPI |
| **Release** | a specific cake you baked using that recipe | a specific install of the package in a specific environment |
| **Target namespace** | which kitchen you used | which environment you installed it in |

A chart is **static** (lives in Git). A release is a **running instance** of that chart in your cluster. The target tells K8s **where** to put the release's resources.

## In your specific case

`helm install mcs-fall-detection .\helm\fall-detection `
             ^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^
             |                  |
             |                  └─ CHART path on disk
             |                     (Chart.yaml says name: fall-detection)
             |
             └─ RELEASE name
                (this specific install of the chart)
       --namespace mcs-fall-detection
                  ^^^^^^^^^^^^^^^^^^
                  └─ TARGET namespace
                     (where the resources land in K8s)`

So you have:

- **Chart:** `fall-detection` (the recipe)
- **Release:** `mcs-fall-detection` (one specific cake baked from it)
- **Target:** `mcs-fall-detection` (the kitchen you used)

## Why this matters

You can install the **same chart multiple times** with different release names + targets — they're independent instances:

`helm install dev-fall-detection ./helm/fall-detection --namespace dev
helm install qa-fall-detection  ./helm/fall-detection --namespace qa
helm install prod-fall-detection ./helm/fall-detection --namespace prod`

Three releases, three namespaces, **one** chart. Same recipe, three different cakes in three different kitchens.

## How to inspect each

| What | Command |
| --- | --- |
| All charts in a folder | `ls ./helm/` |
| Chart's metadata | `cat ./helm/fall-detection/Chart.yaml` |
| All releases (running) | `helm list -A` |
| Specific release's status | `helm status mcs-fall-detection -n mcs-fall-detection` |
| All namespaces | `kubectl get namespaces` |
| Resources in a namespace (release's children) | `kubectl get all -n mcs-fall-detection` |

The chart never appears in `kubectl` output — only its rendered K8s resources do. The release name is what helm uses internally to track "I created these specific resources." The namespace is just metadata on each resource.

## TL;DR

Chart = blueprint (in Git). Release = running instance. Namespace = where it runs.

You can change the release/target without touching the chart, and you can change the chart without affecting any existing releases until you `helm upgrade`.

## Three different naming layers

These are all separate, even though they often look alike:

| Name | What it is | Today's value |
| --- | --- | --- |
| **Chart folder** | filesystem path to the chart | `helm/fall-detection/` |
| **Chart name** (in `Chart.yaml` `name:`) | the chart's identity, used by helm internally | `fall-detection` |
| **Release name** (in `helm install <name>`) | the specific install of the chart | `mcs-fall-detection` |
| **Target namespace** (in `--namespace`) | where the chart's resources land | `mcs-fall-detection` |

These can be all-different, all-same, or partially-overlapping. **There's no requirement** that they match. You can `helm install mcs-fall-detection ./helm/fall-detection --namespace mcs-fall-detection` and it works fine — that's actually what you've been doing.

## Convention in the helm world

Public charts almost always name the chart after the **software**, not the **deployer**:

`postgresql       (the software, not "bitnami-postgresql")
redis            (the software, not "myorg-redis")
nginx-ingress    (the function, not "nginx-mcs")`

The reasoning: the chart is the deployment recipe for the software. The deployer / namespace is whoever installs it. Same chart, many possible installs.

By that convention, `fall-detection` is the right name — it describes WHAT the chart deploys (the fall detection system). When FOCUS DevOps picks up the chart, "fall-detection" reads naturally. `mcs-fall-detection` would feel like our internal branding bleeding into a deliverable they own.

## My recommendation: keep `fall-detection`

For the chart folder + `Chart.yaml` name:

- It describes the software, not the deployer
- Matches helm community convention
- The release name and namespace already give you the "mcs-" identity at install time
- Renaming touches a lot of files for purely cosmetic gain

## ...but if you want to rename anyway

It's a defensible choice — consistency has its own value, and you're not shipping to FOCUS this week. If you do it, here's what changes:

`# 1. Move the folder
git mv helm/fall-detection helm/mcs-fall-detection

# 2. Update Chart.yaml (the `name:` field MUST match the folder per convention,
#    helm lint will warn otherwise)
#    Change line 2 from:    name: fall-detection
#    to:                    name: mcs-fall-detection

# 3. Update the install command in your README + scripts
#    helm install mcs-fall-detection ./helm/mcs-fall-detection ...

# 4. Search and replace path references
Select-String -Path . -Pattern "helm/fall-detection" -Recurse
# update any docs that mention the path`

Files that mention the chart path:

- `_6G_Integration_v2_mqtt/README.md` (folder structure)
- `_6G_Integration_v2_mqtt/helm/fall-detection/README.md` (the README we just wrote)
- `helm/mock-focus/install.ps1` (installs both charts)
- `REFACTOR_DOCUS/helm_guide.md`
- `REFACTOR_DOCUS/todo.md`
- `handover_docs/Tech_integrator.md`
- Possibly more — the `Select-String` above will find them all.

That's about 15–20 minutes of work. Not difficult, just touchy because of the breadth.

## Practical answer for right now

**Don't rename today.** Run the `helm install` first and confirm the chart actually works. If it deploys cleanly, the working version becomes a known-good baseline. *Then* renaming is a low-risk cosmetic change you can do later (or never, if you decide it's fine).

If you rename today and the install fails, you'll be debugging two things at once — was it the rename or a real chart bug? Leave the rename for after the smoke test passes.

----


## StatefulSet vs PVC — they are different layers

Think of it like a computer:

| Concept | Analogy | What it is |
| --- | --- | --- |
| **StatefulSet** | The computer itself | A workload controller — manages running pods |
| **PVC** | The hard drive | Storage — a disk attached to a pod |

`postgres` is a **StatefulSet** — it's the running PostgreSQL process (the app).
That StatefulSet *uses* a **PVC** for its data files (`/var/lib/postgresql/data`).

The PVC is the disk. The StatefulSet is the application running on that disk. They are layered — StatefulSet owns/requests the PVC via `volumeClaimTemplates` in the spec.

**Who are the PVCs in your cluster?** Run this:

`kubectl get pvc -n mcs-fall-detection`

You'll see something like:

`NAME                    STATUS   VOLUME   CAPACITY
postgres-data-postgres-0   Bound    ...      10Gi
minio-data-minio-0         Bound    ...      20Gi`

The PVC names follow the pattern `<claim-name>-<pod-name>`. One PVC per StatefulSet pod.

---

## Breaking down the DNS name

`postgres-0 . postgres . mcs-fall-detection . svc . cluster.local`

| Part | Meaning |
| --- | --- |
| `postgres-0` | **Pod name** — StatefulSet names pods `<name>-0`, `<name>-1`... This targets a specific pod |
| `postgres` | **Service name** — the headless service |
| `mcs-fall-detection` | **Namespace** |
| `svc` | "This is a service DNS record" (fixed K8s keyword) |
| `cluster.local` | Cluster domain suffix (fixed, always this) |

This resolves directly to **the pod's IP** — no virtual IP in between.

---

## How it looks for a regular Deployment

`inference-server . mcs-fall-detection . svc . cluster.local`

There is **no pod prefix**. This resolves to the **virtual ClusterIP** (`10.108.214.68`), which load-balances across all replicas. You never target a specific pod.

`StatefulSet headless:   postgres-0.postgres.mcs-fall-detection.svc.cluster.local  → pod IP directly
Deployment ClusterIP:   inference-server.mcs-fall-detection.svc.cluster.local     → virtual IP → any pod`

---

## Why does StatefulSet need this?

A regular Deployment doesn't care which pod handles a request — they're identical and stateless. But a database pod is **not** interchangeable:

- `postgres-0` has its data on a specific PVC
- If K8s load-balanced across pods randomly, a request might hit the wrong pod (one without the data)
- The stable DNS name `postgres-0` guarantees you always reach the same pod with the same disk

That's the entire point of StatefulSet: **stable identity** (name + storage) across restarts.



----


Two separate questions — let me answer both fully.

---

## `docker save` — what it did and whether you need a registry

**Yes, it saved the image as a file on your PC.** `docker save` exports a Docker image into a `.tar` archive — all layers, metadata, everything. It's a complete portable snapshot of the image.

**Why the first command failed but this worked:**

`# This FAILED — tried to pipe into a "kind-control-plane" container
docker save ... | docker exec -i $(docker ps --filter name=kind-control-plane -q) ctr ...
#                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                  This finds zero containers → empty string → error`

The first command assumed you were using **kind** (a different K8s tool that runs as a Docker container). You're on Docker Desktop K8s, so that container doesn't exist. `docker save` itself always works — it just writes to disk.

**Can you share `.tar` files instead of a registry?**

Technically yes. Your partner would run:

`docker load -i fall-dashboard.tar`

But **don't do this for a real handoff**. Here's why:

|  | `.tar` file sharing | Container registry |
| --- | --- | --- |
| Versioning | Manual, filename-based | Automatic via tags |
| Update | Re-send the whole file (~GB) | `docker push` + `helm upgrade` |
| K8s pull | Must manually load on every node | K8s pulls automatically |
| Multiple nodes | Copy to every node manually | Registry handles it |
| Audit/rollback | You manage files | Registry keeps history |

The registry is not about making it public — companies run **private registries** (GitHub Container Registry, AWS ECR, Azure ACR). FOCUS DevOps almost certainly has one. The `.tar` approach breaks down the moment you have more than one server or need to update anything.

---

## Headless service — what is "head"?

"Head" = the single virtual IP that sits in front of pods and receives all traffic.

`Normal ClusterIP service:
  client → virtual IP (10.108.214.68) → [K8s load balances] → pod-a
                                                              → pod-b
                   ↑ this virtual IP is "the head"

Headless service (clusterIP: None):
  client → DNS → pod-a IP directly
               → pod-b IP directly
                   ↑ no virtual IP — "no head"`

The "head" is removed. DNS returns the actual pod IPs, not a virtual one. K8s does not load balance — the client connects directly to a specific pod.

---

## PVC vs Service — I was NOT using them interchangeably

They are completely different layers. I mentioned both in the same explanation which caused confusion. Let me separate them:

| Layer | Name | What it does |
| --- | --- | --- |
| **Network** | Service | Routes traffic to pods |
| **Storage** | PVC | Attaches a disk to a pod |

In my DNS breakdown I said:

- `postgres` = **the service name** (network layer — how other pods find postgres on the network)
- `postgres-0 has its data on a specific PVC` = **the storage layer** (why pod identity matters — pod-0 owns disk-0)

A Deployment also has a Service (`inference-server` service routes to inference-server pods). A Deployment can also use a PVC, though it's uncommon for stateless apps.

The reason StatefulSet is special is it **guarantees both are stable**:

- `postgres-0` always gets the same DNS name (network identity)
- `postgres-0` always gets the same PVC (storage identity)

If K8s kills and restarts the pod, it comes back as `postgres-0` attached to the same disk. A Deployment pod gets a random name on restart and could attach to any disk.

is my understadnig correct?
in k8s, each service (svc = which means, what is declared as deployment in yml file?) assigned to its own virtual IP (or also called cluster IP) (is it cluster = service in k8s?). becasue of this IP, each service can talk to each other (pod lives inside this service or cluster, so if any pod in cluster A needs to talk to pod b in cluster B, then it needs to go the IP of cluster B). (namespace is a bigger box that contains these cluster) . However, for statefulSet, it does not have any IP and each pod has its own IP so a pod in cluster A can talk directly (but then why do i see postgres in postgres-0.postgres.mcs-fall-detection.svc.cluster.local?). pods of the same statefulSet is linked to one same PVC, so that the same datasource is shared.

in your answer DNS was shown in Headless service, but i did not see it in Normal ClusterIP service. is DNS in use only for headless service?

1. 
    - 
- 
1. 
    - 
- 
1. 
    - 
- 
1. 
    - 
- 
1. 
    - 
- 
1. 
    - 
- 
1. 
    - 
- 
1. 
    - 
- 
1. 
    - 
- 

Your understanding has some good intuitions but several important mix-ups. Let me fix them one by one.

---

## The hierarchy (you had it inverted)

`Cluster  (the entire K8s environment — one per installation)
  └── Namespace  (logical subdivision — mcs-fall-detection, focus-ns)
        └── Pod          (running container)
        └── Service      (network routing rule)
        └── Deployment   (tells K8s how to create/manage pods)
        └── StatefulSet  (same, but for stateful apps)
        └── PVC          (storage)`

Namespace is **inside** the cluster, not the other way around.

---

## Service ≠ Deployment — they are separate objects

> "svc = what is declared as deployment in yml file?"
> 

No. They are two separate YAML files/resources:

- **Deployment** — describes *what to run* (which container image, how many replicas, env vars)
- **Service** — describes *how to reach it* on the network (which port, which pods to route to)

A pod doesn't live "inside" a service. A service just points to pods via a **label selector**:

`Service "inference-server"
  selector: app=inference-server   ← finds pods with this label
  port: 8001
        ↓ routes traffic to ↓
  Pod (app=inference-server, IP=10.1.2.3)
  Pod (app=inference-server, IP=10.1.2.4)`

The pods exist independently. The service is just a routing rule.

---

## ClusterIP ≠ Cluster

> "is it cluster = service in k8s?"
> 

No.

- **Cluster** = the entire K8s environment
- **ClusterIP** = a *type* of Service that gets a virtual IP reachable only inside the cluster

Other service types exist (NodePort, LoadBalancer) — ClusterIP is just the internal-only one.

---

## StatefulSet DOES have a service — headless just means no virtual IP

> "for statefulSet, it does not have any IP"
> 

The **headless service** has no virtual IP. But:

- The StatefulSet still has a **Service object** (named `postgres`) — that's where `postgres` appears in the DNS name
- Each **pod** still has its own real IP

The headless service exists purely to provide DNS. It just doesnves resolve to a virtual IP — it resolves to pod IPs directly.

That's why you see `postgres` in `postgres-0.postgres.mcs-fall-detection.svc.cluster.local`:

`postgres-0          . postgres           . mcs-fall-detection . svc . cluster.local
↑ pod name            ↑ service name       ↑ namespace
  (StatefulSet          (headless service    
   gives it this)       still exists!)`

---

## Each StatefulSet pod gets its OWN separate PVC — not shared

> "pods of the same statefulSet is linked to one same PVC, so that the same datasource is shared"
> 

This is the **opposite** of how it works. Each pod gets its **own dedicated PVC**:

`postgres-0  →  PVC: postgres-data-postgres-0  (its own disk)
postgres-1  →  PVC: postgres-data-postgres-1  (completely separate disk)`

If they shared one disk, two postgres processes would corrupt each other's data. The whole point of StatefulSet is stable, **dedicated** storage per pod. The pod and its disk are permanently paired — even if the pod restarts, it reattaches to the same PVC.

---

## DNS works for ALL services, not just headless

> "DNS was shown in Headless service, but i did not see it in Normal ClusterIP service. is DNS in use only for headless service?"
> 

DNS works for every service in K8s. I just didn't show it explicitly for ClusterIP. Both are valid:

`# ClusterIP service — DNS resolves to virtual IP
inference-server.mcs-fall-detection.svc.cluster.local  →  10.108.214.68 (virtual IP) → pods

# Headless service — DNS resolves to pod IPs directly
postgres-0.postgres.mcs-fall-detection.svc.cluster.local  →  10.1.x.x (actual pod IP)`

In practice inside the cluster, pods just use the short name: `postgres:5432` or `inference-server:8001` — K8s fills in the rest automatically.

---

## Corrected mental model

`Pod A (in inference-server Deployment) wants to reach Postgres:

  uses DNS "postgres:5432"
       ↓
  K8s resolves to pod IP of postgres-0 directly (headless)
       ↓
  connects to postgres-0 pod, which has its own dedicated PVC (disk) mounted`