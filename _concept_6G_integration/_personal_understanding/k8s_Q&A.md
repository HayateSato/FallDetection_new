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