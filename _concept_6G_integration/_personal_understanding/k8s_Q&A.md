`OUR NAMESPACE
│
├── Pod: inference-server-7d4f2  ← only inference_server runs here
├── Pod: inference-server-8c1a3  ← identical copy (replica for HA/load balancing)
├── Pod: inference-server-9b2e1  ← identical copy
│
├── Pod: caregiver-client-3f9d1  ← only caregiver_client runs here (1 replica — stateful timer)
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
- **caregiver_client** — also 1 pod for now. It holds MQTT subscription state and writes to Postgres. You *could* scale it but you'd need to handle duplicate DB writes.
- **MLflow, Prometheus, Grafana** — monitoring/tracking tools that manage their own internal state. Typically run as 1 pod in this kind of setup.

The thing that connects them is the Kubernetes `Service` object — it gives each pod a stable internal hostname so other pods can find it:

`inference_server  →  "postgres:5432"       (resolves to the Postgres pod)
inference_server  →  "mqtt-broker:1883"    (resolves to the MQTT pod)
caregiver_client  →  "postgres:5432"       (same Postgres pod)`

No pod knows or cares about the IP address of another pod — they just use the service name, which is why the config in `.env` uses hostnames like `localhost` in dev and will use service names like `postgres` in the Helm chart.