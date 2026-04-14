**Meeting minutes summary:**

---

**Meeting — 2026-04-13 Key Takeaways**

| Topic | What was confirmed |
| --- | --- |
| Dashboard roles | Two roles: **Admin** (service health + ML model only, no patient data) and **Caregiver** (patient data scoped to their group/floor) |
| Data fetcher | Must be **separated** — their existing fetcher serves the dashboard; a new separate fetcher is needed for the inference pipeline |
| Deployment | **Kubernetes + Helm Charts**, same namespace as their current system. They want a single `helm install` to add all our components. No Docker Compose. |
| Real-time events | **MQTT**, not Redis/SSE. MQTT enables bidirectional communication between the mobile app and inference component |
| Infrastructure | 12-core, 32GB CPU server. InfluxDB is in the same Kubernetes namespace |
| InfluxDB tags | `macAddress` + `Patient ID` confirmed. Bucket name + field names → ask **Isa** |
| Dashboard developer | Possibly **Andreea** — needs confirmation |
| Architecture | See `6G_architecture_overview.png` — our additional components are: Data Fetcher (for inference), API Caller, Inference API, Event Publisher, Event Subscriber, Model Updater |

---
