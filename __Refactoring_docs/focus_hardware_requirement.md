# FOCUS Hardware Requirements — Caregiver Layer

- RAM hard limit: 384Mi total (~403 MB) — well within FOCUS's budget
- RAM actual usage at idle: ~192Mi (requests: 64Mi + 128Mi)
- Disk: 1Gi total across two PVCs — the mosquitto PVC stores MQTT persistence data, the fall-dashboard PVC stores only the SQLite patient list (a few KB in practice)
