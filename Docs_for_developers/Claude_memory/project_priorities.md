---
name: Development priorities and completion status
description: All three priorities complete as of 2026-03-26. Records what was built and why.
type: project
---

All three development priorities are complete as of 2026-03-26.

**Priority 1 — Operator dashboard + ml_server communication — COMPLETE**
Fully working model switching, recent inference log from PostgreSQL, processing config controls, CSV replay via MinIO, model comparison sub-page with interactive Plotly charts.

**Priority 2 — Model comparison — COMPLETE**
Built as a built-in operator dashboard sub-page (`/operator/model_comparison.html`) with 5 Plotly charts (fall rate bar, latency grouped bar, confidence box plot, histogram, scatter) + percentile table + per-recording × model matrix. Fetches from `GET /model/comparison`. Decision: NOT Grafana — simpler architecture, no datasource UID complexity.

**Priority 3 — Patient feedback + caregiver dashboard + emergency alerts — COMPLETE**
- Patient feedback dashboard (`/patient/`) with 10s fall popup, YES/NO flow, POST to `/api/ml/patient/feedback/{id}`
- Two-channel Redis: `patient_alerts` (every fall → patient) and `fall_events` (conditional → caregiver/emergency)
- 12-second asyncio timer in ml_server: auto-alerts emergency if no patient response
- Caregiver dashboard rewritten: two-tab (patient list + fall history), filters by time range/patient/user_fall/need_help, feedback columns in all tables
- Emergency tablet only alerted on `fall_events` (conditional), never on every detection

**Why:** Patient feedback loop closes the loop — avoids false-positive emergency alerts when the patient is fine. Adds ground truth labels (`user_fall`, `need_help`) to `inference_log` for future retraining.

**How to apply:** All planned work is done. Future sessions should focus on testing, bug fixes, or new features the user requests. The next likely area is either automated retraining using `user_fall` labels, or production deployment hardening (SSL, secrets management, multi-worker timer fix).
