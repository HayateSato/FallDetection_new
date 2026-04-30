---
name: handover_docs_2 conventions
description: handover_docs_2/ uses numbered topic files (01_k8s.md ... 08_user_flow_admin.md) plus a running Q&A.md. Active replacement for handover_docs/.
type: project
---

`handover_docs_2/` is the active hand-off folder (replaces the older per-audience `handover_docs/`).

**Convention:** numbered topic files for stable reference docs, plus one running Q&A:
- `01_k8s.md` — Kubernetes deployment
- `02_fall_detection_algorithm.md` — model + retraining
- `03_fall_detection_system.md` — architecture overview
- `04_mobile_app_integration.md` — what the mobile app must implement
- `05_web_app_integration.md` — what the Patient Dashboard must implement
- `06_user_flow_patient.md`, `07_user_flow_caregiver.md`, `08_user_flow_admin.md` — role-based flows
- `Q&A.md` — running list of handover questions, newest at top, each entry self-contained with code refs

**Why:** the old `handover_docs/` was per-audience (ADMIN_..., ISA_..., Tech_integrator.md), which produced a lot of duplication. `handover_docs_2/` is per-topic so each subsystem is described once, and Q&A.md absorbs the ad-hoc questions that don't fit a single topic.

**How to apply:** when the user asks a handover question (and the answer would be useful to a future reader), add it as a new entry at the top of `Q&A.md` rather than starting a new file. Use a new numbered file only if the topic is large enough to warrant its own page.
