---
name: handover_docs_2 conventions
description: handover_docs_2/ uses numbered topic files (01_k8s.md ... 08_user_flow_admin.md) plus a running Q&A.md. Active replacement for handover_docs/.
type: project
originSessionId: e8f08216-13fd-4c07-b839-eb1955ecf810
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

**Per-audience folders at repo root (added 2026-04-29):** for individual colleagues taking over specific scopes, the project also uses per-audience folders at the repo root with numbered files inside — `isa/00_ISA_*.md` (mobile app), `mohammed/00_MOHAMMED_*.md` (K8s integration takeover). This is a hybrid of the older flat `handover_docs/ISA_*.md` and the topic-based `handover_docs_2/`: per-person folder, but numbered files inside (00 = quickstart / first read; 01, 02 = deeper topics).

**Entry point:** the repo-root `README.md` has a "Start here — pick your role" table that routes Mohammed/Isa/FOCUS DevOps to their starting docs. Update that table when adding new audience folders.
