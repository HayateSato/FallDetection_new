---
name: Patient Dashboard — single web app shown to caregiver
description: The caregiver-facing UI is one web app (Patient Dashboard) combining FOCUS-sourced patient info and our fall detection data. Owned by FOCUS DevOps team (not Isa) — Flutter web app.
type: project
---

The "Patient Dashboard" is **one web app** shown to the end-user (caregiver). It combines two panels:

**Panel 1 — Patient info (data from FOCUS)**
- Demographic data (height, weight, etc.) — from FHIR database
- Biosignal data (HR, etc.) — from InfluxDB

**Panel 2 — Fall panel (data from our side)**
- Fall history — from our fall_dashboard API (`GET /api/falls`)
- Real-time fall alerts — via SSE (`GET /api/stream`)

**Owner correction (2026-04-29):** The Patient Dashboard is built and maintained
by the **FOCUS DevOps team**, NOT by Isa. Earlier docs incorrectly said "built
by Isa" — Isa is the **mobile app side** (separate codebase, separate audience).
The Patient Dashboard is a **Flutter web app** they already run; we ask them
to add a fall panel that consumes our REST + SSE.

The two handover docs reflect this split:
- `handover_docs/FOCUS_patient_dashboard_integration.md` → for the FOCUS team adding the fall panel (Flutter / Dart code samples)
- `handover_docs/ISA_mobile_app_contract.md` → for Isa updating the SmarKo mobile app (/predict + MQTT)
- `handover_docs/ISA_web_app_dashboards.md` → STALE on the Patient Dashboard part; the mobile-app context still applies. Refocus or annotate when next touched.

**Why:** fall_dashboard (:8002) is NOT a standalone user-facing app — it is the backend that feeds the fall panel inside the Patient Dashboard. The Patient Dashboard reads our API to embed fall data alongside the patient info it already has from FHIR and InfluxDB.

**Tech stack:** Flutter web app. Suggested packages for the fall integration:
`http` (REST) + `flutter_client_sse` (SSE). Code skeleton in
`FOCUS_patient_dashboard_integration.md` section 4.

**Naming history:** Previously called "FOCUS dashboard" in docs — renamed to "Patient Dashboard" (2026-04-15) to reflect that it is the single unified view for the caregiver.

**How to apply:** When discussing the dashboard UI or fall integration, the
counterpart is the FOCUS DevOps team. Don't route Patient-Dashboard questions
to Isa — he handles the mobile app only.
