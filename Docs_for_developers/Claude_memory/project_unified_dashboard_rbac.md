---
name: Unified caregiver/admin dashboard URL with RBAC
description: Single ingress URL routes to four dashboards based on path + role. Caregiver and admin views are mutually exclusive — admins do NOT see Patient Dashboard.
type: project
---

Decision (2026-04-28): all four dashboards live under a single ingress hostname
(e.g. `dashboard.charite.de`). Path + role-claim gates which view a user sees.

| Route | View | Role | Namespace |
|-------|------|------|-----------|
| `/` | Patient Dashboard — Flutter web app (patient info + biosignals + fall panel; owned by FOCUS DevOps, NOT Isa) | **caregiver** | FOCUS |
| `/admin/ml` | `ml_dashboard` — retrain, register, promote, hot-swap | **admin** | ours |
| `/admin/health` | server health — aggregate `/health` endpoints, plain-language status | **admin** | ours |

**Why:** Reason: caregivers should not access MLOps controls that change the live
model serving real patients; admins don't need clinical patient info.
How to apply: when designing/discussing UI surface area, do not add MLOps controls
into the Patient Dashboard view, and do not add patient demographics into the
admin views. Treat the two roles as fully separated.

**Grafana stays separate** (kept as time-series ops tool, not embedded). ml_dashboard
cross-links to Grafana for "view live metrics" but does not duplicate its panels.

**Auth approach:** FOCUS SSO issues JWT with `role` claim (`caregiver` | `admin`).
Traefik middleware enforces path-level rules; each backend re-validates the role
in the JWT (defence in depth). Implementation requires coordination with FOCUS
DevOps. (Isa is mobile-app only — not involved in dashboard work.)

**Implementation status (2026-04-28):**
- `ml_dashboard` MVP built at `_6G_Integration_v2_mqtt/ml_dashboard/` (port 8004).
  Run via `python -m ml_dashboard.main`. Has retrain (subprocess + log streamer),
  registered-versions table with promote buttons, hot-swap buttons, status panel
  with drift warning, and embedded model-drift guidance.
- Auth NOT yet implemented — UI shows a warning banner. Do not expose beyond
  localhost / cluster-internal until todo.md Step 11.5.4 is done.
- `server_health` page (Step 11.5.3) and ingress wiring (Step 11.5.1) still open.

Tracked in REFACTOR_DOCUS/todo.md Step 11.5 and documented in
REFACTOR_DOCUS/deployment_architecture.md. Drift guidance also lives in
handover_docs/ADMIN_runbook_retrain_and_hotswap.md.
