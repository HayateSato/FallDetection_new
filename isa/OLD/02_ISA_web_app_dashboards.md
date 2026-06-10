# Web App Side — Context for Isa

**Purpose:** make Isa aware of the web-app side of the system so he understands where the fall events his mobile app emits actually surface, and what each dashboard does. **Isa does not build or own the Patient Dashboard.** That belongs to the FOCUS DevOps team (see [`FOCUS_patient_dashboard_integration.md`](../handover_docs/FOCUS_patient_dashboard_integration.md)). This doc is reference + context only.

**Reading order:**
1. Section 1 — who owns what (corrects an earlier misunderstanding)
2. Section 2 — the API contract Isa's events end up consuming
3. Section 3 — the new clinical 3-state requirement (in flight)
4. Section 4 — how to verify your mobile-app changes show up correctly on the web side

If you implement something on the mobile side and want to confirm it lights up on the dashboard, sections 3 and 4 are what you need.

---

## 1. The web app contains five dashboards — corrected ownership

| # | Dashboard | Owner | Purpose | Backend |
|---|-----------|:-----:|---------|---------|
| 1 | **Patient Dashboard** | **FOCUS DevOps** (Flutter web app) | Per-patient personal info (gender, BMI, age) + physiological live readings (HR, SpO2, ACC) + **fall status indicator** | FHIR + InfluxDB + our `fall_dashboard` |
| 2 | **Fall Dashboard** | MCS (us) | All fall events, filterable by period / patient / help-requested | our `fall_dashboard` (port 8002) |
| 3 | **ML Dashboard** | MCS (us) | Admin: retrain models, hot-swap to production | our `ml_dashboard` (port 8004) |
| 4 | **Server Health Dashboard** | MCS (us) | Admin: aggregate service liveness | our `server_health` (port 8006) |
| 5 | **Grafana Dashboard** | MCS (us) | Admin: ML model performance metrics + drift charts | Grafana (port 3000) |

**Important correction (2026-04-29):** earlier docs said the Patient Dashboard was Isa's responsibility. **It is not.** The Patient Dashboard is the FOCUS DevOps team's existing Flutter-web app — they own it, they update it, they decide its tech stack. Isa's responsibility is the **mobile app only** ([`ISA_mobile_app_contract.md`](ISA_mobile_app_contract.md)). The integration guide for the FOCUS Patient Dashboard team is [`FOCUS_patient_dashboard_integration.md`](../handover_docs/FOCUS_patient_dashboard_integration.md) — written for them, with Flutter / Dart code samples.

**Role-based access** (relevant to all dashboards):
- **Caregivers** see dashboards 1 + 2 only.
- **Admins** see dashboards 3 + 4 + 5 only.
- The dashboards share the same web app shell (URL prefix, auth, navigation), but the role determines which menu items are visible.

The role split needs coordination with FOCUS DevOps for SSO / JWT claim names. For the mobile app this is only relevant insofar as the mobile-app may need to call APIs with a JWT — but that's covered in the mobile-app contract.

---

## 2. Where the fall events Isa publishes end up — the API contract

Reference. Isa doesn't implement against this — but it's useful to know what `fall_dashboard` does with the MQTT messages your mobile app publishes.

### 2.1 Production deployment topology (relevant to URL choice)

FOCUS uses a **single namespace** for everything (their existing services + ours). So in-cluster URLs are just `<service-name>:<port>` — no `.<namespace>.svc.cluster.local` cross-namespace dance. Mobile-app traffic enters through Traefik (their load balancer); FOCUS DevOps configures the route. The endpoints below are reachable through whatever public hostname/path FOCUS exposes.

| Service | In-cluster URL | Public URL |
|---------|-----------------|-------------|
| `inference-server` | `http://inference-server:8001` | `https://<focus-host>/predict` (TBD with FOCUS) |
| `fall-dashboard`   | `http://fall-dashboard:8002`   | `https://<focus-host>/api/...` (TBD with FOCUS) |
| `mqtt-broker`      | `mqtt://mqtt-broker:1883` (TCP) / `ws://mqtt-broker:9001` (WS) | exposed via NodePort or LoadBalancer (TBD) |

For local dev, `localhost:<port>` after `kubectl port-forward` — see [`ISA_local_setup_quickstart.md`](ISA_local_setup_quickstart.md).

### 2.2 `fall_dashboard` REST + SSE API

Our backend service `fall_dashboard` (port 8002) exposes a REST + SSE API. The Patient Dashboard (FOCUS) and the Fall Dashboard (us) both consume this.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/patients` | GET | list of registered patients with fall counts |
| `/api/falls` | GET | full or filtered fall history |
| `/api/stream` | GET (SSE) | live event stream — one event per new alert |

CORS is wide-open (`*`) in dev. Auth is TBD with FOCUS for production.

#### `/api/patients` — list with fall counts

```http
GET /api/patients
```

```json
{
  "patients": [
    {
      "patient_id":      "charite-patient-001",
      "mac_id":          "6c:1d:eb:04:a9:e6",
      "fall_count":      3,
      "session_started": "2026-04-27T08:00:00",
      "session_active":  true
    }
  ]
}
```

#### `/api/falls` — full fall history

```http
GET /api/falls?patient_id=charite-patient-001&only_falls=true&limit=200
```

```json
{
  "falls": [
    {
      "id":                42,
      "observation_id":    "3a0a603e-cc5a-4355-ad8c-f53f2e4de1b9",
      "patient_id":        "charite-patient-001",
      "mac_id":            "6c:1d:eb:04:a9:e6",
      "fall_detected":     true,
      "patient_confirmed": "yes",
      "needs_help":        true,
      "detection_time":    "2026-04-28T10:00:00+00:00"
    }
  ]
}
```

Query parameters:
- `patient_id` (optional) — filter to one patient
- `only_falls` (default `true`) — only rows where `fall_detected=true`
- `limit` (default 200, max 2000)

#### `/api/stream` — live SSE feed

Long-lived Server-Sent Events stream. One event per new alert; clients use the browser/HTTP `EventSource` pattern. Currently, the SSE stream emits **only post-confirmation events** — see section 3 for the planned change.

Current event payload matches the `/api/falls` row schema above. After section 3's work lands, a `state` field will be added.

### 2.3 What goes through SSE today (current 2-state behaviour)

`fall_dashboard` filters server-side. The cases that reach SSE today:

| `patient_confirmed` | `needs_help` | Comes through SSE? | Reason |
|---------------------|--------------|:------------------:|--------|
| `not_answered` | n/a | **yes** | patient could not respond — assume serious |
| `yes` | `true` | **yes** | patient confirmed AND asked for help |
| `yes` | `false` | no | patient confirmed but says they're okay |
| `no` | n/a | no | false positive |

Every event the SSE delivers is a real caregiver-attention event. Other states are stored in `fall_history` (queryable via `/api/falls`) but are not pushed live.

### 2.4 The `observation_id` cross-reference

Every prediction is tagged with a UUID (`observation_id`):
- generated by `inference-server` per `/predict`
- returned in the HTTP response to the mobile app
- included in the MQTT payload your app publishes to `fall/alert/<patient_id>`
- stored in both `inference_log` (inference-server side) and `fall_history` (fall-dashboard side)

This is what links a phone-side detection to the dashboard event. Don't drop it from your MQTT publish — without it, the dashboard side can't correlate. Full mobile-app contract details in [`ISA_mobile_app_contract.md`](ISA_mobile_app_contract.md).

---

## 3. Clinical requirement — three dashboard states (in flight)

**Status:** architectural decision agreed with FOCUS DevOps + clinical partner (2026-04-29). Implementation is on Mohamed (taking over the K8s integration). When this lands, the SSE schema above gains a `state` field.

### 3.1 What the clinical partner asked for

Today the dashboard only highlights a patient *after* the patient confirms (or the 10s timeout passes without "no"). Clinical wants caregivers to be alerted **as soon as a fall is detected**, even before patient confirmation, so they can pay attention to the live biosignals.

### 3.2 The three states

| State | When | Suggested colour | Caregiver action |
|-------|------|------------------|--------------------|
| **Idle** | no recent event | white / default | none |
| **Detected — awaiting** | inference-server returned fall=True; patient hasn't confirmed yet | amber | watch the patient's biosignals; be ready |
| **Confirmed — emergency** | patient confirmed via popup OR 10s timeout (no answer) | red | respond / dispatch |
| **Detected — dismissed** | patient said "no fall" within the popup window | back to white (optionally grey for ~5 min as audit trail) | none |

### 3.3 What changes architecturally (Mohamed's work)

- `inference-server` will call `fall-dashboard` via internal HTTP `POST /internal/detected` immediately when fall=True, sending `{patient_id, observation_id, timestamp}`.
- `fall-dashboard` records this in a new `detection_event` table (Alembic migration pending) and broadcasts an SSE event with `state: "detected"`.
- When the matching `fall/alert/<pid>` arrives later (same `observation_id`), `fall-dashboard` updates the state to `confirmed` or `dismissed` and broadcasts again.

### 3.4 What does NOT change for Isa

- **Mobile-app behaviour is unchanged.** The phone still sends `POST /predict` and publishes `fall/alert/<patient_id>` after the popup. The new "detected" event is generated server-side by inference-server, not by your mobile app.
- **Payload from mobile app stays the same.** Same `observation_id`, same `patient_confirmed`, same `needs_help` fields.
- **API auth stays the same.** Same `X-API-Key` header.

The only thing to watch: **don't drop the `observation_id`** from your MQTT publish. The dashboard relies on it to correlate the phone's confirmation event with the inference-server's earlier "detected" event. If `observation_id` isn't present or doesn't match, the dashboard will end up with a stuck "detected" state that never resolves.

### 3.5 Reference UI

The reference implementation of the 3-state UI will land first in the `mock_patient_dashboard` HTML (the local-test dashboard at `localhost:30090`) so that FOCUS's Flutter team and you can see it working before they replicate it in Flutter. When that's ready, screenshot/video will be added here or in `FOCUS_patient_dashboard_integration.md`.

---

## 4. How Isa can verify mobile-app changes light up the dashboard

You don't need to touch the web side — but you do need to confirm that whatever your mobile app is doing produces the right effect on the dashboard. Easiest path:

1. Bring up the local stack: [`ISA_local_setup_quickstart.md`](ISA_local_setup_quickstart.md). One script (`helm/mock-focus/install.ps1`) installs everything.
2. Open `http://localhost:30090/` (the mock_patient_dashboard) in a browser.
3. Drive a fall through your mobile-app code path — either the real app pointed at the local cluster, or `python -m local_dev.mock_app.main` as a known-good reference.
4. Watch the patient card on the dashboard:
   - **Today's behaviour:** patient flips red ~10 s after the fall (after the popup expires or the patient confirms).
   - **After section 3 lands:** patient flips amber immediately on detection, then transitions to red (confirmed) or back to white (dismissed) after the popup result.

If the dashboard doesn't react when you fire a fall, the most common causes are:

| Symptom | Likely cause |
|---------|--------------|
| Card never updates at all | MQTT publish silently failed — check `127.0.0.1` vs `localhost` (Windows IPv6 gotcha — see [`ISA_local_setup_quickstart.md`](ISA_local_setup_quickstart.md)) |
| Card goes amber but never resolves | `observation_id` mismatch between your `/predict` response and your MQTT payload — the dashboard can't correlate |
| Card flips red but never amber | Detected-event endpoint not yet deployed (section 3 work is pending) — expected for now |
| Multiple events stacking up | MQTT QoS 0 + retry logic in your app — drop duplicates on `observation_id` |

---

## 5. The other 4 dashboards — what they are, who builds them

For your awareness only. None are your responsibility.

### Fall Dashboard (MCS owns)
Backend: `fall_dashboard` at port 8002 (same APIs as section 2). Frontend: a minimal HTML at `http://localhost:8002/` for local test; in production, this gets folded into FOCUS's web app shell as a tab. **Built by us** (or FOCUS, depending on how the unified shell ends up looking). Not your concern.

### ML Dashboard (MCS owns — standalone)
`ml_dashboard` at port 8004. Standalone admin UI for retrain, model promote, hot-swap. **Admins only.** Linked from the menu — no integration work needed from anyone outside us.

### Server Health Dashboard (MCS owns — standalone)
`server_health` at port 8006. Aggregates `/health` endpoints of all services into a plain-language status page. Admins only.

### Grafana Dashboard (third-party — already running)
Lives at `http://<host>:3000/`. Three pre-provisioned dashboards: ml_server_overview, model_performance, fall_events_timeline. Admins only — link to it from the menu, no integration work needed.

---

## 6. Auth / role split

In dev: no auth, all dashboards accessible. In production:

- FOCUS SSO issues a JWT with a `role` claim — either `caregiver` or `admin`.
- The web app shell reads the role and conditionally renders the menu:
  - `caregiver` → menu shows Patient Dashboard + Fall Dashboard
  - `admin` → menu shows ML Dashboard + Server Health + Grafana
- Each backend (our `fall_dashboard`, `ml_dashboard`, etc.) re-validates the role from the JWT before returning data — defence in depth.

The exact JWT claim name and SSO integration are TBD between FOCUS DevOps and us. For your mobile app, see [`ISA_mobile_app_contract.md`](ISA_mobile_app_contract.md) — the auth requirements there are different (it's `X-API-Key` header, not JWT, in the current design).

---

## 7. Cross-references

| Doc | When to read |
|-----|--------------|
| [`ISA_mobile_app_contract.md`](ISA_mobile_app_contract.md) | Authoritative contract for what your mobile app must do |
| [`ISA_local_setup_quickstart.md`](ISA_local_setup_quickstart.md) | Bring up the whole local backend so you can test against it |
| [`FOCUS_patient_dashboard_integration.md`](../handover_docs/FOCUS_patient_dashboard_integration.md) | The Patient Dashboard integration guide written for the FOCUS Flutter team. Useful if you're coordinating with them on the data contract. |
| [`MOHAMED_focus_handover.md`](../handover_docs_2/MOHAMED_focus_handover.md) | Why the chart is changing for FOCUS, including the 3-state work in section 3 of this doc |
| [`04_mobile_app_integration.md`](../handover_docs_2/04_mobile_app_integration.md) | Numbered handover doc — same contract as `ISA_mobile_app_contract.md`, sometimes more current |
