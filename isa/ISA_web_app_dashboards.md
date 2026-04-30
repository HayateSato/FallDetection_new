# Web App — Dashboards Update Guide for Isa

**Purpose:** what Isa needs to build/update in the FOCUS caregiver web app to integrate the fall-detection backend.
**Scope:** the Patient Dashboard (existing FOCUS UI Isa updates) and the four sibling dashboards that will live alongside it.
**Reference:** [`overview.txt`](overview.txt) defines the layered architecture (Patient / Backend / Caregiver) — this doc fills in the data-side details Isa needs.

---

## 1. The web app contains five dashboards

Per `overview.txt`, the caregiver web app is a single web application with five dashboards inside it. Two are FOCUS-owned (you build/maintain), three are MCS-owned (we build).

| # | Dashboard | Owner | Purpose | Backend |
|---|-----------|:-----:|---------|---------|
| 1 | **Patient Dashboard** | FOCUS (Isa) | Per-patient personal info (gender, BMI, age) + physiological live readings (HR, SpO2, ACC) + **fall status indicator** | FHIR + InfluxDB + our `fall_dashboard` |
| 2 | **Fall Dashboard** | MCS (us) | All fall events, filterable by period / patient / help-requested | our `fall_dashboard` (port 8002) |
| 3 | **ML Dashboard** | MCS (us) | Admin: retrain models, hot-swap to production | our `ml_dashboard` (port 8004) |
| 4 | **Server Health Dashboard** | MCS (us) | Admin: aggregate service liveness | (not yet built) |
| 5 | **Grafana Dashboard** | MCS (us) | Admin: ML model performance metrics + drift charts | Grafana (port 3000) |

**Role-based access** (see `REFACTOR_DOCUS/deployment_architecture.md` for the full discussion):
- **Caregivers** see dashboards 1 + 2 only.
- **Admins** see dashboards 3 + 4 + 5 only.
- The dashboards share the same web app shell (URL prefix, auth, navigation), but the role determines which menu items are visible.

The role split needs coordination with FOCUS DevOps for SSO / JWT claim names. For now, treat it as "everyone sees everything" in dev and gate it later.

---

## 2. Patient Dashboard — what Isa is responsible for

The Patient Dashboard is mostly FOCUS-owned and you already have its core working. The new piece is **fall status integration** — when our system confirms a fall on a patient, that patient's row/card in your dashboard should change colour.

### What you already do (no change)

- Fetch demographics from the FHIR server (`GET /fhir/Patient/{id}`)
- Fetch biosignals from the FOCUS InfluxDB (HR, SpO2, ACC time series)
- Render the per-patient view

### What's new — fall status indicator

When a fall is confirmed for a patient, the Patient Dashboard needs to:

1. Show a **visual flag** on that patient (red highlight, "FALL" badge, etc.)
2. Distinguish "needs help" from "patient confirmed but okay" — the urgency is different
3. Update in **real time** when a new fall comes in (no need to refresh the page)

There are two ways to get this data: a **list query** (initial load) and a **live stream** (updates).

---

## 3. Where to fetch fall data — our `fall_dashboard` API

Our backend service `fall_dashboard` (port 8002 locally; cluster URL TBD) exposes
a REST + SSE API. Use this — do not query Postgres directly.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/patients` | GET | list of registered patients with fall counts |
| `/api/falls` | GET | full or filtered fall history |
| `/api/stream` | GET (SSE) | live event stream — one event per new confirmed fall |

No auth required at the API layer in dev. CORS is wide-open (`*`) so cross-origin
calls from the FOCUS-namespace web app work without setup.

### `/api/patients` — list with fall counts

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

Use this on initial dashboard load to know how many falls each patient has had.
The `fall_count` is the lifetime count, not "today" — for "today only", filter `/api/falls` by date client-side.

### `/api/falls` — full fall history

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

For the Patient Dashboard's fall status indicator, you only need to know "does this patient have a recent unresolved fall?" — see section 4 for the recommended query pattern.

### `/api/stream` — live SSE feed

```http
GET /api/stream
```

Long-lived Server-Sent Events stream. Connect via the browser's built-in `EventSource`:

```javascript
const es = new EventSource("http://<fall-dashboard-host>:8002/api/stream");

es.addEventListener("connected", () => {
  console.log("connected to fall stream");
});

es.onmessage = (e) => {
  const fall = JSON.parse(e.data);
  // fall has the same fields as /api/falls rows
  // Update the patient's status in the Patient Dashboard
};

es.onerror = () => {
  // EventSource auto-reconnects; no manual retry needed
};
```

**Important:** SSE only fans out the alerts that need caregiver action (see filter table in section 4). False positives and "patient confirmed but okay" events are stored in DB but do NOT come through SSE. So the SSE stream is the right signal source for the Patient Dashboard's red flag — every event you receive needs caregiver attention.

---

## 4. Recommended pattern — wiring it into the Patient Dashboard

```javascript
// 1. On page load, fetch the current state of all patients
async function loadPatientStatuses() {
  const patients = await fetch("/api/patients").then(r => r.json());
  // get the most recent fall per patient (last 24h, say)
  const falls = await fetch("/api/falls?limit=2000").then(r => r.json());

  for (const p of patients.patients) {
    const recentFall = falls.falls.find(f =>
      f.patient_id === p.patient_id &&
      isRecent(f.detection_time, 24 * 3600)   // last 24h
    );
    if (recentFall) {
      flagPatient(p.patient_id, recentFall);
    }
  }
}

// 2. Open SSE for live updates
const es = new EventSource("http://.../api/stream");
es.onmessage = (e) => {
  const fall = JSON.parse(e.data);
  flagPatient(fall.patient_id, fall);
};

// 3. The flagging logic — colour matches urgency
function flagPatient(patient_id, fall) {
  const card = document.querySelector(`[data-patient-id="${patient_id}"]`);
  if (!card) return;

  if (fall.patient_confirmed === "yes" && fall.needs_help === true) {
    card.classList.add("status-fall-help");      // red — patient needs help
  } else if (fall.patient_confirmed === "not_answered") {
    card.classList.add("status-fall-unknown");   // amber — patient unresponsive
  } else {
    // confirmed but okay, or false positive — historical only
    // do not flag
  }
}
```

The colour coding above is a suggestion. The two states that need
visualisation are exactly the two that come through SSE — the other two
(`yes`+`no_help` and `no`) only matter for the Fall Dashboard's history view,
not for the Patient Dashboard's "live patient status".

### Why this filter?

`fall_dashboard` already does the filter on the server side. The two cases
that come through SSE are:

| `patient_confirmed` | `needs_help` | Comes through SSE? | Reason |
|---------------------|--------------|:------------------:|--------|
| `not_answered` | n/a | **yes** | patient could not respond — assume serious |
| `yes` | `true` | **yes** | patient confirmed AND asked for help |
| `yes` | `false` | no | patient confirmed but says they're okay |
| `no` | n/a | no | false positive |

So you don't need to filter again on the client — every event the SSE delivers is a real caregiver-attention event. Clearing the flag is your call (e.g. once a caregiver acknowledges, or after N hours).

---

## 5. What other Postgres data could the Patient Dashboard use?

If you ever want richer info, the data is there. Use our REST API rather than
direct Postgres connections — but here's what's available so you know the limits.

| Table | Has | Reachable via |
|-------|-----|---------------|
| `fall_history` | every confirmed alert (the source of `/api/falls`) | `/api/falls` (already used) |
| `inference_log` | every prediction (including no-fall) — patient_id, model_version, confidence, timestamp | not currently exposed in REST API. Can be added if needed — ask Hayate. |
| `feature_snapshot` | per-feature values for each prediction | internal — only retraining uses this. Not exposed externally. |
| `participant_session` | which patients are registered, session start time | `/api/patients` (already used) |

The data the Patient Dashboard *should* fetch is in `fall_history` already and accessible via `/api/falls` + `/api/stream`. If you find yourself needing per-prediction details (e.g. "show me the last 10 model decisions for this patient even when no fall was detected"), tell Hayate — that needs a new endpoint.

For demographics (height, weight, BMI, age), that data lives in the FOCUS FHIR server, not in our Postgres. **Continue fetching from FHIR for those fields** — we don't store demographics. For biosignals (HR, SpO2), continue fetching from the FOCUS InfluxDB. Our system only knows about *predictions* and *alerts*, not the raw physiological state.

---

## 6. The other 4 dashboards in the web app — context only

You don't need to build these (we will), but the web app shell needs to provide
navigation to them so the role-aware menu can include or exclude them.

### Fall Dashboard (MCS — we build the backend, Isa builds the frontend if FOCUS wants it inside the unified web app)

Backend: `fall_dashboard` at port 8002. Same APIs as section 3.

Required UI:
- Filterable list of all falls (by period, patient_id, only-falls toggle, help-requested filter)
- Same colour coding for needs_help vs not_answered vs okay
- Optionally subscribe to SSE for live new entries

Note: there's already a minimal HTML dashboard at `http://localhost:8002/` which is the local-test version. In production it gets replaced by Isa's integration into the unified web app.

### ML Dashboard (MCS — runs as standalone)

`ml_dashboard` at port 8004. Standalone admin UI for retrain, promote, hot-swap. **Admins only.** From the unified web app, just link to it (or iframe it) — no need to reimplement.

### Server Health Dashboard (MCS — not yet built)

Will aggregate `/health` endpoints of all services into a plain-language status page. Admins only.

### Grafana Dashboard (third-party — already running)

Lives at `http://<host>:3000/`. Three pre-provisioned dashboards: ml_server_overview, model_performance, fall_events_timeline. Admins only — link to it from the menu, no integration work needed.

---

## 7. Auth / role split

In dev: no auth, all dashboards accessible. In production:

- FOCUS SSO issues a JWT with a `role` claim — either `caregiver` or `admin`.
- The web app shell reads the role and conditionally renders the menu:
  - `caregiver` → menu shows Patient Dashboard + Fall Dashboard
  - `admin` → menu shows ML Dashboard + Server Health + Grafana
- Each backend (our `fall_dashboard`, `ml_dashboard`, etc.) re-validates the role from the JWT before returning data — defence in depth.

The exact JWT claim name and SSO integration are TBD between Isa, Hayate, and FOCUS DevOps. Treat it as a placeholder for now and just hide menu items based on a hardcoded role flag during dev.

---

## 8. End-to-end checklist for Isa

When the integration is complete, the following should work:

- [ ] On the Patient Dashboard, each patient shows their normal demographics + biosignals
- [ ] When a fall is confirmed (via the mobile app's MQTT publish), the corresponding patient flashes red within 1–2 seconds
- [ ] If the patient was unresponsive (`not_answered`), they flash amber instead of red
- [ ] If the patient confirmed the fall but said they were okay, the patient stays normal (no flag)
- [ ] If the model fired a false positive (`patient_confirmed=no`), the patient stays normal (no flag)
- [ ] Reloading the Patient Dashboard preserves the recent fall flag (server-fetched on load, not just SSE-delivered)
- [ ] Caregivers see only Patient + Fall dashboards in the menu
- [ ] Admins see only ML + Server Health + Grafana in the menu

---
