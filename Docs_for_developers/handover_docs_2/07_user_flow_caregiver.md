# User Flow — Caregiver

**Audience:** anyone implementing or reviewing the caregiver-side experience — FOCUS DevOps building the Patient Dashboard, Charite clinical staff who will use it, MCS reviewers.
**Scope:** what the caregiver sees, what they do, and what their workflow looks like — from "monitoring the dashboard" to "responding to an alert".

This is the human flow. The technical flow that backs it lives in `05_web_app_integration.md`.

---

## 1. The caregiver in one paragraph

The caregiver is a clinical or care-staff role at Charite (or a similar deployment site). Each shift, one or more caregivers monitor the FOCUS Patient Dashboard for the cohort of patients they're responsible for. They watch for fall flags, respond to actionable ones, and triage. They do **not** interact with the patient's wearable, the mobile app, or the inference backend — only the dashboard. Authentication is via FOCUS SSO (planned); roles are mutually exclusive (caregiver vs admin).

---

## 2. What the caregiver sees

The caregiver opens the FOCUS Patient Dashboard in a browser. The UI is owned by FOCUS DevOps; we (MCS) supply the fall data via REST + SSE. The dashboard already shows demographics (FHIR) and biosignals (InfluxDB) — fall flags are added to the patient cards.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Patient Dashboard — Charite ICU East — caregiver view               │
│  ┌────────────────────────┐  ┌────────────────────────┐              │
│  │ Patient 001 [🔴]       │  │ Patient 002            │              │
│  │ HR: 88 bpm             │  │ HR: 72 bpm             │              │
│  │ Age 78, Room 14        │  │ Age 65, Room 15        │              │
│  │                        │  │                        │              │
│  │ ⚠ FALL — needs help    │  │   (no flag)            │              │
│  │   2 minutes ago        │  │                        │              │
│  │                        │  │                        │              │
│  │ [View detail]          │  │ [View detail]          │              │
│  └────────────────────────┘  └────────────────────────┘              │
│  ┌────────────────────────┐  ┌────────────────────────┐              │
│  │ Patient 003 [🟡]       │  │ Patient 004            │              │
│  │ HR: 95 bpm             │  │ HR: 80 bpm             │              │
│  │                        │  │                        │              │
│  │ ⚠ FALL — unresponsive  │  │   (no flag)            │              │
│  │   30 seconds ago       │  │                        │              │
│  └────────────────────────┘  └────────────────────────┘              │
└──────────────────────────────────────────────────────────────────────┘
```

### Flag colour rules

| Colour | What happened | Urgency |
|--------|---------------|---------|
| 🔴 **Red** | Patient confirmed they fell AND said they need help | Highest — call / dispatch immediately |
| 🟡 **Amber** | Popup timed out — patient didn't answer in 10s | High — patient may be unable to respond. Investigate. |
| (no flag) | Either no recent fall, or patient confirmed they're okay, or false positive | Routine monitoring |

These two are the only flag states. The two non-actionable states (`patient_confirmed='yes'/needs_help=false` and `patient_confirmed='no'`) do **not** appear as flags — they're stored for retraining but should not interrupt the caregiver's attention.

---

## 3. End-to-end caregiver flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  SHIFT START                                                            │
│  • Caregiver logs into FOCUS Patient Dashboard via SSO.                 │
│  • Dashboard shows the cohort of patients they're assigned.             │
│  • For each patient, dashboard checks the last 24h of fall_history.     │
│    Flags any patient with an unresolved actionable fall.                │
│  • Dashboard opens an SSE connection to /api/stream for live updates.   │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ROUTINE MONITORING                                                     │
│  • Dashboard sits open. Most cards show no flag.                        │
│  • Caregiver does normal duties. Glances at dashboard occasionally.     │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                       [a fall fires somewhere]
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ALERT (≈ 1–2 seconds after MQTT publish)                               │
│                                                                         │
│  • SSE event arrives in the dashboard.                                  │
│  • Patient card flips to red or amber depending on the state.           │
│  • Optional: dashboard plays an audible chime / shows a toast.          │
│  • Patient name + room number remains prominent.                        │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                                       ▼
       🔴 Red flag                              🟡 Amber flag
       "needs help"                             "unresponsive"
              │                                       │
              ▼                                       ▼
┌────────────────────────────────┐    ┌────────────────────────────────┐
│  CAREGIVER ACTION — RED        │    │  CAREGIVER ACTION — AMBER      │
│                                │    │                                │
│  1. Glance at HR + biosignals  │    │  1. Glance at HR + biosignals  │
│  2. Call patient room phone    │    │  2. Call patient room phone    │
│  3. If no answer within 30s:   │    │  3. If unanswered after a few  │
│     dispatch nearest staff     │    │     seconds: same as red —     │
│  4. Continue triage as per     │    │     dispatch staff             │
│     unit protocol              │    │                                │
└────────────────────────────────┘    └────────────────────────────────┘
              │                                       │
              └───────────────────┬───────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  RESOLUTION                                                             │
│  • Caregiver verifies patient state in person or by call.               │
│  • Documents the event in the standard care record (out of scope here). │
│  • Optionally acknowledges / clears the flag in the dashboard.          │
│    (Acknowledge feature is a planned extension — currently the flag     │
│     auto-expires after 24h or when a newer state arrives.)              │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                            [back to monitoring]
```

---

## 4. The four end-states — caregiver perspective

| Patient action | Caregiver sees | What the caregiver does |
|----------------|----------------|-------------------------|
| Yes I fell, Yes I need help | 🔴 Red flag, "needs help" label | **Respond immediately** — call, dispatch |
| Timeout (patient unresponsive) | 🟡 Amber flag, "unresponsive" label | **Investigate immediately** — patient may be unable to answer |
| Yes I fell, No I'm okay | Nothing visible | (Reviewed only on the History tab during shift handover, if at all) |
| No I didn't fall (false positive) | Nothing visible | Same — historical only |

The dashboard intentionally does NOT show false positives or "okay" confirmations as flags. The reasoning: caregiver attention is finite and these events do not need a response. They're stored for retraining but visually dropped.

---

## 5. Where the data comes from (for the caregiver)

The caregiver doesn't need to know the plumbing, but their experience is driven by:

1. **Dashboard load** → `GET /api/falls?limit=2000` to our `fall-dashboard` service.
   - Filters in-app: keep only events from the last 24h that are actionable (red or amber).
   - Each surviving event sets a flag on the corresponding patient card.

2. **Live updates** → `GET /api/stream` (SSE) from our `fall-dashboard`.
   - Server-side filter: only actionable events are sent.
   - Each event flips a patient card to red or amber.

3. **Detail view** (if implemented) → `GET /api/falls?patient_id=<pid>` for a per-patient timeline.

---

## 6. Operational expectations

### 6.1 Concurrent caregivers

Multiple caregivers may have the dashboard open simultaneously. They all receive the same SSE events. Each browser independently flags the same card. No coordination is needed — the underlying event is shared and the visualization is per-browser.

If we add an "acknowledge" feature, we'd need to broadcast acknowledgements too (otherwise one caregiver clears the flag and the others still see it). That's a server-side state change — a planned extension, not yet built.

### 6.2 Alert noise budget

Empirically (from internal testing): expect ~5–15% of `/predict` calls to return `fall_detected=true`, of which ~20–40% are false positives at the 0.5 threshold. So in a real deployment with N patients each generating one /predict per 9 seconds, the actionable-alert rate per patient per day is in the order of single digits at most. The dashboard should not become a torrent.

If you observe more than 1 alert/patient/hour, escalate to the admin to retrain or tune the threshold — see `08_user_flow_admin.md`.

### 6.3 Audible alerts

The dashboard should play a sound when a red flag arrives — caregivers won't always be looking at the screen. Amber flags can be audibly distinct (different tone) or silent depending on local policy. Mute should be opt-in per-shift, not default-on.

### 6.4 Shift handover

At shift end, a caregiver should be able to see the day's history (red + amber, plus optionally yes/okay confirmations) in a separate "History" tab. We have the data — `GET /api/falls?only_falls=true` returns everything. The dashboard just needs to render it.

### 6.5 Time zone

`detection_time` is ISO 8601 with timezone (UTC by default). The dashboard should display in the patient's local time zone, with relative formatting near recent events ("2 minutes ago") and absolute formatting for older events ("today at 14:32").

---

## 7. Edge cases the caregiver flow must handle

### 7.1 Dashboard reload / browser refresh

The caregiver refreshes the page (intentionally or via a tab restore). The flag state must persist. Mechanism: REST `/api/falls` on load reconstructs the current flags from the last 24h. The SSE stream then catches new events. If the dashboard relied only on SSE, a refresh would lose all flags.

### 7.2 Network blip / SSE disconnect

The SSE client reconnects (the `flutter_client_sse` package does this automatically). Events that fired while disconnected are **lost** — SSE does not replay history. To compensate, on every SSE reconnect the dashboard could optionally re-fetch `/api/falls` for the last few minutes. Recommended but not strictly required for v1.

### 7.3 The same patient falls twice

Within minutes:
- Each fall is a separate observation_id and a separate row in `fall_history`.
- Each fires its own SSE event.
- The dashboard's per-patient flag map is keyed by `patient_id`, so the second event overwrites the first — only the most recent flag is shown. The "View detail" view shows the full sequence.

This is the right behaviour: showing two flags for the same patient adds noise without action. If the second is escalated (e.g. "now needs help" after "first was unresponsive"), the colour may change — the rendering rules just take the current state.

### 7.4 An admin / non-caregiver tries to access the dashboard

Currently no auth — anyone who can reach the URL sees fall data. In production (planned, todo.md Step 11.5.4), FOCUS SSO issues a JWT with a `role` claim. The fall-dashboard validates and 401s if the role is `admin` (admins use a separate dashboard). Until SSO is wired, gate access at the ingress (basic auth, IP allow-list, or mTLS).

### 7.5 Patient leaves the cohort or is discharged

The dashboard's patient list comes from the FHIR Patient resource (FOCUS-side). When a patient is no longer in the list, their card disappears regardless of historical falls. The data remains in our `fall_history` (we don't delete on patient discharge — that's a separate retention policy, currently no automatic deletion).

### 7.6 The fall-dashboard service is down

The dashboard's REST call fails on load (caregiver sees an error toast or skeleton). The SSE connection refuses or drops. The caregiver loses fall visibility but still sees demographics and biosignals (those go through different services). The admin should be alerted via `server_health` (port 8006) — see `08_user_flow_admin.md`.

---

## 8. What the caregiver does NOT do

| Action | Why not |
|--------|---------|
| Trigger a fall manually | The system is the trigger; manual entries belong in the standard care record |
| Configure the model, threshold, sample rate | Admin scope only |
| See per-window confidence histograms or model-performance graphs | Admin scope (Grafana, ml_dashboard) |
| Access the inference server, MLflow, or backend Postgres directly | Architecture forbids it; UI is read-only via fall-dashboard |
| Edit historical fall events | Read-only; corrections go through admin |
| Acknowledge / clear flags | (Planned extension) — currently flags auto-expire on time or new state |

---

## 9. UI / UX guidelines for the caregiver dashboard (recommended)

These are recommendations to FOCUS DevOps building the Flutter web app:

| Aspect | Requirement |
|--------|-------------|
| Layout | Patient cards in a grid. Active flags float to the top or are visually emphasized. |
| Flag size | Large, unambiguous icon — visible from across the room |
| Flag text | Short label: "FALL — needs help" / "FALL — unresponsive". Avoid jargon. |
| Time | Always show "X minutes/seconds ago" + absolute timestamp on hover/tap |
| Sound | Distinct audible tone for red (urgent) and amber (less urgent). Mute is opt-in, not default. |
| Dismissal | Flags do not have a manual "dismiss" button in v1 (extension). They expire on time or new state. |
| History tab | Last 7 days of events for the cohort, sortable, filterable by `patient_confirmed` |
| Per-patient detail | Timeline of all events with confidence + state. Click `observation_id` shows the JSON payload (helps support tickets). |
| Accessibility | High contrast, screen-reader labels for flag colour and state, keyboard navigation |
| Localisation | German + English minimum |

---

## 10. Step-by-step from the caregiver's perspective — a single shift

```
07:00  Caregiver logs into FOCUS Patient Dashboard (SSO).
        Dashboard loads. 12 patient cards. Cards 003 and 007 have amber flags
        from overnight (timeouts — patient was asleep both times).
        Caregiver reviews each: confirms via room camera both are sleeping
        normally. Mentally acknowledges (no UI action in v1).
07:30  Routine rounds. Dashboard stays open in a browser tab.
10:14  Audible chime. Card 005 flips to red — "FALL, needs help, 2s ago".
        Caregiver opens room phone, dials Patient 005's room.
        Patient answers — "yes I tripped, my hip hurts."
        Caregiver dispatches a colleague to the room. Stays on call until
        colleague arrives. Documents the event in the care record.
10:25  Colleague reports patient is okay (bruise but no fracture). Caregiver
        notes the event for the shift handover sheet.
        [Card 005 still shows red — flag will auto-clear after 24h or after
        the next non-fall event for this patient. In a future iteration, the
        caregiver can manually click "acknowledge".]
13:42  Card 008 flips to amber. Patient timeout. Caregiver calls room — patient
        answers, was just napping. Confirms patient is fine. No further action.
14:50  Routine rounds; otherwise quiet.
17:30  Card 005 still shows the morning's red flag. (Stale — informational only.)
        Hours have passed; caregiver mentally treats it as historical.
19:00  Shift end. Caregiver opens History tab, exports the day's events for
        the handover sheet (or does this through whatever the unit's process is).
        Logs out.
```

---

## 11. Open clinical decisions (for Charite review)

These are workflow decisions we'd like clinical input on before go-live:

1. **Acknowledge / clear flag UI** — should v1 have it, or wait? Adding it is server-side state change, modest effort. Without it, a stale red flag from 11h ago can be confusing.
2. **Audible tone style** — match what the unit currently uses for other monitor alerts? Or differentiate intentionally?
3. **Cohort scope per caregiver** — does each caregiver see ALL patients or only their assigned ones? FOCUS Patient Dashboard already handles patient assignment; we just inherit it.
4. **Cross-shift escalation** — if a red flag is open at shift end, should the outgoing caregiver explicitly hand it off? Process question, not technical.
5. **False-positive review process** — should false-positive falls (`patient_confirmed='no'`) ever be surfaced for any reason? Currently invisible to the caregiver. We do surface them on the History tab if the dashboard implements it.
6. **Multi-cohort handover** — when a caregiver covers multiple units' dashboards, how does the audible alert disambiguate? UX question for FOCUS DevOps.

---

## 12. Cross-references

- [`05_web_app_integration.md`](05_web_app_integration.md) — technical contract for the dashboard (REST + SSE)
- [`06_user_flow_patient.md`](06_user_flow_patient.md) — what the patient does on the other side
- [`08_user_flow_admin.md`](08_user_flow_admin.md) — how the admin tunes the system to keep caregiver noise low
- [`03_fall_detection_system.md`](03_fall_detection_system.md) — system architecture context

---

## 13. Contact

| For | Reach out to |
|-----|--------------|
| Patient Dashboard UI implementation | FOCUS DevOps |
| Clinical workflow, alert escalation policy, audible alert preferences | Charite clinical leads |
| Fall data API, REST/SSE shape, event semantics | Hayate (MCS) |
| Mobile-app behaviour upstream of these alerts | Isa (SmarKo) |
