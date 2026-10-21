# User Flow — Patient

**Audience:** anyone implementing or reviewing the patient-side experience — Isa (mobile-app dev), clinical reviewers at Charite, FOCUS DevOps verifying the popup behaviour.
**Scope:** what the patient sees, hears, feels, and does — from "wearing the wearable" to "alert escalates" or "alert clears".

This is the human flow. The technical flow that backs it lives in `04_mobile_app_integration.md`.

---

## 1. The patient in one paragraph

The patient wears the SmarKo wearable on their body and carries a phone running the SmarKo mobile app. Both devices are paired via Bluetooth. The patient does not interact with the wearable directly — it just streams data. Their only active touchpoint is the mobile-app popup that appears when the system thinks they have fallen. Everything else is silent.

---

## 2. What the patient sees / does, end-to-end

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PRE-FALL                                                               │
│  • Wearable is on the patient's body, BLE-connected to the phone.       │
│  • Mobile app runs in the background (or foreground, doesn't matter).   │
│  • No UI is shown to the patient. No notifications. No interruptions.   │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
        [The patient may or may not actually fall — it doesn't matter.
         What matters is the model decided fall_detected=true on a window.]
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  TRIGGER (≈ 0–5 seconds after the candidate fall)                       │
│                                                                         │
│  • Phone wakes the screen. Plays an audible tone.                       │
│  • Optional vibration pattern (per phone OS guidelines).                │
│  • Modal popup appears, full-screen, dismiss-blocking:                  │
│                                                                         │
│        ┌──────────────────────────────────────┐                         │
│        │   Did you just fall?                 │                         │
│        │                                      │                         │
│        │      [ Yes ]   [ No ]                │                         │
│        │                                      │                         │
│        │         ⏳ 10 seconds remaining       │                         │
│        └──────────────────────────────────────┘                         │
│                                                                         │
│  • Countdown ticks down visibly (and audibly, e.g. faster pulse near    │
│    timeout — clinical preference).                                      │
│  • Buttons must be reachable from a hand-on-floor posture (large hit    │
│    targets, high contrast, reachable thumb).                            │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
        Patient taps         Patient taps        No tap within 10s
            "Yes"               "No"                  ⏳
              │                   │                   │
              ▼                   ▼                   ▼
┌────────────────────┐ ┌────────────────────┐ ┌──────────────────────────────┐
│  YES I FELL        │ │  NO I DIDN'T       │ │  TIMEOUT                     │
│                    │ │  (false positive)  │ │                              │
│  Show Q2:          │ │                    │ │  Treated as "not_answered".  │
│                    │ │  Popup closes      │ │                              │
│  ┌──────────────┐  │ │  silently.         │ │  Caregiver IS notified       │
│  │ Need help?   │  │ │                    │ │  (conservative — patient     │
│  │              │  │ │  Caregiver NOT     │ │   may be hurt and unable to  │
│  │ [Yes] [No]   │  │ │  notified.         │ │   answer).                   │
│  │  ⏳ 10 sec   │  │ │                    │ │                              │
│  └──────────────┘  │ │  Stored in DB for  │ │  Stored in DB.               │
│                    │ │  retraining.       │ │                              │
└────────────────────┘ └────────────────────┘ └──────────────────────────────┘
       │
       ├── tap "Yes" ─►  YES I NEED HELP    → Caregiver notified (RED).  Stored in DB.
       │                                      Patient may also see a "Help is on the
       │                                      way" confirmation screen (recommended UX).
       │
       └── tap "No" ──►  YES I'M OKAY       → Caregiver NOT notified.    Stored in DB.
                                              Patient may see a "Glad you're okay,
                                              please rest a moment" screen.
```

---

## 3. The four end-states — what the patient and caregiver each experience

| Patient's choice | Patient's screen after | Caregiver experience |
|------------------|------------------------|----------------------|
| Yes I fell, Yes I need help | "Help is on the way" message | **Red flag** appears on the Patient Dashboard within 1–2s, audible alert (whatever the dashboard does for red) |
| Yes I fell, No I'm okay | "Glad you're okay" message | Nothing visible — flag is NOT raised |
| No I didn't fall (false positive) | Popup closes silently | Nothing visible |
| Timeout (no tap in 10s) | Popup closes; the phone may continue to attempt to reach the patient with subsequent alerts (depends on mobile-app policy) | **Amber flag** appears on the Patient Dashboard within 1–2s |

> **Hidden invariant:** all four end-states are stored in the backend `fall_history` table for retraining. The "caregiver alert" filter is independent of the storage — even false positives have value as labelled examples for the model.

---

## 4. Edge cases the patient flow must handle

### 4.1 Phone is locked or asleep

The popup must wake the phone. This usually means a high-priority push-style notification or full-screen intent (Android: `USE_FULL_SCREEN_INTENT`). The patient should not have to unlock the phone to answer — buttons must be reachable from the lock screen.

### 4.2 Patient is unconscious / can't respond

This is the **timeout path** (10s). It is intentionally conservative — the timeout is treated as `not_answered`, which **does** notify the caregiver. The system errs on "wake the caregiver and let them check" rather than "stay silent because the patient was vague".

### 4.3 Patient is mid-action (e.g. sitting down quickly)

This is the **false-positive path**. The patient taps "No" — popup goes away immediately, no caregiver alert, no further escalation. The system records the decision and uses it to retrain.

### 4.4 Multiple consecutive falls

Each /predict call that fires generates its own popup and its own observation_id. If the patient is in a chaotic situation (multiple stumbles within minutes), they may see multiple popups. This is intentional — each is a separate datum. The mobile app should NOT debounce, throttle, or merge popups; it lets the backend dedupe if needed (currently it doesn't — every alert is a row).

### 4.5 No internet

If `/predict` fails, the popup never appears (we couldn't classify the window). The mobile app retries the POST per the error policy in `04_mobile_app_integration.md`. If the popup completes but the MQTT publish fails, the mobile app should buffer the payload and retry. If MQTT eventually fails, the data is lost — there's no other path. This is a known limitation; offline buffering is an extension.

### 4.6 The patient uses the phone normally

Should not interfere. The mobile app runs in the background. The popup is high-priority — it preempts whatever the patient is doing. Once dismissed (any of the 4 paths), the patient resumes normal use.

---

## 5. UI / UX guidelines for the popup

These are recommendations from the clinical reviewers at Charite + general accessibility principles.

| Aspect | Requirement |
|--------|-------------|
| Modal | Full-screen, blocks dismissal by tapping outside |
| Language | Adult, plain, calm. "Did you just fall?" not "Has a fall event been detected?" |
| Buttons | Two large rectangles. Yes on the left or right consistently across the app — match locale conventions |
| Hit target | Minimum 64dp/pt per side, with spacing — patient may have shaky hands |
| Contrast | High — usable in low light or by patients with reduced vision |
| Sound | Audible tone on appearance. Optional escalation tone as countdown hits 5s, 3s |
| Vibration | Pattern that wakes a patient who is dozing |
| Countdown | Visible numeric + visible bar/circle. Both — redundancy |
| Localisation | German + English at minimum (Charite is in Berlin). Other languages per site |
| Accessibility | VoiceOver / TalkBack must announce "Did you just fall? Yes button. No button. 10 seconds remaining." |
| Theme | Forced light theme during alert (or follow OS — but ensure contrast is met either way). High-stakes UI is not the place to honor "dark mode" muscle memory if it harms readability |
| Animation | Minimal — no celebratory transitions. This is a serious moment. |

---

## 6. The patient does NOT do these things

| Action | Why not |
|--------|---------|
| Manually trigger a fall report | The whole point is the system detects it. There is no user-initiated "I fell" button in the current design. (This could be added — extension; not in scope.) |
| Configure thresholds, models, sample rates | Admin scope only |
| See or interact with the caregiver dashboard | Distinct UI, distinct role |
| Cancel the popup mid-countdown without answering | The popup must be answered or timed out |
| Talk to the inference server, MQTT broker, or any backend | Mobile app handles all of that |

---

## 7. Step-by-step from the patient's perspective — a single day

Concrete walk-through to help reviewers reason about the flow:

```
07:00  Patient puts on the wearable. App is already running. No UI shown.
07:00–11:30  Patient goes about their morning. Walks, sits, lies down. App is silent.
              [Background: each 9s window POSTed to /predict.
              Most return fall_detected=false. No popup.]
11:32  Patient stumbles in the bathroom, doesn't actually fall — recovers balance.
        [Window posted, model returns fall_detected=true with confidence=0.62.]
        Phone vibrates + tones. Popup appears: "Did you just fall?"
        Patient taps "No" within 2s.
        Popup closes. No caregiver alert.
        [Backend: fall_history row stored with patient_confirmed='no'.]
11:35  Patient continues normally. No further interruption.
14:50  Patient genuinely trips over a rug, falls to the floor.
        [Window posted, model returns fall_detected=true with confidence=0.87.]
        Phone vibrates + tones. Popup appears: "Did you just fall?"
        Patient taps "Yes" within 4s.
        Popup updates: "Do you need help?"
        Patient taps "Yes" — they want help.
        Popup updates to "Help is on the way. Please stay where you are."
        [Backend: fall_history row stored with patient_confirmed='yes', needs_help=true.
        SSE event fires. Caregiver dashboard shows red flag for this patient.]
14:51  Caregiver responds (calls patient, dispatches if necessary).
14:53  Caregiver arrives or speaks with patient over intercom. Manual care begins.
        [System has done its job. Subsequent care is human/clinical.]
17:30  (Hypothetical) Phone vibrates + tones. Popup. Patient is asleep.
        Patient does not respond within 10s.
        [Timeout. Backend: fall_history row stored with patient_confirmed='not_answered'.
        SSE event fires. Caregiver dashboard shows amber flag.]
17:31  Caregiver investigates (calls patient, checks via room camera if available).
        Patient turns out to be merely asleep. Caregiver dismisses the flag manually
        in the dashboard (extension feature — currently they just acknowledge mentally).
```

---

## 8. The patient never sees

- The model version
- The confidence score
- The observation_id (UUID)
- The popup's underlying state machine
- Any technical error message — if /predict fails or MQTT fails, the popup either doesn't appear or quietly finishes; the patient is never shown a stack trace or "service unavailable" toast

---

## 9. What happens if the patient turns the wearable off

`/predict` calls stop firing because the mobile app gets no BLE data. No popups appear. The system goes silent. There's no proactive "wearable disconnected" alert in the current design — it's an extension (`server_health` admin dashboard tracks pod health but not wearable BLE liveness).

If we add an alert for this (e.g. "wearable hasn't sent data in 15 minutes"), it would be a caregiver-side notification, not a patient-side popup.

---

## 10. Cross-references

- [`04_mobile_app_integration.md`](04_mobile_app_integration.md) — technical contract behind the popup (HTTP /predict, MQTT publish, observation_id flow)
- [`07_user_flow_caregiver.md`](07_user_flow_caregiver.md) — what the caregiver sees on the other side of this flow
- [`03_fall_detection_system.md`](03_fall_detection_system.md) — system architecture context
- `local_dev/mock_app/patient_server.py` — reference implementation of the popup as a browser page (HTML inline at the bottom)

---

## 11. Open clinical decisions (for Charite review)

These are UX decisions we'd like clinical input on before go-live:

1. **Audible tone style** — should it be calm (inviting an answer) or alarming (assuming serious)? Charite preference?
2. **Vibration pattern** — long single buzz vs three short buzzes vs continuous. Local norms.
3. **Yes button placement** — left or right? German UX convention vs English.
4. **Post-confirmation screen** — should we explicitly say "Help is on the way" if the caregiver hasn't yet acknowledged? Risk: false reassurance if the caregiver isn't reachable. Alternative: "Your caregiver has been notified."
5. **What if the patient repeatedly false-positives** — at some point the model is over-firing for this person. Should we surface "this patient has X false positives this week" to the admin so they can retrain or recalibrate? (Currently visible in `ml_dashboard` if you query Postgres manually.)

---

## 12. Contact

| For | Reach out to |
|-----|--------------|
| Mobile-app implementation | Isa (SmarKo) |
| Clinical UX, popup wording, accessibility | Charite |
| /predict and MQTT contract | Hayate (MCS) |
