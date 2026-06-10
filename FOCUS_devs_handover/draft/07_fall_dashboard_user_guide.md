# Fall Dashboard — User Guide

## What It Is

The fall dashboard is a web UI served at your FOCUS subdomain
(e.g. `https://fall.focus-hospital.de`). It gives the caregiver a real-time view
of fall events across all registered patients.

It has two things: a **live alert panel** (notifies the caregiver as events happen)
and a **fall history view** per patient (sourced from InfluxDB).

---

## Accessing the Dashboard

Open a browser and navigate to your fall-dashboard domain:

```
https://fall.focus-hospital.de
```

No login is required in the current version. The dashboard loads the patient list
and opens a live event stream automatically on page load.

---

## Dashboard Layout

### Header stats bar

At the top of the page, three values update in real time:

| Stat | What it shows |
|------|--------------|
| **Patients** | Total number of registered patients |
| **Falls (30 days)** | Total confirmed fall count across all patients in the last 30 days |
| **Live stream** | `Connected` (green) or `Disconnected` (red) — whether the SSE link to the server is active |

If the live stream shows `Disconnected`, fall alerts will not appear in real time.
See the Debug Guide for how to investigate.

---

## Patient Cards

Each registered patient appears as a card showing:

- **Patient ID** — the identifier used in the mobile app and InfluxDB
- **Display name** — optional human-readable name (set when adding the patient)
- **Fall count** — total confirmed falls in the last 30 days (from InfluxDB)
- **MAC address** — the SmarKo device MAC (optional, shown as a badge)

Click any card to open the **patient detail view** for that patient.

### Live alert states

Patient cards change appearance when a fall event arrives:

| State | Visual | What it means |
|-------|--------|--------------|
| **Normal** | Default card | No active alert |
| **Possible fall** | Card turns pale red, yellow notice bar at top: *"Possible fall, wait for confirmation"* | Mobile app detected a fall — patient has not yet answered the confirmation popup |
| **Alert** | Card stays pale red | Patient confirmed the fall, or did not respond within 10 seconds — caregiver should check on the patient |

The pale red state clears when the caregiver **clicks the card open**. Opening the
card is treated as acknowledgement.

> **When does an alert appear?**
> Not every fall event triggers a visible alert. The dashboard only activates the
> alert state when:
> - Patient did not respond within 10 seconds (`not_answered`), **or**
> - Patient confirmed it was a fall AND said they need help
>
> If the patient tapped "No, I didn't fall" or "I'm fine", the event is recorded
> in InfluxDB for history but the caregiver card does **not** turn red.

---

## Patient Detail View

Click a patient card to open the detail view. It shows:

### Summary stats (last 24 hours)

| Field | What it shows |
|-------|--------------|
| **Falls detected** | Total fall events (including unconfirmed) in the last 24 hours |
| **Confirmed** | Events where patient said "yes, I fell" |
| **Help requested** | Events where patient said "yes, I need help" |

### Fall history table

Each row is one fall event with these columns:

| Column | Description |
|--------|-------------|
| **Time** | Detection timestamp (local time) |
| **Confirmed** | Patient's response: `Confirmed` / `Not a fall` / `No response` |
| **Help needed** | Whether the patient requested help: `Yes` / `No` |
| **Confidence** | AI model confidence as a percentage (e.g. `99%`) |

**Confirmation status labels:**

| Label | Meaning | Integer in InfluxDB |
|-------|---------|:-------------------:|
| `Confirmed` | Patient responded "yes, I fell" | `1` |
| `Not a fall` | Patient responded "no, it was not a fall" | `0` |
| `No response` | Patient did not respond within 10 seconds | `-1` |

Press **← Back** to return to the patient list.

---

## Managing Patients

### Adding a patient

1. Click **+ Add Patient** (button in the top-right of the page)
2. Fill in the form:
   - **Patient ID** *(required)* — must exactly match the patient ID used in the mobile app and InfluxDB tag `patient_id`. Case-sensitive.
   - **Display name** *(optional)* — human-readable name shown on the card
   - **MAC address** *(optional)* — SmarKo device MAC address (e.g. `6c:1d:eb:04:a9:d9`)
3. Click **Add Patient**

The patient appears immediately. No pod restart needed — the patient list is stored
in SQLite on a persistent volume and survives pod restarts.

> **Patient ID must match exactly.** If the mobile app sends MQTT events for
> `patient_001` but the dashboard has `Patient_001` registered, the card will
> not update. Use the same ID as in the mobile app configuration.

### Deleting a patient

Click the **×** button in the top-right corner of a patient card. A confirmation
prompt appears. Confirming removes the patient from the dashboard.

**Deleting a patient does not delete their fall history from InfluxDB.** If the
same patient ID is added again later, their history will reappear.

---

## Fall History — Data Source

The fall history tab reads from the `fall_events` measurement in your FOCUS InfluxDB.

- History is only shown after the mobile app writes `fall_events` points to InfluxDB
  (this happens after the patient answers the confirmation popup).
- If the history tab is empty for a patient who has had alerts, the mobile app may
  not yet be writing to InfluxDB — contact MCS (Isa).
- The detail view shows the **last 24 hours** by default. The overall fall count
  on the patient card counts the **last 30 days**.

---

## Live Stream — How It Works

The dashboard opens a persistent Server-Sent Events (SSE) connection to
`/api/stream` on page load. The server pushes an event over this connection
whenever the MQTT broker receives a fall alert from the mobile app.

The browser reconnects automatically if the connection drops. The `Connected`
indicator in the header reflects the current state of this SSE connection.

The SSE connection is per browser tab — if the caregiver has the dashboard open
in multiple tabs, each tab receives its own event stream independently.
