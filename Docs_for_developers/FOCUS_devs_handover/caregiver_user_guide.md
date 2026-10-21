# Caregiver Dashboard — User Guide

## Overview

The Caregiver Dashboard is a browser-based tool that shows real-time fall alerts and fall history for registered patients. Open it in any browser by navigating to the fall dashboard URL provided by your IT team.

The header shows three summary numbers at all times:

| Indicator | What it shows |
|-----------|--------------|
| **Patients** | Total number of registered patients |
| **Falls today** | Number of fall events detected today across all patients |
| **Live status** | `Connected` (green) or `Disconnected` (red) — whether the dashboard is receiving live alerts |

If **Live status** shows `Disconnected`, refresh the page. Alerts are only received in real time when this shows `Connected`.

---

## Patient Cards

Each registered patient appears as a card. The card changes appearance depending on what is happening with that patient.

### Normal state — no alert

The card shows the patient's name, ID, MAC address (if set), and the total number of recorded falls. No banner is shown.

### Possible fall — waiting for confirmation

A light banner reading **"Possible fall, wait for confirmation"** appears at the top of the card, and the card background turns light red.

This means the mobile app detected a fall and is waiting for the patient to respond to the confirmation popup on their phone. No action is needed from the caregiver yet — wait to see if the alert escalates.

### Fall confirmed — help requested

The banner changes to **"Fall is confirmed"** (solid red background). Below the patient name, a large message **"Help is requested"** appears on a pale red background.

This means:
- The patient confirmed they fell **and** indicated they need assistance, **or**
- The patient did not respond within the timeout window (10 seconds per question)

**Immediate caregiver action is required.**

### Clearing an alert

Click anywhere on the patient card to open the detail view. This automatically clears the "Possible fall" notice from the card.

For a confirmed fall + help requested alert, the card returns to normal state when you open the detail view and close it again.

---

## Adding a Patient

1. Open the Fall Dashboard in your browser.
2. Click **+ Add Patient** (top right of the patient list).
3. Fill in the form:
   - **Patient ID** (required) — must exactly match the patient ID the mobile app sends in the `/predict` API call. This is how the dashboard links incoming fall alerts to the correct patient card.
   - **Name** (optional) — display name shown on the card.
   - **MAC address** (optional) — the SmarKo wearable's BLE MAC address.
4. Click **Add Patient** — the card appears immediately. No restart needed.

> **Important:** The Patient ID must match exactly what the mobile app sends. If the IDs do not match, fall alerts arrive but cannot be linked to a card and will be silently ignored. Confirm the correct Patient IDs with the mobile app developer (Isa) before adding patients.

---

## Removing a Patient

1. Find the patient card you want to remove.
2. Click the **×** button in the top-right corner of the card.
3. The card is removed immediately.

Fall history stored in the database is not deleted — only the card is removed from the dashboard view. If you re-add the same Patient ID later, the historical data will be visible again.

---

## Viewing a Patient's Fall History

Click anywhere on a patient card to open the detail view for that patient.

### Summary stats

Three numbers are shown at the top:

| Stat | What it shows |
|------|--------------|
| **Falls in last 24h** | Number of fall events detected in the past 24 hours |
| **Confirmed falls** | Falls where the patient confirmed it was a fall |
| **Help requested** | Falls where the patient requested assistance (or did not respond) |

### Fall history table

The table lists individual fall events in reverse chronological order (most recent first). Each row shows:

| Column | What it shows |
|--------|--------------|
| **Detection time** | Date and time when the fall was detected |
| **Patient response** | `Yes` (confirmed fall) / `No` (denied, false positive) / `Not answered` (no response within timeout) |
| **Needs help** | `Yes` / `No` |

### Interpreting patient response

| Response | Meaning |
|----------|---------|
| **Yes** | Patient confirmed on their phone that they fell |
| **No** | Patient indicated it was a false alarm |
| **Not answered** | Patient did not respond within 10 seconds — treat as a real fall |

---

## What Triggers an Alert

The dashboard receives alerts published by the patient's mobile app. The sequence is:

1. The SmarKo wearable detects movement. The mobile app sends the sensor data to the inference server.
2. If the inference server predicts a fall, the mobile app immediately shows the patient a popup: **"Did you fall?"** (10-second timer).
3. The dashboard shows **"Possible fall, wait for confirmation"** on the patient's card.
4. The mobile app then asks: **"Do you need help?"** (10-second timer).
5. If the patient confirms a fall and requests help — or does not respond to either question — the dashboard escalates to **"Fall is confirmed / Help is requested"**.

No alert appears on the dashboard for events where the patient actively denied the fall (`No` response to "Did you fall?").

---

## Troubleshooting

| Problem | What to check |
|---------|--------------|
| **Live status shows Disconnected** | Refresh the page. If it persists, contact IT — the dashboard service may be down. |
| **Patient card not appearing after add** | Check that the Patient ID was entered correctly. Refresh the page if the card does not appear immediately. |
| **Alert arrives but no card highlights** | The Patient ID in the mobile app does not match the ID registered in the dashboard. Confirm the correct ID with Isa. |
| **Fall history table is empty** | The mobile app may not yet be writing fall events to InfluxDB. Contact MCS / the mobile app developer. |
| **"Possible fall" notice stays after opening the card** | This clears automatically when you click the card. If it does not clear, refresh the page. |
