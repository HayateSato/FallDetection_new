# System Overview — What You Are Adding and Why

## Background

Your FOCUS network already runs an infrastructure for monitoring elderly patients wearing SmarKo
wearable sensors. The mobile app reads the patient's biometric data (heart rate, SpO2, accelerometer)
and writes it continuously to your InfluxDB. Your Flutter patient dashboard reads from that InfluxDB
and displays the live health status per patient.

The fall detection system adds one new capability on top of this existing setup:
**automatic detection of patient falls and real-time alerting to the caregiver.**

---

## What Changes in Your Cluster

You are adding **two new pods** to your existing k3s cluster. Everything else remains unchanged.

| Pod | What it is | Why it is needed |
|-----|-----------|-----------------|
| `mosquitto` | MQTT message broker | The mobile app needs a channel to send fall alerts into your network in real time. Mosquitto is the broker that sits in the middle — the mobile app publishes to it, and the caregiver dashboard subscribes to it. |
| `fall-dashboard` | Caregiver alert web UI | Displays incoming fall alerts to the caregiver in real time. Also shows a per-patient fall history tab, reading from your existing InfluxDB. |

- Your existing services — InfluxDB, patient dashboard, caregiver webapp — are not modified.
- Your two pods (mosquitto, fall-dashboard) do not connect to the services running on MCS at all. The only system that talks to the inference server is the mobile app.
---

## How the New Pods Sit Inside Your Cluster

```
Your existing k3s cluster (FOCUS network)
 ┌────────────────────────────────────────────────────────────────────┐
 │                                                                    │
 │  Existing (no changes)            New (added by this deployment)   │
 │  ─────────────────────            ──────────────────────────────   │
 │  InfluxDB           (F1)          MQTT broker / mosquitto  (F2)    │
 │  Patient dashboard  (F1)     ◄──  Fall dashboard           (F2)    │
 │  Caregiver webapp   (F1)          │                                │
 │                                   │ reads fall_events              │
 │                        InfluxDB ◄─┘                                │
 └────────────────────────────────────────────────────────────────────┘
          ▲ MQTT WSS (:443)              ▲ HTTPS (:443)
          │                              │
    Mobile app (patient side)      Caregiver browser
```
- Both new pods are exposed through your **existing Traefik ingress** on standard ports only
(443 HTTPS / WSS). No new firewall rules are needed if Traefik is already serving other services.
- F1 indidates exsiting services while F2 indicates new ones.

---

## How a Fall Alert Reaches the Caregiver

1. The patient's SmarKo wearable sends accelerometer data to the mobile app via Bluetooth.
2. The mobile app sends this data over HTTPS to the **MCS inference server** (hosted by MCS on
   Hetzner — not in your cluster). The inference server runs the fall detection AI model and returns
   a result within a few hundred milliseconds.
3. If a fall is detected, the mobile app first shows the patient a confirmation popup
   ("Did you fall? Do you need help?").
4. The mobile app **publishes an MQTT message** to your `mosquitto` pod via WebSocket over port 443.
   Two messages are published:
   - `fall/possible/<patient_id>` — immediately on detection (before patient answers)
   - `fall/alert/<patient_id>` — after the patient responds (or after a 10-second timeout)
5. The `fall-dashboard` pod subscribes to both topics on the broker and **pushes a live alert to the
   caregiver's browser** via Server-Sent Events (SSE).
6. After the patient responds, the mobile app also writes a `fall_events` data point to your InfluxDB.
   The fall-dashboard reads this measurement to populate the **fall history tab**.

```
SmarKo wearable
    │  Bluetooth
    ▼
Mobile app
    │  HTTPS /predict ──────────────────────► MCS inference server (Hetzner)
    │                  ◄──────────────────────  { fall_detected, observation_id }
    │
    ├─ MQTT publish ──► mosquitto pod (your cluster, port 443 WSS)
    │                       │  internal TCP (:1883)
    │                       ▼
    │                  fall-dashboard pod ──► caregiver browser (SSE live alert)
    │
    └─ InfluxDB write ──► your InfluxDB (fall_events measurement)
                              ▲
                         fall-dashboard reads this for fall history tab
```

<!-- --- -->

<!-- ## What the MCS Inference Server Is (and Why You Don't Manage It) -->

<!-- MCS hosts an inference server on Hetzner (their own machine). It does one thing: receive a
9-second sensor window from the mobile app, run an XGBoost model, and return whether a fall was
detected. This is entirely MCS-managed.

**You have no dependency on it** — your two pods (mosquitto, fall-dashboard) do not connect to
Hetzner at all. The only system that talks to the inference server is the mobile app. -->

---

## What Each New Pod Uses From Your Existing Stack

| New pod | Uses from your existing stack | What for |
|---------|------------------------------|----------|
| `fall-dashboard` | Your InfluxDB | Reads the `fall_events` measurement to display fall history per patient |
| `fall-dashboard` | Your patient list (from SQLite on PVC) | Knows which patient IDs to subscribe to and display |
| `mosquitto` | Nothing | Self-contained broker; only the mobile app and fall-dashboard connect to it |

The fall-dashboard does **not** write to your InfluxDB — the mobile app writes `fall_events`.
The fall-dashboard only reads.

---

## What Gets Added to the Caregiver's View

Your caregiver currently uses the Flutter patient dashboard (F1) to view patient health status.
The fall-dashboard (F2) is a **separate web page** served at its own subdomain
(e.g. `https://fall.focus-hospital.de`). It is not embedded inside the patient dashboard.

> Future option (Q1): if FOCUS grants source repo access to the Flutter patient dashboard, the
> fall alert features can be merged directly into the existing dashboard — eliminating the need
> for a separate URL. This is a planned upgrade, not part of the current deployment.

The fall-dashboard provides:
- **Live alert panel** — card per patient that turns pale red on possible fall, full alert on
  confirmed fall; shows whether the patient requested help
- **Fall history tab** — table of all recorded falls, filterable by patient, date range, and
  help-requested status; sourced from `fall_events` in InfluxDB

---

## Summary of What Is and Is Not Your Responsibility

| Component | Owner | Your action |
|-----------|-------|-------------|
| SmarKo wearable | Patient / hardware team | None |
| Mobile app | MCS (Isa) | None — you share MQTT credentials with Isa |
| MCS inference server (Hetzner) | MCS (Mohammed) | None |
| Your InfluxDB | FOCUS | No changes — fall-dashboard only reads from it |
| Your patient dashboard (Flutter) | FOCUS DevOps | No changes |
| **MQTT broker (mosquitto)** | **FOCUS DevOps** | **Deploy and configure** |
| **Fall dashboard** | **FOCUS DevOps** | **Deploy and configure** |

For deployment steps, see [`focus_devops_handover.md`](../focus_devops_handover.md).
For configuration values, see [`config_checklist_focus_devops.md`](../config_checklist_focus_devops.md).
