# Isa — Mobile App To-Do List

This document describes a list of features mobile app should include.
Derived from `_6G_integration_v3_docker_focus/mock_app`, which simulates exactly what the real mobile app must do.

# Running services on Hetzner
- Inference Server URL: https://fall-api.smarko-health.de/
    - API key: J7gEYm2mOgaLqRRKhCAOZZpYOz_VnoHo-CwlFUJBMfc
- Fall Dashboard URL: http://5.75.255.114:8002/ (it should be https://internal.e-healthservice.de/<xxx> in FOCUS)
- Influx URL (mock version of FOCUS's influx): http://5.75.255.114:8086/signin (the real one is https://influxdb.internal.e-healthservice.de/)

---

## 1. Sensor Data Reading

- [ ] **Read a 9-second ACC window from the SmarKo wearable over BLE.**
  - 3 axes (X, Y, Z), timestamps in milliseconds
  - 50 Hz = 450 samples per window
  - Query `GET /model/info` first to check `uses_barometer`. If `true`, include `pressure[]` and `pressure_timestamps_ms[]` in the `/predict` payload.

- [ ] **Write raw sensor data to FOCUS InfluxDB continuously.**
  - Measurement: `SMART_DATA`
  - Tags: `macAddress` (device MAC), `patient_id`
  - Fields: `acc_x`, `acc_y`, `acc_z`, `pressure` (if available)
  - This is separate from `fall_events` — it feeds the caregiver dashboard's biosignal display and the fall history tab.

---

## 2. Fall Detection (calling inference server)

- [ ] **Query `GET /model/info` once on startup (or once per session).** Cache the `uses_barometer` flag. Use it to decide whether to include pressure data in `/predict`.

- [ ] **POST `/predict` to the MCS inference server.**
  - URL: inference server URL (from Mohammed after Hetzner deploy)
  - Header: `X-API-Key: <key from Mohammed>`
  - Body:
    ```json
    {
      "patient_id":             "<pid>",
      "device_id":              "<MAC>",
      "acc_x":                  [float, ...],
      "acc_y":                  [float, ...],
      "acc_z":                  [float, ...],
      "timestamps_ms":          [float, ...],
      "pressure":               [float, ...],
      "pressure_timestamps_ms": [float, ...]
    }
    ```
    (`pressure` and `pressure_timestamps_ms` are optional — include only if `uses_barometer=true`)
  - Response: `{ inference: { fall_detected, confidence, model_version }, observation_id, timestamp }`
  - **Save `observation_id`** — needed for the `/confirm` call later.

---

## 3. Immediate MQTT Publish (before patient sees popup)

- [ ] **Publish `fall/possible/<patient_id>` as soon as `fall_detected=true` is returned — do NOT wait for patient confirmation.**
  - Topic: `fall/possible/<patient_id>`
  - Payload:
    ```json
    {
      "patient_id":     "<pid>",
      "observation_id": "<uuid from /predict>",
      "fall_detected":  true,
      "confidence":     0.91,
      "model_version":  "v0",
      "timestamp":      "<ISO 8601>",
      "status":         "pending"
    }
    ```
  - This shows a "possible fall" notice on the caregiver dashboard while the patient confirmation countdown is running.

---

## 4. Patient Confirmation Popup

- [ ] **Each question has its own 10-second timer — NOT a shared 10-second window for both questions.**

  **Step 1:** Show "Did you fall?" popup. Timer: 10 seconds.
  - Yes → go to Step 2
  - No → `patient_confirmed = "no"`, `needs_help = null`. Done.
  - Timeout → `patient_confirmed = "not_answered"`, `needs_help = null` (alert still sent to caregiver).

  **Step 2** (only if Step 1 = Yes): Show "Do you need help?" popup. Timer: 10 seconds.
  - Yes → `needs_help = true`
  - No → `needs_help = false`
  - Timeout → `needs_help = null`
  - `patient_confirmed = "yes"` in all cases.

---

## 5. After Patient Confirmation (or timeout)

Steps 5, 6, and 7 all fire together after the popup completes.

- [ ] **POST `/inference/{observation_id}/confirm` to MCS inference server.**
  - URL: `POST <inference-server>/inference/<observation_id>/confirm`
  - Header: `X-API-Key`
  - Body:
    ```json
    {
      "patient_confirmed": "yes" | "no" | "not_answered",
      "needs_help":        true | false | null
    }
    ```
  - This updates the `inference_log` row in MCS Postgres — used for model retraining.

- [ ] **Publish `fall/alert/<patient_id>` to MQTT broker via WebSocket.**
  - Topic: `fall/alert/<patient_id>`
  - Protocol: MQTT over WebSocket (WSS) — **NOT raw TCP**. React Native cannot open raw TCP sockets. Use MQTT.js with WebSocket transport: `wss://<FOCUS-domain>:443`
  - Payload:
    ```json
    {
      "patient_id":        "<pid>",
      "observation_id":    "<uuid>",
      "fall_detected":     true,
      "confidence":        0.91,
      "model_version":     "v0",
      "timestamp":         "<detection ISO 8601>",
      "alert_time":        "<confirmation ISO 8601>",
      "patient_confirmed": "yes" | "no" | "not_answered",
      "needs_help":        true | false | null
    }
    ```
  - MQTT connection details (from FOCUS DevOps after they deploy):
    - host: `mosquitto.ingress.host` (e.g. `mqtt.focus-hospital.de`)
    - port: `443` (production via Traefik)
    - username: `fallDashboard.mqtt.username` (blank if anonymous allowed)
    - password: `fallDashboard.mqtt.password`

- [ ] **Write `fall_events` point to FOCUS InfluxDB.**
  - Measurement: `fall_events`
  - Tags: `patient_id`, `device_id` (MAC address)
  - Fields:

    | Field | Type | Value |
    |-------|------|-------|
    | `fall_detected` | bool | `true` |
    | `patient_confirmed` | int | `1` = yes, `0` = no, `-1` = not_answered |
    | `needs_help` | bool | |
    | `confidence` | float | |
    | `observation_id` | string | UUID from `/predict` response |
    | `model_version` | string | |

  - Timestamp: **detection time** (from `/predict` response) — NOT the current time when writing.
  - InfluxDB write credentials (from FOCUS DevOps): URL, org, bucket, token — same InfluxDB instance as `SMART_DATA`.

---

## 6. Timestamp Logging

- [ ] **Record and store the following three timestamps per fall event:**
  - `predicted_at` — timestamp from `/predict` response
  - `confirmed_at` — time patient responded (or timeout expired)
  - `help_requested_at` — time patient answered "Do you need help?" (`null` if they said No at Step 1 or timed out before Step 2)

---

## Summary

- [Isa — Mobile App To-Do List](#isa--mobile-app-to-do-list)
- [Running services on Hetzner](#running-services-on-hetzner)
  - [1. Sensor Data Reading](#1-sensor-data-reading)
  - [2. Fall Detection (calling inference server)](#2-fall-detection-calling-inference-server)
  - [3. Immediate MQTT Publish (before patient sees popup)](#3-immediate-mqtt-publish-before-patient-sees-popup)
  - [4. Patient Confirmation Popup](#4-patient-confirmation-popup)
  - [5. After Patient Confirmation (or timeout)](#5-after-patient-confirmation-or-timeout)
  - [6. Timestamp Logging](#6-timestamp-logging)
  - [Summary](#summary)
