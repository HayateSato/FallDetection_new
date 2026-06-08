# InfluxDB Schema — Fall Events

## Two InfluxDB instances

| Instance | Hosted by | Role |
|----------|-----------|------|
| **MCS cloud** (`ecosystem-influxdb.smarko-health.de`) | MCS/SmarKo | Dev/testing only. Stores raw ACC biosignal data (written by real SmarKo wearable pipeline). Used by mock_app as fake BLE input. |
| **FOCUS InfluxDB** | FOCUS (their existing infra) | Production. Stores biosignals written by the real mobile app AND `fall_events` points written after patient confirmation. |

This document covers `fall_events` only — the fall history written after each patient confirmation.

---

## `fall_events` measurement

### Who writes it

- **Dev/testing:** `mock_app/influx_writer.py` (`inject_fall_marker()`) — simulates what the real mobile app does
- **Production:** Isa's real mobile app — must write the same point after the patient confirmation popup

### When it is written

After the patient confirmation step:
1. Fall detected by inference server → `fall_detected: true` in `/predict` response
2. Mobile app shows popup: "Did you fall? Do you need help?" (10s timeout)
3. After popup closes (confirmed / denied / timed out) → write one point to InfluxDB

One point is written per fall event, regardless of patient response.

---

### Schema

**Measurement:** `fall_events`

| Kind | Name | Type | Example | Notes |
|------|------|------|---------|-------|
| Tag | `patient_id` | string | `patient_test_50` | Used for per-patient filtering in Flux queries |
| Tag | `device_id` | string | `6c:1d:eb:04:a9:d9` | MAC address of the wearable |
| Field | `fall_detected` | bool | `true` | Always `true` — only written on fall events |
| Field | `patient_confirmed` | int | `1` | See encoding table below |
| Field | `needs_help` | bool | `true` | Whether patient requested help |
| Field | `confidence` | float | `0.998` | ML model confidence score (0.0–1.0) |
| Field | `observation_id` | string | `"3fa85f64-..."` | UUID from inference server `/predict` response — cross-reference key |
| Field | `model_version` | string | `"v3"` | Model used for inference |
| Timestamp | — | nanoseconds | | Detection time (when the fall was detected, not when the popup closed) |

### `patient_confirmed` encoding

| Value | Meaning | MQTT payload string |
|-------|---------|---------------------|
| `1` | Patient confirmed it was a fall (pressed "Yes") | `"yes"` |
| `0` | Patient denied (false positive, pressed "No") | `"no"` |
| `-1` | No response within timeout (popup closed automatically) | `"not_answered"` |

**Important:** The encoding differs by layer:
- **MQTT payload** (`fall/alert/<pid>`): string — `"yes"` / `"no"` / `"not_answered"`
- **InfluxDB field**: int — `1` / `0` / `-1`
- **SSE event to browser**: int (fall_dashboard converts from string on receipt)

The conversion happens in `fall_dashboard/main.py` (`_on_fall_mqtt`) for the MQTT→SSE path, and in `mock_app/influx_writer.py` (`_CONFIRMED_TO_INT`) for the InfluxDB write path.

---

## How the fall-dashboard reads it back

`fall_dashboard/db.py` runs two Flux queries:

### 1. Fall count per patient (`_get_fall_counts`)

```flux
from(bucket: "fd_test")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "fall_events")
  |> filter(fn: (r) => r["_field"] == "fall_detected")
  |> filter(fn: (r) => r["_value"] == true)
  |> group(columns: ["patient_id"])
  |> count()
```

Returns: `{ patient_id -> count }` — shown as fall count badge on patient cards.

### 2. Fall history (`list_falls`)

```flux
from(bucket: "fd_test")
  |> range(start: -720h)
  |> filter(fn: (r) => r["_measurement"] == "fall_events")
  |> filter(fn: (r) => r["patient_id"] == "<patient_id>")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> filter(fn: (r) => r["fall_detected"] == true)
  |> sort(columns: ["_time"], desc: true)
  |> limit(n: 200)
```

Returns per-event rows with: `patient_id`, `fall_detected`, `patient_confirmed` (int), `needs_help`, `confidence`, `detection_time`.

---

## Visualizing in InfluxDB Explorer

`fall_detected` is always `true` (=1) — plotting it shows a flat line at 1. This is expected.

To see meaningful variation, switch the field to:

| Field | What you see |
|-------|-------------|
| `patient_confirmed` | `-1` / `0` / `1` — patient response per event |
| `confidence` | ML confidence score (0.0–1.0) per event |
| `needs_help` | `0` (false) / `1` (true) per event |

**Recommended visualization type:** **Table** — shows all fields per row, much more readable for event data than a line chart.

---

## Example Flux query (manual inspection)

```flux
from(bucket: "fd_test")
  |> range(start: -24h)
  |> filter(fn: (r) => r["_measurement"] == "fall_events")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time", "patient_id", "patient_confirmed", "needs_help", "confidence", "observation_id", "model_version"])
  |> sort(columns: ["_time"], desc: true)
```

---

## Mock app vs real mobile app

The mock_app (`mock_app/influx_writer.py`) simulates the InfluxDB write. The real mobile app (Isa) must replicate the same write after the confirmation popup.

Isa's checklist:
- Measurement name: `fall_events`
- Tags: `patient_id`, `device_id`
- Fields: all 6 fields listed above (especially `patient_confirmed` as **int**, not string)
- Timestamp: use the fall **detection time** from the `/predict` response, not the current time when writing
- Bucket: FOCUS's own InfluxDB bucket (not MCS cloud)
