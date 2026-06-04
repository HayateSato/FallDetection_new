# Mobile App ↔ Caregiver Layer — Publisher Contract (for Isa)

**To:** Isa (mobile app)
**From:** MCS (fall detection / caregiver layer)
**Purpose:** what the mobile app must send so the caregiver dashboard shows fall
alerts correctly, once it replaces the mock app.

> The mock app (`mock-app-publisher`) currently does everything below. When your
> real mobile app takes over, **the broker and dashboard need no changes** — they
> route purely by topic, not by who publishes. You just have to match the topics
> and the JSON payload shape exactly. This doc is that exact contract.

---

## 1. The big picture — what the mobile app is responsible for

On each fall, the mobile app does **four** things (the mock app does all four; you
replace it):

1. Call the inference server `POST /predict` with the raw ACC window → get back
   `fall_detected`, `confidence`, `observation_id`.
2. If `fall_detected = true` → **publish `fall/possible/<patient_id>`** (pre-alert,
   so the caregiver sees a "possible fall" immediately) and show the patient popup
   ("Did you fall? / Do you need help?").
3. After the patient responds (or the timeout) → **publish `fall/alert/<patient_id>`**
   with the patient's answer.
4. **Write one `fall_events` point to InfluxDB** (for fall history) and call
   `POST /inference/{observation_id}/confirm` on the inference server (for the
   retraining ground-truth label).

The dashboard subscribes to `fall/alert/#` and `fall/possible/#` and reacts. It
never talks back to you — MQTT is one-way here.

---

## 2. MQTT connection

| Setting | Value |
|---------|-------|
| Broker host | the caregiver-layer machine's IP (the laptop running `focus_mqtt`) |
| Port | `1883` (plain TCP) |
| Auth | none for local testing (`allow_anonymous true`). Production will add user/pass — ask MCS. |
| **Client ID** | **must be unique.** Do NOT reuse `mock-app-publisher`. Use something stable per device, e.g. `mobile-<deviceid>`. ⚠️ See §6. |
| QoS | currently 0 (fire-and-forget). See §7 — alerts may move to QoS 1. |

---

## 3. Topic structure (must match exactly)

Two topics. The `<patient_id>` is appended as the last level.

| Topic | When you publish it | Dashboard reaction |
|-------|--------------------|--------------------|
| `fall/possible/<patient_id>` | **Immediately** on fall detection, before the patient answers | Shows a yellow "Possible fall" badge on that patient's card |
| `fall/alert/<patient_id>` | **After** the patient answers (or the popup times out) | Decides whether to raise the red alert (see §4 logic) |

Rules:
- The prefix must be exactly `fall/possible/` or `fall/alert/` (that is what the
  dashboard subscribed to). Anything else is ignored.
- `<patient_id>` must be the **same id** used by the sensor/InfluxDB (e.g.
  `patient_test_4`). It is the join key across MQTT, InfluxDB, and the dashboard
  patient list. If it doesn't match, the alert won't tie to the right patient.

Example: patient `patient_test_4` →
`fall/possible/patient_test_4` then `fall/alert/patient_test_4`.

---

## 4. Payload format (JSON) — the critical part

Both topics carry a **JSON object**. The dashboard reads specific fields; missing
or wrong fields cause the message to be **silently dropped**, so match these.

### 4a. `fall/possible/<patient_id>` payload (pre-confirmation)

```json
{
  "patient_id":     "patient_test_4",
  "device_id":      "6c:1d:eb:04:a9:d9",
  "observation_id": "a1b2c3d4-...uuid-from-/predict...",
  "fall_detected":  true,
  "confidence":     0.998,
  "status":         "pending",
  "timestamp":      "2026-06-04T10:36:47.622Z",
  "model_version":  "v1.3"
}
```

### 4b. `fall/alert/<patient_id>` payload (after patient answers)

```json
{
  "patient_id":        "patient_test_4",
  "device_id":         "6c:1d:eb:04:a9:d9",
  "observation_id":    "a1b2c3d4-...same-uuid...",
  "fall_detected":     true,
  "confidence":        0.998,
  "patient_confirmed": "yes",
  "needs_help":        true,
  "alert_time":        "2026-06-04T10:36:50.840Z",
  "model_version":     "v1.3"
}
```

### Field reference

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `patient_id` | string | ✅ | Must match sensor/InfluxDB id. |
| `fall_detected` | bool | ✅ **must be `true`** | Dashboard drops the message if this is missing/false. This is the #1 gotcha. |
| `observation_id` | string (UUID) | ✅ | The UUID returned by `/predict`. Same value in both the possible and alert messages — the dashboard uses it to clear the "possible" badge when the alert arrives. Also the key for `/confirm`. |
| `patient_confirmed` | string | ✅ on `fall/alert` | One of **`"yes"` / `"no"` / `"not_answered"`** (string, lowercase). Drives the alert logic below. Omit on `fall/possible`. |
| `needs_help` | bool | ✅ on `fall/alert` | `true` if the patient asked for help. Drives the alert logic. |
| `status` | string | recommended on `fall/possible` | `"pending"`. (The dashboard also infers this from the topic, but send it for clarity.) |
| `confidence` | float 0–1 | recommended | Shown in the alert text. |
| `device_id` | string | recommended | Sensor MAC. Shown on the card / used as fallback label. |
| `timestamp` / `alert_time` | ISO-8601 string | recommended | Detection time / alert time. |
| `model_version` | string | optional | For traceability. |

### ⚠️ Critical: `patient_confirmed` is a STRING here

On the **MQTT message** send the **string** `"yes"` / `"no"` / `"not_answered"`.
The caregiver layer converts it to an int internally (`1` / `0` / `-1`) for
InfluxDB and the browser — **you do not send the int.** Do not send booleans
(`true`/`false`) for this field.

---

## 5. When does the dashboard actually raise the RED alert?

Important so you understand why some confirmed falls don't show a red banner.
The dashboard raises the urgent caregiver alert **only** when:

- `patient_confirmed = "not_answered"`  (patient could not respond → treat as serious), **OR**
- `patient_confirmed = "yes"` **AND** `needs_help = true`  (confirmed fall, help requested)

It stays **silent** (records to history, no red alert) when:
- `patient_confirmed = "no"`  (patient says it wasn't a fall → false positive), or
- `patient_confirmed = "yes"` but `needs_help = false`  (fell but is OK)

So `needs_help` matters even when the patient confirms the fall. Make sure the
popup captures both answers ("did you fall?" **and** "do you need help?").

---

## 6. Client ID — don't get kicked off

MQTT requires every connected client to have a **unique client ID**. If two
clients connect with the same ID, the broker disconnects one of them — you'll see
rapid disconnect/reconnect loops and dropped messages.

- Do **not** use `mock-app-publisher` (that's the mock's ID).
- Use a stable, unique ID per device, e.g. `mobile-<serial>` or `mobile-<patientid>`.
- If you run multiple phones, each needs its **own** ID. They can all publish to
  their own `fall/alert/<patient_id>` — the dashboard receives all of them via the
  `#` wildcard. Multiple publishers → one broker → all forwarded to the dashboard.

---

## 7. QoS (delivery reliability) — heads-up

Currently everything is **QoS 0** ("fire and forget") — the broker does not
confirm the dashboard received the message, and an alert sent while the dashboard
is momentarily disconnected is **lost**. For safety-critical alerts MCS may move
the `fall/alert` path to **QoS 1** (broker retries until acknowledged).

If/when we do that, you'd publish the alert with `qos=1`. QoS 1 can deliver
**duplicates**, so make the receiver/idempotency safe by keying on
`observation_id` (already unique per fall). MCS will coordinate this change — just
be aware it may come.

---

## 8. InfluxDB write (step 4) — schema you must match

Besides MQTT, after the patient answers, write **one** `fall_events` point so the
dashboard's fall **history** and **counts** work. Schema (agreed MCS↔FOCUS):

| | |
|---|---|
| Measurement | `fall_events` |
| Tags | `patient_id`, `device_id` |
| Fields | `fall_detected` (bool), `patient_confirmed` (**int**: 1/0/-1), `needs_help` (bool), `confidence` (float) |
| Timestamp | the fall detection time |

⚠️ Note the asymmetry: in **InfluxDB**, `patient_confirmed` is an **int** (`1` yes
/ `0` no / `-1` not_answered). In the **MQTT message** it is a **string**. Same
concept, different encoding per channel. (MCS handles this in the mock app; your
app must do the same on both channels.)

Also call `POST /inference/{observation_id}/confirm` on the inference server with
the patient's answer so the retraining pipeline gets the ground-truth label.

---

## 9. Quick checklist for Isa

- [ ] Connect to broker `tcp://<caregiver-machine-ip>:1883`, **unique client ID**.
- [ ] On fall detected → publish `fall/possible/<patient_id>` with `fall_detected:true`, `status:"pending"`, `observation_id`.
- [ ] Show popup capturing **both** "did you fall?" and "do you need help?".
- [ ] On answer/timeout → publish `fall/alert/<patient_id>` with `patient_confirmed` (`"yes"`/`"no"`/`"not_answered"`) and `needs_help` bool, same `observation_id`.
- [ ] `patient_confirmed` = **string** on MQTT, **int** on InfluxDB.
- [ ] Write the `fall_events` point to InfluxDB (schema in §8).
- [ ] Call `/inference/{observation_id}/confirm`.
- [ ] `patient_id` identical across MQTT, InfluxDB, and the sensor.

---

## 10. How to test your app against the live dashboard

1. Make sure the mock app is **stopped** (`docker stop focus_mock_app`) so it
   doesn't publish competing messages.
2. Point your app's broker host at the caregiver-layer laptop's IP, port 1883.
3. Publish a test `fall/alert/<patient_id>` and watch it arrive:
   - Broker log: `docker logs -f focus_mqtt` — look for `Received PUBLISH from <your-client-id> 'fall/alert/...'` followed by `Sending PUBLISH to fall-detection-caregiver`.
   - Dashboard log: `docker logs -f focus_fall_dashboard` — confirms the app *processed* it.
   - Browser: `http://<caregiver-ip>:8002/` — the alert/badge should appear.

If you see the broker `Received` line but the badge doesn't show, it's almost
always a payload mismatch — re-check §4 (especially `fall_detected:true` and the
`patient_confirmed` string values).

---

## Reference: the exact mock-app code this mirrors

- MQTT publishes: `_6G_Integration_v2_mqtt/local_dev/mock_app/poller.py`
  (possible-fall ≈ line 230, alert ≈ line 132).
- InfluxDB write: `_6G_Integration_v2_mqtt/local_dev/mock_app/influx_writer.py`.
- Dashboard side (for reference, not yours to change):
  `fall_dashboard/mqtt_listener.py` (subscriptions + the `fall_detected` filter)
  and `fall_dashboard/main.py` (the alert logic in §5).

Related: `__Refactoring_docs/mqtt_broker_logs_and_qos.md` (how to read the broker log + QoS detail).
