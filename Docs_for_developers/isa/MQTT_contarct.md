## MQTT publishes the mobile app must make

There are only **two topics**, not three. "Help requested" is not a separate publish — it's a field inside the same `fall/alert` message.

---

### Topic 1 — `fall/possible/<patient_id>`

**When:** Immediately after `/predict` returns `fall_detected: true`, before showing the patient popup.

`{
  "patient_id":     "patient_test_50",
  "observation_id": "<UUID from /predict response>",
  "fall_detected":  true,
  "confidence":     0.91,
  "model_version":  "v0",
  "timestamp":      "<detection ISO 8601>"
}`

**What the dashboard does with it** (`mqtt_listener.py:165`, `app.js:148`):

- Stores `observation_id → patient_id` in a pending map
- Adds a **"Possible fall, wait for confirmation"** notice on the patient card (pale-red card)
- No database write — SSE fan-out only

---

### Topic 2 — `fall/alert/<patient_id>`

**When:** After BOTH confirmation questions are done (or timed out). Publish once with both answers combined.

`{
  "patient_id":        "patient_test_50",
  "observation_id":    "<UUID from /predict response>",
  "fall_detected":     true,
  "confidence":        0.91,
  "model_version":     "v0",
  "timestamp":         "<detection ISO 8601>",
  "alert_time":        "<time popup closed ISO 8601>",
  "patient_confirmed": "yes" | "no" | "not_answered",
  "needs_help":        true | false | null
}`

**What the dashboard does with it** (`main.py:117`):

| `patient_confirmed` | `needs_help` | Live alert to caregiver? |
| --- | --- | --- |
| `"not_answered"` | any | **Yes** — patient couldn't respond, treated as serious |
| `"yes"` | `true` | **Yes** — confirmed fall, help needed |
| `"yes"` | `false` | **No** — stored in history only |
| `"no"` | `null` | **No** — false positive, stored in history only |

---

### What feeds the fall history tab

The live MQTT messages do **not** feed the history tab. The history tab queries **FOCUS InfluxDB** directly (`db.py:60`). So if Isa doesn't write the `fall_events` point to InfluxDB, the history tab stays empty even if the live alert works perfectly.


---

### `fall/possible/<patient_id>`

| Field | Required? | Reason |
| --- | --- | --- |
| `fall_detected: true` | **Yes** | Dropped silently if `false` or missing |
| `patient_id` | **Yes** | Used to find and mark the patient card |
| `observation_id` | **Yes** | Stored in pending map — needed to clear the notice when caregiver opens the card |
| `confidence` | No | Not read |
| `model_version` | No | Not read |
| `timestamp` | No | Not read |

Minimum payload:

`{
  "fall_detected":  true,
  "patient_id":     "patient_test_50",
  "observation_id": "<UUID from /predict>"
}`

---

### `fall/alert/<patient_id>`

| Field | Required? | Reason |
| --- | --- | --- |
| `fall_detected: true` | **Yes** | Dropped silently if `false` or missing |
| `patient_id` | **Yes** | Used to identify the patient |
| `patient_confirmed` | **Yes** | Defaults to `"not_answered"` if missing → alert always fires regardless |
| `needs_help` | **Yes** | Missing = `None` → confirmed falls (`"yes"`) never trigger caregiver alert |
| `observation_id` | **Yes** | Needed to clear the pending notice from `fall/possible` |
| `confidence` | No | Not read from MQTT |
| `model_version` | No | Not read from MQTT |
| `timestamp` | No | Not read |
| `alert_time` | No | Not read |

Minimum payload:

`{
  "fall_detected":     true,
  "patient_id":        "patient_test_50",
  "observation_id":    "<UUID from /predict>",
  "patient_confirmed": "yes" | "no" | "not_answered",
  "needs_help":        true | false | null
}`


---

### Summary for Isa

| Event | Publish? | Topic | Trigger |
| --- | --- | --- | --- |
| Possible fall detected | Yes | `fall/possible/<pid>` | Immediately after `/predict` returns true |
| Fall confirmed by patient | Yes (combined) | `fall/alert/<pid>` | After BOTH popup questions close |
| Help requested by patient | No separate publish | — | `needs_help: true` inside the same `fall/alert` |
| Fall history in dashboard | Not MQTT | InfluxDB write | After popup closes, write `fall_events` point |