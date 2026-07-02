# InfluxDB Schema and Queries

## Which InfluxDB Instance

Two InfluxDB instances exist in this system, but only one is yours:

| Instance | Hosted by | Purpose |
|----------|-----------|---------|
| **FOCUS InfluxDB** | FOCUS (your existing infra) | **Production** — stores biosignal data AND `fall_events` written after patient confirmation |
| MCS cloud (`ecosystem-influxdb.smarko-health.de`) | MCS / SmarKo | Dev/testing only — used by the MCS mock app for simulated data. FOCUS does not touch this. |

This document covers the `fall_events` measurement only — fall history written by the mobile
app after each patient confirmation. The biosignal measurements already in your bucket
(SMART_DATA, etc.) are unaffected.

---

## The `fall_events` Measurement

### Who writes it

| Environment | Writer |
|-------------|--------|
| Testing / staging | `mock_app` on the MCS side (`inject_fall_marker()`) — simulates what the real mobile app will do |
| Production | **Isa's real mobile app** — must write one point after the patient answers the confirmation popup |

> **Current status:** The mobile app is not yet writing to InfluxDB (open gap as of 2026-06-10).
> Until Isa completes this, the fall history tab in the dashboard will be empty.
> Live alerts via MQTT still work — they do not depend on InfluxDB writes.

### When it is written

1. SmarKo wearable detects a fall -> inference server returns `fall_detected: true`
2. Mobile app shows the patient a popup: "Did you fall? Do you need help?" (10-second timeout)
3. After the popup closes (confirmed / denied / timed out) -> **write one point to InfluxDB**

One point per fall event, regardless of the patient's response.

---

## Schema

**Measurement:** `fall_events`

| Kind | Name | Type | Example | Notes |
|------|------|------|---------|-------|
| Tag | `patient_id` | string | `patient_test_50` | Must match the ID used in the mobile app and dashboard. Used for filtering in Flux queries before pivot. |
| Tag | `device_id` | string | `6c:1d:eb:04:a9:d9` | MAC address of the SmarKo wearable |
| Field | `fall_detected` | bool | `true` | Always `true` -- only fall events are written |
| Field | `patient_confirmed` | **int** | `1` | Patient response -- see encoding table below |
| Field | `needs_help` | bool | `true` | Whether the patient said they need help |
| Field | `confidence` | float | `0.998` | ML model confidence score (0.0 to 1.0) |
| Field | `observation_id` | string | `"3fa85f64-..."` | UUID from the inference server -- links this event back to MCS inference logs |
| Field | `model_version` | string | `"v3"` | Which ML model version was used |
| Timestamp | -- | nanoseconds UTC | | Fall **detection time** -- not the time the popup closed |

### `patient_confirmed` encoding

| InfluxDB value | Meaning | MQTT string equivalent |
|:--------------:|---------|------------------------|
| `1` | Patient pressed "Yes, I fell" | `"yes"` |
| `0` | Patient pressed "No, I did not fall" | `"no"` |
| `-1` | No response -- popup timed out after 10 seconds | `"not_answered"` |

The `fall_dashboard` converts from the MQTT string to the integer when it receives
the event. Isa's mobile app must write the **integer** directly to InfluxDB.

---

## Critical: Write `patient_confirmed` as an Integer

InfluxDB locks a field's type on the first write to a measurement. If `patient_confirmed`
is written as a string (`"1"` or `"yes"`) instead of an integer (`1`), all subsequent
integer writes will silently fail and InfluxDB will only store the string values.

**The field type cannot be changed without dropping the measurement.**

Isa's checklist for the InfluxDB write:
- `patient_confirmed`: integer (`1`, `0`, or `-1`) -- not a string
- `fall_detected`: boolean (`true`) -- not a string
- `needs_help`: boolean (`true`/`false`) -- not a string
- `confidence`: float (`0.998`) -- not a string
- `observation_id`: string (UUID) -- this one is a string
- Timestamp: use the fall **detection time** from the `/predict` response, not the time of the write

---

## Bucket Configuration

The fall-dashboard reads from the bucket configured in `values_production.yaml`:

```yaml
fallDashboard:
  influxdb:
    url: "https://influxdb.focus-hospital.de"
    org: "focus"
    fallEventsBucket: "fd_production"   # bucket where fall_events are written
    token: ""                            # read-only token -- set in the secret block
```

The mobile app writes to the same bucket. If the bucket name differs between the mobile
app config and `fallEventsBucket`, fall history will not appear in the dashboard.

If FOCUS uses a single bucket for all measurements (including biosignals), set
`fallEventsBucket` to that bucket name -- the Flux queries filter by measurement name
(`fall_events`) so other measurements are not affected.

---

## Flux Queries Used by the Dashboard

The fall-dashboard runs these queries from `fall_dashboard/db.py`.

### 1. Fall count per patient (card badges)

Shown as the fall count on each patient card. Counts distinct fall events in the
last 30 days across all patients in one query.

```flux
from(bucket: "fd_production")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "fall_events")
  |> filter(fn: (r) => r["_field"] == "fall_detected")
  |> filter(fn: (r) => r["_value"] == true)
  |> group(columns: ["patient_id"])
  |> count()
```

Returns one row per patient with a `_value` count. The dashboard maps this to
`{ patient_id -> count }`.

### 2. Fall history for a patient (detail view)

Used in the patient detail view. Returns all fall events for a single patient
as wide rows (one column per field) rather than the default long format.

```flux
from(bucket: "fd_production")
  |> range(start: -720h)
  |> filter(fn: (r) => r["_measurement"] == "fall_events")
  |> filter(fn: (r) => r["patient_id"] == "patient_test_50")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> filter(fn: (r) => r["fall_detected"] == true)
  |> sort(columns: ["_time"], desc: true)
  |> limit(n: 200)
```

> **Why `pivot` before filtering `fall_detected`:**
> `patient_id` is a tag and can be filtered before `pivot`. `fall_detected` is
> a field -- it only exists as a column after `pivot`. Filtering a field before
> `pivot` (when it is still in `_field`/`_value` rows) would drop all other fields
> for that timestamp.

After pivot, each row has these columns: `patient_id`, `fall_detected`,
`patient_confirmed` (int), `needs_help`, `confidence`, `observation_id`,
`model_version`, `_time`.

The patient detail view shows the last 24 hours by default -- this is the same
query with `range(start: -24h)`.

---

## Manual Inspection in InfluxDB Data Explorer

Use this query to inspect fall events directly in the InfluxDB UI:

```flux
from(bucket: "fd_production")
  |> range(start: -24h)
  |> filter(fn: (r) => r["_measurement"] == "fall_events")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time", "patient_id", "patient_confirmed", "needs_help",
                    "confidence", "observation_id", "model_version"])
  |> sort(columns: ["_time"], desc: true)
```

**Recommended visualization:** Table view -- shows all fields per event as columns,
which is far more readable than a line chart for discrete event data.

> **Note on `fall_detected` in a line chart:** `fall_detected` is always `true` (=1)
> because only fall events are written. Plotting it produces a flat line at 1, which
> is expected and not useful. Switch the field to `confidence` or `patient_confirmed`
> to see meaningful variation.

---

## Filtering on `patient_confirmed`

Because `patient_confirmed` is an integer field, filter with integers -- not strings:

```flux
// Confirmed falls only
|> filter(fn: (r) => r["patient_confirmed"] == 1)

// False positives (patient denied)
|> filter(fn: (r) => r["patient_confirmed"] == 0)

// No response
|> filter(fn: (r) => r["patient_confirmed"] == -1)
```

Using string comparisons (`== "1"` or `== "yes"`) will match nothing.

---

## Linking Events Back to MCS

Each `fall_events` point contains an `observation_id` field -- a UUID returned by the
MCS inference server in the `/predict` response. This UUID is also stored in the MCS
`inference_log` Postgres table, which holds the raw inference result plus which patient
and model were used.

If you need to investigate a specific event that appeared in the dashboard, share the
`observation_id` with MCS (Mohammed or the MCS dev team) to look up the full
inference record.

---

## Related Documents

- [07_fall_dashboard_user_guide.md](07_fall_dashboard_user_guide.md) -- explains what the dashboard
  shows and where the fall history data comes from (from the caregiver's perspective)
- [08_debug_guide.md](08_debug_guide.md) -- "Fall history tab is empty" troubleshooting steps
- [03_k3s_values_and_secrets.md](03_k3s_values_and_secrets.md) -- `fallEventsBucket`,
  `INFLUXDB_URL`, `INFLUXDB_TOKEN`, and `INFLUXDB_ORG` are all set there
