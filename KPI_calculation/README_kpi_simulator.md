# KPI Multi-Device Simulator — How to Run

## Where it lives

```
KPI_calculation/
  kpi_multi_device_simulator.py     <- the script
  extract_predict_payload.py        <- generates the *_predict_payload.json files it consumes
  *_predict_payload.json            <- sample sensor windows (450 ACC samples each), one per "recording"
  kpi_results_<timestamp>.csv       <- output, one row per simulated device, written after each run
  logs/kpi_run_<timestamp>.log      <- full run log, written after each run
```

## What it does

Simulates N mobile-app devices, each with its own `patient_id`/`device_id`, running
concurrently instead of pressing "Simulate (real POST)" by hand N times. Each
device:

1. **KPI1** — POSTs to `/predict` on the real inference server
   (`fall-api.smarko-health.de`), timing the round trip.
2. **KPI2** — publishes `fall/alert/<patient_id>` to the real MQTT broker
   (`fall-mqtt.smarko-health.de`, WSS), waits for the caregiver dashboard's ack
   on `fall/ack/<patient_id>`, and times that round trip.

Both request shapes, headers, and the MQTT topic/payload/ack contract mirror
the real mobile app (`6g-path-app`) exactly — this hits the same production
inference server and MQTT broker the real app uses, so results show up on the
live caregiver dashboard like a real (simulated) fall.

## Prerequisites

```bash
pip install requests paho-mqtt
```

You need at least one `*_predict_payload.json` file in this folder. If you
don't have any yet, generate one from a raw sensor CSV:

```bash
python extract_predict_payload.py <path_to_raw_csv>
```

## Basic run

```bash
cd KPI_calculation
python kpi_multi_device_simulator.py --num-devices 8
```

Runs 8 simulated devices (patient IDs `sim-9001` … `sim-9008` by default, see
`FAKE_DEVICES` below), cycling through whatever `*_predict_payload.json` files
exist in this folder. Prints a summary table, and writes both a CSV and a log
file (see below).

## Patient IDs and MAC addresses — `FAKE_DEVICES`

Patient IDs and device (MAC) IDs are **not** generated programmatically —
they come from a plain dict near the top of `kpi_multi_device_simulator.py`:

```python
FAKE_DEVICES = {
    "sim-9001": "6c:1d:eb:04:a9:01",
    "sim-9002": "6c:1d:eb:04:a9:02",
    ...
    "sim-9008": "6c:1d:eb:04:a9:08",
}
```

`--num-devices N` takes the first N entries from this dict, in order. Edit
the dict directly (in your editor, not via a CLI flag) if you need:

- specific patient IDs, e.g. to match patients already registered in the
  caregiver dashboard's `PATIENT_IDS`/`MAC_IDS` env vars
- specific/realistic MAC addresses
- more than 8 simulated devices — just add more `"patient_id": "mac"` lines

If `--num-devices` asks for more than the dict has entries, the script exits
with a clear error telling you to add more pairs first — it will not silently
invent extra patient IDs.

## Configuring the run

All other options are CLI flags — run `python kpi_multi_device_simulator.py --help`
for the full list. The ones you'll actually touch:

| Flag | Default | What it controls |
|---|---|---|
| `--num-devices N` | `8` | How many simulated devices/patients to run — takes the first N entries from `FAKE_DEVICES` |
| `--payloads file1.json file2.json ...` | all `*_predict_payload.json` in this folder | Which sample recordings to cycle through across devices |
| `--scenario {sequential,concurrent}` | `concurrent` | **Timing control** — see below |
| `--stagger-s N` | `0` | Only used with `--scenario concurrent` — see below |
| `--confirmation-delay-s N` | `0` | Simulate the patient-confirmation popup wait (real app waits up to ~20s) before publishing the MQTT alert. `0` = publish immediately, best for a pure KPI2 (MQTT) measurement |
| `--skip-alert` | off | Only run KPI1 (`/predict`); skip the MQTT alert/ack step entirely |
| `--patient-confirmed {yes,no,not_answered}` | `yes` | What patient-confirmation value to put in the alert payload |
| `--needs-help` / `--no-needs-help` | `--needs-help` | Whether the alert requests caregiver help |
| `--output path.csv` | `kpi_results_<timestamp>.csv` in this folder | Where to write the CSV table |
| `--log-dir path/` | `KPI_calculation/logs/` | Where to write the run log |

### Scenario 1 (one after another) vs. scenario 2 (all at once) — `--scenario`

`--scenario` picks which of the two test patterns to run:

- **`--scenario sequential`** (scenario 1): devices run one after another —
  device 1 fully completes (predict, publish alert, wait for ack) before
  device 2 even starts its `/predict` call. `--stagger-s` has no effect here
  (there's nothing to stagger; it's already fully serial).
- **`--scenario concurrent`** (scenario 2, default): all devices run at the
  same time via a thread pool. Combine with `--stagger-s` to control exactly
  how "at the same time" that is:
  - `--stagger-s 0` (default): **all devices start at exactly the same
    instant** — every device's `/predict` request fires simultaneously.
  - `--stagger-s 2`: device *i* waits `i * 2` seconds before starting. So with
    8 devices, device 1 starts immediately, device 2 after 2s, device 3 after
    4s, ... device 8 after 14s — a realistic staggered burst instead of a
    single simultaneous spike.

Examples:

```bash
# Scenario 1: one device at a time, fully sequential
python kpi_multi_device_simulator.py --num-devices 8 --scenario sequential

# Scenario 2: all 8 devices fire at exactly the same instant
python kpi_multi_device_simulator.py --num-devices 8 --scenario concurrent --stagger-s 0

# Scenario 2, spread out: devices start 3 seconds apart from each other
python kpi_multi_device_simulator.py --num-devices 8 --scenario concurrent --stagger-s 3

# KPI1 only (no MQTT alert), useful for a quick /predict-only load test
python kpi_multi_device_simulator.py --num-devices 8 --skip-alert
```

## Output

### 1. CSV table (for your KPI report)

`kpi_results_<timestamp>.csv`, one row per device:

```
patient_id, device_id, observation_id, fall_detected, confidence, model_version,
kpi1_predict_ms, kpi2_alert_rtt_ms, ack_received, error
```

This is the file to import into Excel/Sheets to build your KPI table.

Below the per-device rows, the same CSV also has a small stats block (after a
blank separator row):

```
metric,n,min,max,mean,var,std
kpi1_predict_ms,8,139.3,261.3,198.4,1450.2,38.1
kpi2_alert_rtt_ms,8,19.7,28.2,23.5,9.8,3.1
```

`n` is how many devices actually produced a value for that KPI (a device that
errored out or never got an ack is excluded, not counted as 0). `var`/`std`
are sample variance/standard deviation (N-1); with only 1 successful sample
they're reported as 0. The same numbers are also printed to the console and
the log file as `KPI1 predict: N=... min=... max=... mean=... var=... std=...`.

### 2. Log file (full run detail)

`logs/kpi_run_<timestamp>.log` — every run automatically writes a timestamped
log file (in addition to printing the same lines to the console), containing:

- The device list and payload assignment at start
- A `[start]` line per device (with its actual delay applied)
- A `[done]` line per device the moment its KPI1/KPI2 numbers are ready
  (in completion order, not start order — useful for confirming the stagger
  actually happened)
- The final summary table
- The paths of the CSV and log file written

Nothing is silently dropped — if a device's `/predict` call or MQTT
publish/ack fails, the `error` field is filled in on both the console output,
the log file, and the CSV row (so a failed device shows up in your table
instead of disappearing).

## A note on hitting production

This script sends real requests to the live inference server and live MQTT
broker on Hetzner — a `fall_detected: true` alert will appear on the actual
caregiver dashboard, same as a real (simulated) mobile app would. That's
intentional (it's what makes the KPI numbers meaningful), just be aware a test
run is visible to anyone watching the dashboard at the time.
