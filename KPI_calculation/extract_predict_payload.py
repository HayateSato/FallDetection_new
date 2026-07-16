"""
Converts a raw InfluxDB-export CSV (long format: time, field, value, ...) into
a clean JSON payload matching the inference server's /predict request schema
(_6G_integration_v3_docker_mcs/inference_server/server.py: PredictRequest).

Usage:
    python extract_predict_payload.py <input_csv> [output_json]

Defaults to both_peak_20260113_161505.csv in this folder if no input given.
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ACC_FIELDS = {"bosch_acc_x": "acc_x", "bosch_acc_y": "acc_y", "bosch_acc_z": "acc_z"}
PRESSURE_FIELD = "pressure_in_pa"   # already in Pa -- matches what /predict expects

# server.py requires WINDOW_SIZE_SECONDS * HARDWARE_ACC_SAMPLE_RATE = 9s * 50Hz raw
# samples. Real recordings land on 448-449 due to sensor timing jitter, so pad up
# to this count by repeating the last sample -- fine for KPI round-trip testing,
# not meant to preserve model accuracy.
REQUIRED_ACC_SAMPLES = 450
SAMPLE_STEP_MS = 1000.0 / 50  # 20ms at 50Hz


def _to_epoch_ms(ts: str) -> float:
    # e.g. "2026-01-13 14:14:52.269000+00:00"
    return datetime.fromisoformat(ts.strip()).timestamp() * 1000.0


def load_sensor_rows(csv_path: Path):
    """Skip the '# Detection Metadata' block and read only the '# Sensor Data' section."""
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "# Sensor Data")
    header_idx = start + 1  # the "time,field,value,..." header line right after
    reader = csv.DictReader(lines[header_idx:])
    return list(reader)


def build_payload(rows, patient_id="test-patient", device_id="test-device"):
    series = {}  # field -> list of (epoch_ms, value)
    for row in rows:
        field = row["field"]
        if field not in ACC_FIELDS and field != PRESSURE_FIELD:
            continue
        series.setdefault(field, []).append(
            (_to_epoch_ms(row["time"]), float(row["value"]))
        )

    payload = {
        "patient_id": patient_id,
        "device_id": device_id,
    }

    for raw_field, out_key in ACC_FIELDS.items():
        pairs = sorted(series.get(raw_field, []))
        if not pairs:
            raise ValueError(f"No rows found for required field '{raw_field}'")
        payload[out_key] = [v for _, v in pairs]
        # all three axes share timestamps -- write once
        payload["timestamps_ms"] = [t for t, _ in pairs]

    pressure_pairs = sorted(series.get(PRESSURE_FIELD, []))
    if pressure_pairs:
        payload["pressure"] = [v for _, v in pressure_pairs]
        payload["pressure_timestamps_ms"] = [t for t, _ in pressure_pairs]

    pad_to_required(payload)
    return payload


def pad_to_required(payload: dict) -> None:
    """Top up acc_x/acc_y/acc_z/timestamps_ms to REQUIRED_ACC_SAMPLES by repeating
    the last sample, if the recording came in short (sensor timing jitter)."""
    n = len(payload["acc_x"])
    missing = REQUIRED_ACC_SAMPLES - n
    if missing <= 0:
        return

    last_ts = payload["timestamps_ms"][-1]
    for i in range(1, missing + 1):
        payload["timestamps_ms"].append(last_ts + i * SAMPLE_STEP_MS)
        for key in ("acc_x", "acc_y", "acc_z"):
            payload[key].append(payload[key][-1])


def main():
    here = Path(__file__).parent
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "both_peak_20260113_161505.csv"
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else here / f"{in_path.stem}_predict_payload.json"

    rows = load_sensor_rows(in_path)
    raw_count = sum(1 for r in rows if r["field"] in ACC_FIELDS)
    raw_count //= len(ACC_FIELDS)  # rows split evenly across the 3 axes
    payload = build_payload(rows)

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    n = len(payload["acc_x"])
    duration_s = (payload["timestamps_ms"][-1] - payload["timestamps_ms"][0]) / 1000.0
    print(f"Wrote {out_path}")
    print(f"  ACC samples: {n}  (~{duration_s:.2f}s window)")
    if n > raw_count:
        print(f"  Padded {n - raw_count} sample(s) (recording had {raw_count}, "
              f"server requires {REQUIRED_ACC_SAMPLES}) by repeating the last sample.")
    if "pressure" in payload:
        print(f"  Pressure samples: {len(payload['pressure'])}")
    else:
        print("  No pressure data found (ACC-only model OK, v3 needs barometer).")


if __name__ == "__main__":
    main()
