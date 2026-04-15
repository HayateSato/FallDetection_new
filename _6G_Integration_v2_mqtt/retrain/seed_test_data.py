"""
Seed Postgres with labelled test data to enable testing the retraining pipeline.

WHY this script exists
----------------------
Charite patient data cannot be used until a data sharing agreement is signed.
But the retraining pipeline (retrain.py) needs rows in Postgres to run.
This script fills that gap by generating synthetic data or by processing
real data from OUR OWN InfluxDB instance (fd_test bucket).

The seeder inserts directly into Postgres — it does NOT call the inference
server over HTTP. This lets you test the full pipeline without running the
inference server.

Two modes
---------
  --synthetic N
        Generate N synthetic windows with realistic feature distributions.
        Fast, no external dependencies. Best for pipeline testing.

        Example:
          python -m retrain.seed_test_data --synthetic 100

  --influxdb
        Fetch real windows from our InfluxDB (reads .env for credentials),
        run the same preprocessing pipeline as server.py, store in Postgres.
        Produces realistic features but requires InfluxDB to be accessible.

        Example:
          python -m retrain.seed_test_data --influxdb --lookback-hours 24

Data inserted per window
------------------------
  inference_log      — one row (observation_id, patient_id, fall_detected, features, etc.)
  feature_snapshot   — N rows (one per feature)
  fall_history       — one row IF fall_detected=True (patient_confirmed='yes')
                       This creates the ground-truth label for the retraining JOIN.

After running this script, run:
  python -m retrain.retrain --dry-run   (check dataset stats)
  python -m retrain.retrain             (train + log to MLflow)
"""

import argparse
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Window parameters — must match what the inference server uses
WINDOW_SECONDS  = 9
SAMPLE_RATE_HZ  = 50
WINDOW_SAMPLES  = WINDOW_SECONDS * SAMPLE_RATE_HZ   # 450

# Ratio of fall events inserted by the synthetic seeder
SYNTHETIC_FALL_RATIO = 0.2   # 20% falls, 80% non-falls


# ---------------------------------------------------------------------------
# Synthetic data mode
# ---------------------------------------------------------------------------

def _synthetic_features_nonfal() -> dict:
    """Realistic feature values for a normal walking / standing window."""
    rng = np.random
    acc_mag_mean = rng.uniform(0.95, 1.05)   # near 1g
    return {
        "acc_x_min":   rng.uniform(-0.3, -0.1),
        "acc_x_max":   rng.uniform(0.1,  0.3),
        "acc_x_mean":  rng.uniform(-0.05, 0.05),
        "acc_x_var":   rng.uniform(0.001, 0.01),
        "acc_y_min":   rng.uniform(-0.3, -0.1),
        "acc_y_max":   rng.uniform(0.1,  0.3),
        "acc_y_mean":  rng.uniform(-0.05, 0.05),
        "acc_y_var":   rng.uniform(0.001, 0.01),
        "acc_z_min":   rng.uniform(0.8,  0.9),
        "acc_z_max":   rng.uniform(1.0,  1.1),
        "acc_z_mean":  rng.uniform(0.95, 1.0),
        "acc_z_var":   rng.uniform(0.001, 0.005),
        "acc_mag_min": rng.uniform(0.85, 0.95),
        "acc_mag_max": rng.uniform(1.05, 1.15),
        "acc_mag_mean": acc_mag_mean,
        "acc_mag_var": rng.uniform(0.001, 0.008),
        # v3 barometer features
        "pressure_shift":        rng.uniform(-0.5, 0.5),
        "middle_slope":          rng.uniform(-0.2, 0.2),
        "post_fall_slope":       rng.uniform(-0.2, 0.2),
        "filtered_pressure_var": rng.uniform(0.01, 0.5),
    }


def _synthetic_features_fall() -> dict:
    """Realistic feature values for a fall window (high impact, large variance)."""
    rng = np.random
    return {
        "acc_x_min":   rng.uniform(-4.0, -2.0),
        "acc_x_max":   rng.uniform(2.0,  5.0),
        "acc_x_mean":  rng.uniform(-0.5, 0.5),
        "acc_x_var":   rng.uniform(1.0,  4.0),
        "acc_y_min":   rng.uniform(-4.0, -2.0),
        "acc_y_max":   rng.uniform(2.0,  5.0),
        "acc_y_mean":  rng.uniform(-0.5, 0.5),
        "acc_y_var":   rng.uniform(1.0,  4.0),
        "acc_z_min":   rng.uniform(-2.0, -0.5),
        "acc_z_max":   rng.uniform(3.0,  6.0),
        "acc_z_mean":  rng.uniform(-0.2, 0.5),
        "acc_z_var":   rng.uniform(2.0,  6.0),
        "acc_mag_min": rng.uniform(0.1,  0.5),
        "acc_mag_max": rng.uniform(4.0,  8.0),
        "acc_mag_mean": rng.uniform(1.5, 3.0),
        "acc_mag_var": rng.uniform(2.0,  5.0),
        # v3 barometer features — larger shift during a fall
        "pressure_shift":        rng.uniform(-5.0, -1.0),
        "middle_slope":          rng.uniform(-3.0, -0.5),
        "post_fall_slope":       rng.uniform(0.2,  1.5),
        "filtered_pressure_var": rng.uniform(2.0,  8.0),
    }


def seed_synthetic(
    n_windows:    int,
    model_version: str,
    patient_ids:  List[str],
    fall_ratio:   float = SYNTHETIC_FALL_RATIO,
    verbose:      bool = True,
) -> None:
    """
    Insert n_windows synthetic inference rows into Postgres.

    fall_ratio controls the fraction that are labelled as falls.
    Each fall window also gets a fall_history row (patient_confirmed='yes').
    """
    from shared.db.session import SessionLocal, init_db
    from shared.db.models import InferenceLog, FeatureSnapshot, FallHistory

    init_db()
    db = SessionLocal()

    n_falls      = 0
    n_nonfalls   = 0
    base_time    = datetime.now(timezone.utc) - timedelta(hours=n_windows // 6)

    try:
        for i in range(n_windows):
            is_fall    = np.random.random() < fall_ratio
            patient_id = np.random.choice(patient_ids)
            obs_id     = str(uuid.uuid4())
            det_time   = base_time + timedelta(minutes=10 * i)
            confidence = np.random.uniform(0.65, 0.95) if is_fall else np.random.uniform(0.05, 0.35)

            features = _synthetic_features_fall() if is_fall else _synthetic_features_nonfal()

            # inference_log row
            log = InferenceLog(
                observation_id = obs_id,
                patient_id     = patient_id,
                device_id      = None,
                model_version  = model_version,
                fall_detected  = is_fall,
                confidence     = round(confidence, 4),
                window_size    = WINDOW_SAMPLES,
                latency_ms     = int(np.random.uniform(50, 200)),
                detection_time = det_time,
            )
            db.add(log)
            db.flush()   # get log.id for FK in feature_snapshot

            # feature_snapshot rows
            for name, value in features.items():
                db.add(FeatureSnapshot(
                    inference_id  = log.id,
                    feature_name  = name,
                    feature_value = float(value),
                ))

            # fall_history row — patient confirmed YES (creates the retraining label)
            if is_fall:
                db.add(FallHistory(
                    observation_id    = obs_id,
                    patient_id        = patient_id,
                    fall_detected     = True,
                    patient_confirmed = "yes",
                    needs_help        = bool(np.random.random() < 0.7),
                    detection_time    = det_time,
                ))
                n_falls += 1
            else:
                n_nonfalls += 1

        db.commit()

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    if verbose:
        print(f"\nSynthetic seed complete:")
        print(f"  Total windows  : {n_windows}")
        print(f"  Falls (label=1): {n_falls}  (patient_confirmed='yes' in fall_history)")
        print(f"  Non-falls (l=0): {n_nonfalls}")
        print(f"  Model version  : {model_version}")
        print(f"  Patients       : {patient_ids}")
        print(f"\nNext step:")
        print(f"  python -m retrain.retrain --model-version {model_version} --dry-run")


# ---------------------------------------------------------------------------
# InfluxDB mode
# ---------------------------------------------------------------------------

def seed_from_influxdb(
    model_version:   str,
    lookback_hours:  int,
    fall_confirm_threshold: float = 0.5,
    verbose:         bool = True,
) -> None:
    """
    Fetch real data from our InfluxDB instance, run feature extraction,
    insert into Postgres.

    The seeder batches the raw InfluxDB data into non-overlapping 9s windows
    and runs the same preprocessing as server.py for each window.

    A fall_history row (patient_confirmed='yes') is inserted for any window
    where the model's predicted confidence exceeds fall_confirm_threshold —
    simulating a patient confirming the fall. Adjust the threshold to
    control how many positive examples are seeded.
    """
    from shared.db.session import SessionLocal, init_db
    from shared.db.models import InferenceLog, FeatureSnapshot, FallHistory
    from config.settings import (
        ACC_SAMPLE_RATE, HARDWARE_ACC_SAMPLE_RATE, RESAMPLING_METHOD,
        ACC_SENSOR_TYPE,
    )
    from app.core.inference_engine import PipelineSelector
    from app.core.model_registry import get_model_config, get_model_name, get_model_path
    from app.data_input.data_converter import (
        convert_acc_nparray_to_df, convert_lsb_to_g, compose_detection_window,
    )
    from app.data_input.accelerometer_processor.acc_resampler import AccelerometerResampler
    from app.data_input.data_loader.influx_client_manager import _get_influxdb_client
    from app.data_input.data_converter import (
        convert_acc_from_flux_to_numpy_array, convert_baro_from_flux_to_numpy_array,
    )
    from config.settings import (
        ACC_FIELD_X, ACC_FIELD_Y, ACC_FIELD_Z, BAROMETER_FIELD,
        INFLUXDB_BUCKET, PATIENT_IDS, MAC_IDS,
    )

    model_type   = get_model_name(model_version)
    model_path   = get_model_path(model_type)
    model_config = get_model_config(model_type)
    engine       = PipelineSelector(model_version, model_path)
    uses_baro    = model_config.uses_barometer

    required = int(WINDOW_SECONDS * ACC_SAMPLE_RATE)
    lookback_seconds = lookback_hours * 3600

    logger.info(
        f"Fetching from InfluxDB  lookback={lookback_hours}h  "
        f"model={model_version}  uses_barometer={uses_baro}"
    )

    # Parse patient / MAC mapping
    patient_list = [p.strip() for p in PATIENT_IDS.split(",") if p.strip()]
    mac_list     = [m.strip() for m in (MAC_IDS or "").split(",") if m.strip()]
    mac_map      = dict(zip(patient_list, mac_list)) if mac_list else {}

    init_db()
    db = SessionLocal()
    n_inserted = 0
    n_falls    = 0

    try:
        client = _get_influxdb_client()
        fields_filter = (
            f'r["_field"] == "{ACC_FIELD_X}" or '
            f'r["_field"] == "{ACC_FIELD_Y}" or '
            f'r["_field"] == "{ACC_FIELD_Z}"'
        )
        if uses_baro:
            fields_filter += f' or r["_field"] == "{BAROMETER_FIELD}"'

        for patient_id in patient_list:
            mac = mac_map.get(patient_id)
            mac_filter = f'  |> filter(fn: (r) => r["macAddress"] == "{mac}")\n' if mac else ""

            query = f'''from(bucket: "{INFLUXDB_BUCKET}")
  |> range(start: -{lookback_seconds}s)
  |> filter(fn: (r) => r["_measurement"] == "SMART_DATA")
  |> filter(fn: (r) => {fields_filter})
{mac_filter}'''

            logger.info(f"Querying InfluxDB for patient={patient_id}  mac={mac}")
            try:
                tables  = client.query_api().query(query)
                records = [rec for table in tables for rec in table.records]
            except Exception as exc:
                logger.error(f"InfluxDB query failed: {exc}")
                continue

            if not records:
                logger.warning(f"No data returned for patient={patient_id}")
                continue

            acc_data, acc_time = convert_acc_from_flux_to_numpy_array(
                records, ACC_FIELD_X, ACC_FIELD_Y, ACC_FIELD_Z
            )
            pressure, pressure_time = None, None
            if uses_baro:
                pressure, pressure_time = convert_baro_from_flux_to_numpy_array(
                    records, BAROMETER_FIELD
                )

            if acc_data is None or acc_data.shape[1] == 0:
                logger.warning(f"No ACC data for patient={patient_id}")
                continue

            # Resample if needed
            if HARDWARE_ACC_SAMPLE_RATE != ACC_SAMPLE_RATE:
                resampler = AccelerometerResampler(
                    source_rate=HARDWARE_ACC_SAMPLE_RATE,
                    target_rate=ACC_SAMPLE_RATE,
                    method=RESAMPLING_METHOD,
                )
                acc_data, acc_time = resampler.process(acc_data, acc_time)

            # LSB → g
            if not model_config.acc_in_lsb:
                acc_data = convert_lsb_to_g(acc_data)

            df_full = convert_acc_nparray_to_df(acc_data, acc_time)

            # Slide non-overlapping 9s windows through the data
            step = required   # non-overlapping
            n_windows_found = (len(df_full) - required) // step + 1
            logger.info(f"  {len(df_full)} samples → {n_windows_found} windows "
                        f"(patient={patient_id})")

            for w_idx in range(n_windows_found):
                start_i = w_idx * step
                end_i   = start_i + required
                window_df = df_full.iloc[start_i:end_i].copy().reset_index(drop=True)

                # Extract matching pressure window
                w_pressure, w_pressure_time = None, None
                if pressure is not None and len(pressure) > 0:
                    w_start_ms = window_df["Device_Timestamp_[ms]"].iloc[0]
                    w_end_ms   = window_df["Device_Timestamp_[ms]"].iloc[-1]
                    mask = (pressure_time >= w_start_ms) & (pressure_time <= w_end_ms)
                    w_pressure      = pressure[mask]
                    w_pressure_time = pressure_time[mask]

                # Feature extraction + inference
                result     = engine.predict(window_df, w_pressure, w_pressure_time)
                is_fall    = result["is_fall"]
                confidence = result["confidence"]
                features   = result.get("features", {})
                obs_id     = str(uuid.uuid4())
                det_time   = datetime.now(timezone.utc) - timedelta(
                    seconds=(n_windows_found - w_idx) * WINDOW_SECONDS
                )

                log = InferenceLog(
                    observation_id = obs_id,
                    patient_id     = patient_id,
                    device_id      = mac,
                    model_version  = model_version,
                    fall_detected  = is_fall,
                    confidence     = round(float(confidence), 4),
                    window_size    = len(window_df),
                    latency_ms     = None,
                    detection_time = det_time,
                )
                db.add(log)
                db.flush()

                for fname, fval in features.items():
                    try:
                        fval_float = float(fval)
                    except (TypeError, ValueError):
                        fval_float = None
                    db.add(FeatureSnapshot(
                        inference_id  = log.id,
                        feature_name  = str(fname),
                        feature_value = fval_float,
                    ))

                # Create confirmed fall label if confidence exceeds threshold
                if is_fall and confidence >= fall_confirm_threshold:
                    db.add(FallHistory(
                        observation_id    = obs_id,
                        patient_id        = patient_id,
                        fall_detected     = True,
                        patient_confirmed = "yes",
                        needs_help        = True,
                        detection_time    = det_time,
                    ))
                    n_falls += 1

                n_inserted += 1

        db.commit()

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    if verbose:
        print(f"\nInfluxDB seed complete:")
        print(f"  Windows inserted   : {n_inserted}")
        print(f"  Fall windows (l=1) : {n_falls}  (confidence >= {fall_confirm_threshold})")
        print(f"  Non-fall (l=0)     : {n_inserted - n_falls}")
        print(f"\nNext step:")
        print(f"  python -m retrain.retrain --model-version {model_version} --dry-run")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Seed Postgres with test data for retraining pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", type=int, metavar="N",
                      help="Generate N synthetic windows (no InfluxDB needed)")
    mode.add_argument("--influxdb", action="store_true",
                      help="Fetch real data from InfluxDB (uses .env credentials)")

    parser.add_argument("--model-version", default="v3",
                        help="Model version for feature extraction (default: v3)")
    parser.add_argument("--patient-ids", default=None,
                        help="Comma-separated patient IDs (default: reads from .env PATIENT_IDS)")
    parser.add_argument("--lookback-hours", type=int, default=24,
                        help="InfluxDB lookback window in hours (default: 24)")
    parser.add_argument("--fall-threshold", type=float, default=0.5,
                        help="Confidence threshold for seeding fall labels in InfluxDB mode (default: 0.5)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = _parse_args()

    if args.synthetic:
        from config.settings import PATIENT_IDS as _DEFAULT_PATIENT_IDS
        patient_ids = (
            [p.strip() for p in args.patient_ids.split(",")]
            if args.patient_ids
            else [p.strip() for p in _DEFAULT_PATIENT_IDS.split(",") if p.strip()]
        ) or ["seed-patient-001"]

        seed_synthetic(
            n_windows     = args.synthetic,
            model_version = args.model_version,
            patient_ids   = patient_ids,
        )
    else:
        seed_from_influxdb(
            model_version          = args.model_version,
            lookback_hours         = args.lookback_hours,
            fall_confirm_threshold = args.fall_threshold,
        )
