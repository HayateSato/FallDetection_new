"""
Data export utilities for Fall Detection system.

Handles exporting detection data to CSV files and data conversion helpers.
"""
import logging
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path

from config.settings import (
    MODEL_VERSION,
    COLLECT_ADDITIONAL_SENSORS,
    FALL_DATA_EXPORT_DIR,
    TIMEZONE_OFFSET_HOURS,
)
# from config.hardware_config import ACC_SENSOR_SENSITIVITY
from app.utils import shared_state

logger = logging.getLogger(__name__)


def compose_detection_window(df: pd.DataFrame, required_samples: int,
                   pressure: np.ndarray = None,
                   pressure_time: np.ndarray = None):
    """Extract detection window from data."""
    if len(df) < required_samples:
        raise ValueError(f"Insufficient ACC data: need {required_samples}, got {len(df)}")

    window_df = df.tail(required_samples).copy().reset_index(drop=True)

    windowed_pressure = None
    windowed_pressure_time = None

    if pressure is not None and len(pressure) > 0:
        window_start_ms = window_df['Device_Timestamp_[ms]'].iloc[0]
        window_end_ms = window_df['Device_Timestamp_[ms]'].iloc[-1]
        mask = (pressure_time >= window_start_ms) & (pressure_time <= window_end_ms)
        windowed_pressure = pressure[mask]
        windowed_pressure_time = pressure_time[mask]

    return window_df, windowed_pressure, windowed_pressure_time


def save_detection_window_to_csv(flux_records, is_fall: bool, confidence: float,
                          participant_name: str, participant_gender: str,
                          manual_truth_fall: int, timestamp_utc):
    """Export detection data to CSV."""
    try:
        timestamp_local = timestamp_utc + timedelta(hours=TIMEZONE_OFFSET_HOURS)
        sensor_mode = "with_additional_sensors" if COLLECT_ADDITIONAL_SENSORS else "basic_sensors"

        base_dir = Path(FALL_DATA_EXPORT_DIR) / sensor_mode / participant_name
        base_dir.mkdir(parents=True, exist_ok=True)

        prefix = "fall" if is_fall else "no_fall"
        filename = f"{prefix}_{timestamp_local.strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = base_dir / filename

        records_list = []
        for record in flux_records:
            records_list.append({
                'time': record.get_time(),
                'field': record.get_field(),
                'value': record.get_value(),
                'measurement': record.get_measurement() if hasattr(record, 'get_measurement') else 'imu',
                'prediction': 1 if is_fall else 0
            })

        df = pd.DataFrame(records_list)

        metadata_df = pd.DataFrame([{
            'detection_timestamp_utc': timestamp_utc.isoformat(),
            'detection_timestamp_local': timestamp_local.isoformat(),
            'timezone_offset_hours': TIMEZONE_OFFSET_HOURS,
            'confidence': confidence,
            'fall_detected': is_fall,
            'manual_truth_marker': manual_truth_fall,
            'user_feedback': -1,  # -1 = pending (will be updated by /fall_feedback)
            'participant_name': participant_name,
            'participant_gender': participant_gender,
            'model_type': MODEL_VERSION
        }])

        with open(filepath, 'w', newline='') as f:
            f.write("# Detection Metadata\n")
            metadata_df.to_csv(f, index=False)
            f.write("\n# Sensor Data\n")
            df.to_csv(f, index=False)

        # Track last exported CSV so /fall_feedback can retroactively update it
        with shared_state.csv_path_lock:
            shared_state.last_exported_csv_path = str(filepath)

        logger.info(f"Detection data exported to: {filepath}")

    except Exception as e:
        logger.error(f"Error exporting detection data: {e}", exc_info=True)
