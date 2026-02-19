"""
Shared sensor data pipeline for fall detection.

Encapsulates the full pipeline from InfluxDB query to processed numpy arrays:
  - Query building (ACC + optional BARO + annotation fields)
  - Data fetching via InfluxDB client
  - Format conversion (FluxRecord -> numpy arrays)
  - Resampling (if hardware rate differs from model rate)
  - Sensor calibration (if non-Bosch hardware is used)

Used by both the /trigger route (detection.py) and ContinuousMonitor
(continuous_monitoring.py) to avoid duplicating the same ~50-line pipeline.
"""

import numpy as np
from typing import Optional, Tuple, List

from app.data_input.data_loader.data_fetcher import fetch_data
from app.data_input.data_processor import preprocess_acc, preprocess_barometer
from app.data_input.accelerometer_processor.acc_resampler import AccelerometerResampler
from app.data_input.accelerometer_processor.nonbosch_calibration import transform_acc_array as calibrate_non_bosch_to_bosch

from config.settings import (
    INFLUXDB_BUCKET,
    BAROMETER_FIELD,
    ACC_FIELD_X,
    ACC_FIELD_Y,
    ACC_FIELD_Z,
    HARDWARE_ACC_SAMPLE_RATE,
    MODEL_ACC_SAMPLE_RATE,
    RESAMPLING_ENABLED,
    RESAMPLING_METHOD,
    BAROMETER_ENABLED,
    SENSOR_CALIBRATION_ENABLED,
)


def fetch_and_preprocess_sensor_data(
    uses_barometer: bool,
    lookback_seconds: int,
) -> Tuple:
    """
    Build InfluxDB query, fetch data, convert to numpy, resample, and calibrate.

    Args:
        uses_barometer: Whether the current model requires barometer data.
        lookback_seconds: How far back in time to query (e.g. 30 for last 30s).

    Returns:
        (acc_data, acc_time, pressure, pressure_time, flux_records)
        - acc_data:      numpy (3, N) - raw accelerometer values (int16 units)
        - acc_time:      numpy (N,)   - timestamps in milliseconds
        - pressure:      numpy (M,)   - pressure in Pa (empty array if barometer not used)
        - pressure_time: numpy (M,)   - barometer timestamps in ms
        - flux_records:  list[FluxRecord] - full record list for CSV export

        Returns (None, None, None, None, None) if no data is available.
    """
    # Build field filter
    fields_filter = (
        f'r["_field"] == "{ACC_FIELD_X}" or '
        f'r["_field"] == "{ACC_FIELD_Y}" or '
        f'r["_field"] == "{ACC_FIELD_Z}"'
    )
    if uses_barometer and BAROMETER_ENABLED:
        fields_filter += f' or r["_field"] == "{BAROMETER_FIELD}"'

    # Always include annotation fields so they appear in exported CSVs
    fields_filter += ' or r["_field"] == "manual_truth_marker" or r["_field"] == "user_feedback"'

    query = f'''from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -{lookback_seconds}s)
      |> filter(fn: (r) => {fields_filter})
    '''

    tables = fetch_data(query)
    if tables is None:
        return None, None, None, None, None

    flux_records = [record for table in tables for record in table.records]
    if not flux_records:
        return None, None, None, None, None

    # Convert FluxRecord objects to numpy arrays
    acc_data, acc_time = preprocess_acc(
        flux_records,
        acc_field_x=ACC_FIELD_X,
        acc_field_y=ACC_FIELD_Y,
        acc_field_z=ACC_FIELD_Z,
    )

    if acc_data is None or acc_data.shape[1] == 0:
        return None, None, None, None, None

    # Resample if hardware rate differs from the model's expected rate
    if RESAMPLING_ENABLED:
        resampler = AccelerometerResampler(
            source_rate=HARDWARE_ACC_SAMPLE_RATE,
            target_rate=MODEL_ACC_SAMPLE_RATE,
            method=RESAMPLING_METHOD,
        )
        acc_data, acc_time = resampler.process(acc_data, acc_time)

    # Calibrate if using non-Bosch sensor hardware
    if SENSOR_CALIBRATION_ENABLED:
        acc_data = calibrate_non_bosch_to_bosch(acc_data)

    # Extract barometer data if the model uses it
    pressure, pressure_time = np.array([]), np.array([])
    if uses_barometer and BAROMETER_ENABLED:
        pressure, pressure_time = preprocess_barometer(flux_records, BAROMETER_FIELD)

    return acc_data, acc_time, pressure, pressure_time, flux_records
