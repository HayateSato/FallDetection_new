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
from influxdb_client.client.flux_table import FluxRecord

from app.data_input.data_loader.data_fetcher import fetch_data
# from app.data_input.data_processor import preprocess_acc, preprocess_barometer
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


def preprocess_acc(
    records: List[FluxRecord],
    acc_field_x: str = 'bosch_acc_x',
    acc_field_y: str = 'bosch_acc_y',
    acc_field_z: str = 'bosch_acc_z'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess InfluxDB records for accelerometer-only models.

    Supports configurable field names for different hardware modes:
    - 50Hz mode: bosch_acc_x/y/z (default)
    - 100Hz mode: acc_x/y/z

    Args:
        records: List of FluxRecord objects from InfluxDB query
        acc_field_x: Field name for X axis accelerometer
        acc_field_y: Field name for Y axis accelerometer
        acc_field_z: Field name for Z axis accelerometer

    Returns:
        Tuple containing (acc_data_array, acc_time_array):
        - acc_data_array: numpy array shape (3, N) - accelerometer x/y/z data
        - acc_time_array: numpy array shape (N,) - accelerometer timestamps in ms
    """
    # Initialize data structures for accelerometer
    acc_data = {'x': [], 'y': [], 'z': []}
    acc_times = {'x': [], 'y': [], 'z': []}

    # Route each record to appropriate axis
    for record in records:
        field = record.get_field()
        value = record.get_value()
        time = record.get_time().timestamp() * 1000  # Convert to milliseconds

        # Accelerometer data routing based on configurable field names
        if field == acc_field_x:
            acc_data['x'].append(value)
            acc_times['x'].append(time)
        elif field == acc_field_y:
            acc_data['y'].append(value)
            acc_times['y'].append(time)
        elif field == acc_field_z:
            acc_data['z'].append(value)
            acc_times['z'].append(time)

    # Convert to numpy arrays
    # Shape: (3, N) where 3 = [x, y, z] axes and N = number of samples
    acc_data_array = np.array([acc_data['x'], acc_data['y'], acc_data['z']])

    # Use x-axis timestamps as reference
    acc_time_array = np.array(acc_times['x'])

    return acc_data_array, acc_time_array


def preprocess_barometer(
    records: List[FluxRecord],
    barometer_field: str = 'bmp_pressure'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract barometer data from InfluxDB records.

    Args:
        records: List of FluxRecord objects from InfluxDB query
        barometer_field: Field name for barometer pressure data

    Returns:
        Tuple containing (pressure_array, time_array):
        - pressure_array: numpy array shape (N,) - pressure values in Pa
        - time_array: numpy array shape (N,) - timestamps in ms
    """
    pressure_data = []
    pressure_times = []

    for record in records:
        field = record.get_field()
        if field == barometer_field:
            pressure_data.append(record.get_value())
            pressure_times.append(record.get_time().timestamp() * 1000)

    return np.array(pressure_data), np.array(pressure_times)
