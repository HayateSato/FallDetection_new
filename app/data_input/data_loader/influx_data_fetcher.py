"""

InfluxDB Data Fetcher - Singleton Client
Provides a persistent connection to prevent port exhaustion on Windows.

AND

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
from typing import Tuple
import logging


from app.data_input.data_loader.influx_client_manager import _get_influxdb_client
from app.data_input.accelerometer_processor.acc_resampler import AccelerometerResampler
from app.data_input.accelerometer_processor.nonbosch_calibration import transform_acc_array as calibrate_non_bosch_to_bosch
from app.data_input.data_converter import convert_acc_from_flux_to_numpy_array, convert_baro_from_flux_to_numpy_array
from config.settings import (
    INFLUXDB_URL,
    INFLUXDB_ORG,
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

logger = logging.getLogger(__name__)

def fetch_data(query):
    """
    Fetches data from InfluxDB using a persistent connection.

    This function reuses the same InfluxDB client to avoid connection exhaustion
    issues on Windows (WinError 10048).
    """
    try:
        client = _get_influxdb_client()
        query_api = client.query_api()
        tables = query_api.query(query)
        logger.info(f"Query successful, received {len(tables)} tables")
        return tables
    except Exception as e:
        logger.error(f"InfluxDB connection error: {type(e).__name__}: {str(e)}")
        logger.error(f"URL: {INFLUXDB_URL}")
        logger.error(f"Org: {INFLUXDB_ORG}")
        logger.error(f"Bucket: {INFLUXDB_BUCKET}")

        # Provide specific troubleshooting hints
        error_msg = str(e).lower()
        if "timeout" in error_msg:
            logger.error("TROUBLESHOOTING: Connection timeout detected")
            logger.error("  Check if VPN is connected (if required)")
            logger.error("  Verify firewall settings")
        elif "ssl" in error_msg or "certificate" in error_msg:
            logger.error("TROUBLESHOOTING: SSL certificate issue")
            logger.error("  Try setting verify_ssl=False in data_fetcher.py")
        elif "unauthorized" in error_msg or "401" in error_msg:
            logger.error("TROUBLESHOOTING: Authentication failed")
            logger.error("  Check if token is valid and not expired")

        raise


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
    acc_data, acc_time = convert_acc_from_flux_to_numpy_array(
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
        pressure, pressure_time = convert_baro_from_flux_to_numpy_array(flux_records, BAROMETER_FIELD)

    return acc_data, acc_time, pressure, pressure_time, flux_records

