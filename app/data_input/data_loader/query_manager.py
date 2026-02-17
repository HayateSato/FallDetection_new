"""
Query Manager for InfluxDB
Builds and executes Flux queries for sensor data retrieval.
"""

import logging
from app.data_input.data_loader.data_fetcher import fetch_data
from config import settings

logger = logging.getLogger(__name__)

INFLUXDB_BUCKET = settings.INFLUXDB_BUCKET


def execute_query(collect_additional_sensors: bool, lookback_seconds: int):
    """
    Execute InfluxDB query for sensor data.

    Args:
        collect_additional_sensors: If True, fetch all sensors; if False, only IMU
        lookback_seconds: How many seconds of data to fetch

    Returns:
        List of InfluxDB tables with sensor data
    """
    try:
        if collect_additional_sensors:
            logger.debug("Collecting all sensor data (IMU + additional sensors)...")
            query = f'''from(bucket: "{INFLUXDB_BUCKET}")
              |> range(start: -{lookback_seconds}s)
              |> filter(fn: (r) => r["_field"] == "bosch_acc_x" or r["_field"] == "bosch_acc_z" or r["_field"] == "bosch_acc_y" or
                                   r["_field"] == "bosch_gyr_x" or r["_field"] == "bosch_gyr_y" or r["_field"] == "bosch_gyr_z" or
                                   r["_field"] == "skin_temperature" or
                                   r["_field"] == "green" or r["_field"] == "infra_red" or
                                   r["_field"] == "hr" or r["_field"] == "hr_conf" or
                                   r["_field"] == "pressure" or r["_field"] == "pressure_in_pa" or r["_field"] == "red" or
                                   r["_field"] == "manual_truth_marker" or r["_field"] == "user_feedback")
            '''
        else:
            logger.debug("Collecting IMU sensor data only...")
            query = f'''from(bucket: "{INFLUXDB_BUCKET}")
                      |> range(start: -{lookback_seconds}s)
                      |> filter(fn: (r) => r["_field"] == "bosch_gyr_y" or
                                           r["_field"] == "bosch_gyr_z" or
                                           r["_field"] == "bosch_acc_x" or
                                           r["_field"] == "bosch_acc_y" or
                                           r["_field"] == "bosch_acc_z" or
                                           r["_field"] == "bosch_gyr_x" or
                                           r["_field"] == "manual_truth_marker" or r["_field"] == "user_feedback")
            '''
        tables = fetch_data(query)
        return tables

    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise
