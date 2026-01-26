"""
Ground Truth Writer for InfluxDB

Writes ground truth annotations directly to InfluxDB as a separate field.
When user presses the ground truth button, value '1' is written at that exact timestamp.
"""

import logging
from datetime import datetime, timezone
from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS
from app.data_fetcher import _get_influxdb_client

logger = logging.getLogger(__name__)

from config import settings
INFLUXDB_BUCKET = settings.INFLUXDB_BUCKET


def write_ground_truth_marker(value: int, measurement: str = "SMART_DATA") -> bool:
    """
    Write a ground truth marker to InfluxDB at the current timestamp.
    Uses SMART_DATA measurement to appear at same level as sensor data.

    Args:
        value: Ground truth value (1 = fall event marker)
        measurement: Measurement name in InfluxDB (default: "SMART_DATA")

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get singleton InfluxDB client (reused connection to prevent port exhaustion)
        client = _get_influxdb_client()
        write_api = client.write_api(write_options=SYNCHRONOUS)

        # Create timestamp (current moment in UTC)
        timestamp = datetime.now(timezone.utc)

        # Create InfluxDB point with same measurement as sensor data
        # This ensures ground_truth appears as a field alongside bosch_acc_x, etc.
        point = Point(measurement) \
            .field("ground_truth", value) \
            .time(timestamp)

        # Write to InfluxDB
        write_api.write(bucket=INFLUXDB_BUCKET, record=point)

        logger.info(f"Ground truth marker written: value={value} at {timestamp.isoformat()}")
        return True

    except Exception as e:
        logger.error(f"Failed to write ground truth marker: {e}", exc_info=True)
        return False
    # Note: Don't close the client - it's a singleton that will be reused


def write_ground_truth_event(event_type: str, value: int = 1) -> bool:
    """
    Write a ground truth event marker to InfluxDB.

    Args:
        event_type: Type of event (e.g., "fall_start", "fall_end")
        value: Event value (default: 1)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get singleton InfluxDB client (reused connection to prevent port exhaustion)
        client = _get_influxdb_client()
        write_api = client.write_api(write_options=SYNCHRONOUS)

        timestamp = datetime.now(timezone.utc)

        # Create point with event type as tag
        point = Point("ground_truth") \
            .tag("event_type", event_type) \
            .field("value", value) \
            .time(timestamp)

        write_api.write(bucket=INFLUXDB_BUCKET, record=point)

        logger.info(f"Ground truth event written: {event_type}={value} at {timestamp.isoformat()}")
        return True

    except Exception as e:
        logger.error(f"Failed to write ground truth event: {e}", exc_info=True)
        return False
    # Note: Don't close the client - it's a singleton that will be reused
