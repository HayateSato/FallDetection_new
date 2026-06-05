"""
InfluxDB fall-event writer — mock_app component.

Simulates what the real mobile app does after patient confirmation:
  writes one `fall_events` point to InfluxDB so the caregiver dashboard
  can query fall history.

InfluxDB schema (agreed by MCS, communicated to FOCUS DevOps):
  Measurement : fall_events
  Tags        : patient_id, device_id
  Fields      : fall_detected      (bool)
                patient_confirmed  (int)   1 = confirmed, 0 = denied, -1 = not answered
                needs_help         (bool)
                confidence         (float)
  Timestamp   : detection_time of the fall event
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

INFLUXDB_FALL_EVENTS_BUCKET = os.getenv("INFLUXDB_FALL_EVENTS_BUCKET") or os.getenv("INFLUXDB_BUCKET", "")

_CONFIRMED_TO_INT = {"yes": 1, "no": 0, "not_answered": -1}


def inject_fall_marker(
    patient_id:        str,
    observation_id:    str,
    patient_confirmed: str,
    needs_help:        bool,
    confidence:        float,
    model_version:     str,
    detection_time:    Optional[datetime] = None,
    device_id:         Optional[str] = None,
) -> None:
    """
    Write one fall_events point to InfluxDB.

    patient_confirmed is stored as an integer:
      1  = patient confirmed fall (yes)
      0  = patient denied fall (no)
     -1  = no response within timeout (not_answered)
    needs_help is stored as a bool.
    """
    try:
        from influxdb_client import Point
        from influxdb_client.client.write_api import SYNCHRONOUS
        from ml_pipeline.data_input.data_loader.influx_client_manager import _get_influxdb_client
    except ImportError as exc:
        logger.warning(f"InfluxDB client not available — skipping fall marker write: {exc}")
        return

    bucket = INFLUXDB_FALL_EVENTS_BUCKET
    if not bucket:
        logger.warning("INFLUXDB_BUCKET not configured — skipping fall marker write")
        return

    ts = detection_time or datetime.now(timezone.utc)
    confirmed_int = _CONFIRMED_TO_INT.get(patient_confirmed, -1)

    point = (
        Point("fall_events")
        .tag("patient_id", patient_id)
        .tag("device_id", device_id or "")
        .field("fall_detected",     True)
        .field("patient_confirmed", confirmed_int)
        .field("needs_help",        bool(needs_help))
        .field("confidence",        round(float(confidence), 4))
        .time(ts, "ns")
    )

    try:
        client    = _get_influxdb_client()
        write_api = client.write_api(write_options=SYNCHRONOUS)
        write_api.write(bucket=bucket, record=point)
        logger.info(
            f"InfluxDB fall_events write OK  "
            f"patient={patient_id}  confirmed={patient_confirmed}({confirmed_int})  "
            f"needs_help={needs_help}"
        )
    except Exception as exc:
        logger.warning(f"InfluxDB fall_events write failed (non-fatal): {exc}")
