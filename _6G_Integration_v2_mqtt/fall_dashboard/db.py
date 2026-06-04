"""
Fall Dashboard database layer.
==============================

No Postgres in the caregiver layer. The patient list comes from the PATIENT_IDS
env var; fall history and fall counts come from InfluxDB.

InfluxDB schema (measurement: fall_events):
  Tags  : patient_id, device_id
  Fields: fall_detected (bool), patient_confirmed (int: 1/0/-1), needs_help (bool),
          observation_id (str), confidence (float), model_version (str)
  Time  : fall detection time

patient_confirmed int encoding:
   1  = patient confirmed it was a fall  ('yes')
   0  = patient denied (false positive)  ('no')
  -1  = no response within timeout       ('not_answered')
"""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

_FALL_EVENTS_BUCKET = os.getenv("INFLUXDB_FALL_EVENTS_BUCKET") or os.getenv("INFLUXDB_BUCKET", "")
_PATIENT_IDS = [p.strip() for p in os.getenv("PATIENT_IDS", "").split(",") if p.strip()]


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def _get_fall_counts() -> dict:
    """Return {patient_id: fall_count} from InfluxDB. Returns {} on any error."""
    bucket = _FALL_EVENTS_BUCKET
    if not bucket:
        return {}
    try:
        from ml_pipeline.data_input.data_loader.influx_client_manager import _get_influxdb_client
        query = f'''from(bucket: "{bucket}")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "fall_events")
  |> filter(fn: (r) => r["_field"] == "fall_detected")
  |> filter(fn: (r) => r["_value"] == true)
  |> group(columns: ["patient_id"])
  |> count()
'''
        tables = _get_influxdb_client().query_api().query(query)
        return {
            record.values.get("patient_id", ""): int(record.get_value())
            for table in tables
            for record in table.records
            if record.values.get("patient_id")
        }
    except Exception as exc:
        logger.warning(f"InfluxDB fall count query failed (non-fatal): {exc}")
        return {}


def list_patients() -> List[dict]:
    """Return one dict per patient from PATIENT_IDS env var. Fall counts from InfluxDB."""
    fall_counts = _get_fall_counts()
    return [
        {
            "patient_id": pid,
            "fall_count": fall_counts.get(pid, 0),
        }
        for pid in _PATIENT_IDS
    ]


def list_falls(
    patient_id: Optional[str] = None,
    only_falls: bool = True,
    limit: int = 200,
    hours: int = 720,
) -> List[dict]:
    """
    Return fall history by querying the `fall_events` measurement in InfluxDB.

    patient_confirmed is an int: 1 = confirmed, 0 = denied, -1 = not answered.

    hours : how far back to look (default 720 = 30 days). Use hours=24 for
            the patient detail view.
    """
    bucket = _FALL_EVENTS_BUCKET
    if not bucket:
        logger.warning("INFLUXDB_BUCKET not configured — list_falls() returning empty")
        return []

    try:
        from ml_pipeline.data_input.data_loader.influx_client_manager import _get_influxdb_client
    except ImportError:
        logger.warning("InfluxDB client not available — list_falls() returning empty")
        return []

    # patient_id is a TAG so it can be filtered before pivot
    patient_filter = (
        f'  |> filter(fn: (r) => r["patient_id"] == "{patient_id}")\n'
        if patient_id else ""
    )
    # fall_detected is a FIELD — must filter AFTER pivot
    only_falls_post = (
        '  |> filter(fn: (r) => r["fall_detected"] == true)\n'
        if only_falls else ""
    )

    query = f'''from(bucket: "{bucket}")
  |> range(start: -{hours}h)
  |> filter(fn: (r) => r["_measurement"] == "fall_events")
{patient_filter}  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
{only_falls_post}  |> sort(columns: ["_time"], desc: true)
  |> limit(n: {limit})
'''

    try:
        client = _get_influxdb_client()
        tables = client.query_api().query(query)
    except Exception as exc:
        logger.error(f"InfluxDB list_falls query failed: {exc}")
        return []

    results = []
    for i, table in enumerate(tables):
        for j, record in enumerate(table.records):
            v = record.values
            results.append({
                "id":                i * 10000 + j,
                "patient_id":        v.get("patient_id") or "",
                "fall_detected":     bool(v.get("fall_detected", True)),
                "patient_confirmed": v.get("patient_confirmed"),  # int: 1/0/-1
                "needs_help":        v.get("needs_help"),
                "confidence":        v.get("confidence"),
                "detection_time":    record.get_time().isoformat() if record.get_time() else None,
            })

    return results
