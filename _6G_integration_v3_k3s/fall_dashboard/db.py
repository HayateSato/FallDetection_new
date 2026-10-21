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

import atexit
import logging
import os
from typing import List, Optional

from influxdb_client import InfluxDBClient

from fall_dashboard import patient_store

logger = logging.getLogger(__name__)

_FALL_EVENTS_BUCKET = os.getenv("INFLUXDB_FALL_EVENTS_BUCKET") or os.getenv("INFLUXDB_BUCKET", "")

_influxdb_client_instance = None


def _get_influxdb_client() -> InfluxDBClient:
    global _influxdb_client_instance
    if _influxdb_client_instance is None:
        _influxdb_client_instance = InfluxDBClient(
            url=os.getenv("INFLUXDB_URL", ""),
            token=os.getenv("INFLUXDB_TOKEN", ""),
            org=os.getenv("INFLUXDB_ORG", ""),
            timeout=30_000,
            verify_ssl=True,
        )
        atexit.register(_influxdb_client_instance.close)
    return _influxdb_client_instance


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def _get_fall_counts(hours: int = 720) -> dict:
    """Return {patient_id: fall_count} from InfluxDB for the given time window.

    Counts rows where fall_detected == true. Using fall_detected (not observation_id)
    as the anchor field makes this robust when the mobile app omits observation_id.
    Default window is 720h (30 days) — used for per-card badges.
    Pass hours=24 for the "Falls today" header stat.
    """
    bucket = _FALL_EVENTS_BUCKET
    if not bucket:
        return {}
    try:
        query = f'''from(bucket: "{bucket}")
  |> range(start: -{hours}h)
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
    """
    Return one dict per patient from the SQLite patient store, with fall counts
    joined in from InfluxDB.

    The patient list is read live from SQLite on every call (no module-level
    caching), so patients added via sync_from_env() on container recreate show
    up immediately.
    """
    fall_counts_30d = _get_fall_counts(hours=720)
    fall_counts_today = _get_fall_counts(hours=24)
    patients = patient_store.list_patients()
    for p in patients:
        p["fall_count"] = fall_counts_30d.get(p["patient_id"], 0)
        p["fall_count_today"] = fall_counts_today.get(p["patient_id"], 0)
    return patients


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
  |> filter(fn: (r) => r["observation_id"] != "simulated-fall")
{only_falls_post}  |> unique(column: "observation_id")
  |> sort(columns: ["_time"], desc: true)
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
