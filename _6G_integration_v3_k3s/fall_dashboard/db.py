"""
Fall Dashboard database layer.
==============================

No Postgres in the caregiver layer. The patient list comes from the PATIENT_IDS
env var; fall history and fall counts come from InfluxDB.

fall_events points are written by the mobile app, and also by this dashboard
when FALL_EVENTS_WRITE=true (see write_fall_event).

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
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from fall_dashboard import patient_store

logger = logging.getLogger(__name__)

_FALL_EVENTS_BUCKET = os.getenv("INFLUXDB_FALL_EVENTS_BUCKET") or os.getenv("INFLUXDB_BUCKET", "")

# When true, the dashboard writes a fall_events point for every fall/alert it
# receives, so fall counts no longer depend on which InfluxDB the mobile app
# writes to. Off by default: in FOCUS production the mobile app writes
# fall_events to the same InfluxDB the dashboard reads.
FALL_EVENTS_WRITE = os.getenv("FALL_EVENTS_WRITE", "false").strip().lower() == "true"

_SIMULATED_OBSERVATION_ID = "simulated-fall"

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

def _get_fall_counts() -> Tuple[dict, dict]:
    """Return ({patient_id: count_30d}, {patient_id: count_24h}) from InfluxDB.

    Counts rows where fall_detected == true, one per fall: rows sharing an
    observation_id (e.g. written by both the mobile app and this dashboard)
    count once. Rows without an observation_id, or with the shared
    "simulated-fall" id, are counted per timestamp instead.
    """
    bucket = _FALL_EVENTS_BUCKET
    if not bucket:
        return {}, {}
    try:
        query = f'''from(bucket: "{bucket}")
  |> range(start: -720h)
  |> filter(fn: (r) => r["_measurement"] == "fall_events")
  |> filter(fn: (r) => r["_field"] == "fall_detected" or r["_field"] == "observation_id")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> filter(fn: (r) => r["fall_detected"] == true)
  |> keep(columns: ["_time", "patient_id", "observation_id"])
'''
        tables = _get_influxdb_client().query_api().query(query)
    except Exception as exc:
        logger.warning(f"InfluxDB fall count query failed (non-fatal): {exc}")
        return {}, {}

    # patient_id -> {dedup key: latest time}
    falls: dict = {}
    for table in tables:
        for record in table.records:
            pid = record.values.get("patient_id")
            if not pid:
                continue
            obs = record.values.get("observation_id")
            t = record.get_time()
            key = obs if obs and obs != _SIMULATED_OBSERVATION_ID else f"time:{t.isoformat()}"
            seen = falls.setdefault(pid, {})
            if key not in seen or t > seen[key]:
                seen[key] = t

    cutoff_24h = datetime.now(timezone.utc) - timedelta(hours=24)
    counts_30d = {pid: len(keys) for pid, keys in falls.items()}
    counts_24h = {pid: sum(1 for t in keys.values() if t >= cutoff_24h) for pid, keys in falls.items()}
    return counts_30d, counts_24h


def write_fall_event(event: dict) -> None:
    """Write one fall_events point for a fall/alert received over MQTT.

    Same schema as the mobile app's write (see module docstring). The timestamp
    is the detection time from the payload, so a point the mobile app writes
    for the same fall to the same InfluxDB lands on the same time.
    patient_confirmed must already be the int encoding (1 / 0 / -1).
    Non-fatal: failures are logged and the live alert path is unaffected.
    """
    bucket = _FALL_EVENTS_BUCKET
    if not bucket:
        logger.warning("INFLUXDB_FALL_EVENTS_BUCKET not configured — skipping fall_events write")
        return

    patient_id = event.get("patient_id") or ""
    if not patient_id:
        logger.warning("fall/alert without patient_id — skipping fall_events write")
        return

    ts = datetime.now(timezone.utc)
    raw_ts = event.get("timestamp")
    if raw_ts:
        try:
            ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning(f"Unparseable timestamp {raw_ts!r} — using receive time")

    device_id = event.get("device_id") or patient_store.get_mac_map().get(patient_id, "")

    point = (
        Point("fall_events")
        .tag("patient_id", patient_id)
        .tag("device_id", device_id)
        .field("fall_detected", True)
        .field("patient_confirmed", int(event.get("patient_confirmed", -1)))
        .time(ts)
    )
    # needs_help is null when the patient did not answer the second question;
    # leave the field out rather than storing a misleading false
    if event.get("needs_help") is not None:
        point.field("needs_help", bool(event["needs_help"]))
    if event.get("confidence") is not None:
        point.field("confidence", round(float(event["confidence"]), 4))
    if event.get("observation_id"):
        point.field("observation_id", str(event["observation_id"]))
    if event.get("model_version"):
        point.field("model_version", str(event["model_version"]))

    try:
        write_api = _get_influxdb_client().write_api(write_options=SYNCHRONOUS)
        write_api.write(bucket=bucket, record=point)
        logger.info(
            f"InfluxDB fall_events write OK  patient={patient_id}  "
            f"observation_id={event.get('observation_id')}  time={ts.isoformat()}"
        )
    except Exception as exc:
        logger.warning(f"InfluxDB fall_events write failed (non-fatal): {exc}")


def list_patients() -> List[dict]:
    """
    Return one dict per patient from the SQLite patient store, with fall counts
    joined in from InfluxDB.

    The patient list is read live from SQLite on every call (no module-level
    caching), so patients added via sync_from_env() on container recreate show
    up immediately.
    """
    fall_counts_30d, fall_counts_today = _get_fall_counts()
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
