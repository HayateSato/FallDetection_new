"""
Fall Dashboard database layer.
==============================

Tables written by this module:
  participant_session  — one row per active patient session

Fall history is now stored in InfluxDB (written by the mobile app after patient
confirmation). list_falls() queries the `fall_events` measurement.

InfluxDB schema (measurement: fall_events):
  Tags  : patient_id, device_id
  Fields: fall_detected (bool), patient_confirmed (str), needs_help (bool),
          observation_id (str), confidence (float), model_version (str)
  Time  : fall detection time
"""

import logging
import os
from contextlib import contextmanager
from typing import Generator, List, Optional

from sqlalchemy import select

logger = logging.getLogger(__name__)

_FALL_EVENTS_BUCKET = os.getenv("INFLUXDB_FALL_EVENTS_BUCKET") or os.getenv("INFLUXDB_BUCKET", "")

# ---------------------------------------------------------------------------
# Session factory + models
# ---------------------------------------------------------------------------
from shared_db.db.session import SessionLocal, init_db as _shared_init_db
from shared_db.db.models import ParticipantSession


# ---------------------------------------------------------------------------
# Public init
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables on first run. Safe to call repeatedly."""
    _shared_init_db()
    logger.info("Database ready (shared schema)")


# ---------------------------------------------------------------------------
# Session helper
# ---------------------------------------------------------------------------

@contextmanager
def session_scope() -> Generator:
    """Context-managed session — commits on success, rolls back on error."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def ensure_session(patient_id: str) -> None:
    """Open a participant_session row for this patient if no active one exists."""
    with session_scope() as db:
        existing = db.scalar(
            select(ParticipantSession)
            .where(ParticipantSession.participant_name == patient_id)
            .where(ParticipantSession.end_time.is_(None))
        )
        if existing is None:
            db.add(ParticipantSession(participant_name=patient_id))


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
    """Return one row per patient. Fall counts come from InfluxDB fall_events."""
    fall_counts = _get_fall_counts()
    with session_scope() as db:
        sessions = db.execute(
            select(ParticipantSession).order_by(ParticipantSession.participant_name)
        ).scalars().all()

        seen = set()
        out = []
        for s in sessions:
            if s.participant_name in seen:
                continue
            seen.add(s.participant_name)
            out.append({
                "patient_id":      s.participant_name,
                "fall_count":      fall_counts.get(s.participant_name, 0),
                "session_started": s.start_time.isoformat() if s.start_time else None,
                "session_active":  s.end_time is None,
            })
        return out


def list_falls(
    patient_id: Optional[str] = None,
    only_falls: bool = True,
    limit: int = 200,
) -> List[dict]:
    """
    Return fall history by querying the `fall_events` measurement in InfluxDB.

    This replicates what FOCUS's Flutter caregiver dashboard needs to implement.
    The data is written by the mobile app (or mock_app in local testing) after
    each patient confirmation popup.

    Flux query (reference for FOCUS DevOps Flutter implementation):
      from(bucket: "<bucket>")
        |> range(start: -30d)
        |> filter(fn: (r) => r["_measurement"] == "fall_events")
        |> filter(fn: (r) => r["patient_id"] == "<patient_id>")   // optional
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> sort(columns: ["_time"], desc: true)
        |> limit(n: <limit>)
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

    patient_filter = (
        f'  |> filter(fn: (r) => r["patient_id"] == "{patient_id}")\n'
        if patient_id else ""
    )
    only_falls_filter = (
        '  |> filter(fn: (r) => r["fall_detected"] == true)\n'
        if only_falls else ""
    )

    query = f'''from(bucket: "{bucket}")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "fall_events")
{patient_filter}{only_falls_filter}  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
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
                "observation_id":    v.get("observation_id"),
                "patient_id":        v.get("patient_id") or v.get("_measurement", ""),
                "fall_detected":     bool(v.get("fall_detected", True)),
                "patient_confirmed": v.get("patient_confirmed"),
                "needs_help":        v.get("needs_help"),
                "detection_time":    record.get_time().isoformat() if record.get_time() else None,
            })

    return results
