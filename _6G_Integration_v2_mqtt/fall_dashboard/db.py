"""
Fall Dashboard database layer.
==============================

Tables written by this module:
  participant_session  — one row per active patient session

Tables written by inference_server (read-only here):
  inference_log        — one row per /predict call
  feature_snapshot     — one row per feature per inference

NOTE: fall_history table has been removed (migration 0003).
Fall timestamps are now written to FOCUS InfluxDB by the mobile app.
list_falls() currently returns an empty list — it needs to be replaced
with an InfluxDB query once the FOCUS InfluxDB schema is confirmed.
"""

import logging
from contextlib import contextmanager
from typing import Generator, List, Optional

from sqlalchemy import select

logger = logging.getLogger(__name__)

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
            db.add(ParticipantSession(participant_name=patient_id, fall_count=0))


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def list_patients() -> List[dict]:
    """Return one row per patient from participant_session."""
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
                "fall_count":      s.fall_count or 0,
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
    Return fall history rows.

    TODO: replace with InfluxDB query once FOCUS InfluxDB schema is confirmed.
    Fall timestamps are written to FOCUS InfluxDB by the mobile app.
    The caregiver dashboard fall-history view should read from there.
    """
    logger.warning(
        "list_falls() is not yet implemented — "
        "needs InfluxDB integration (FOCUS side). Returning empty list."
    )
    return []
