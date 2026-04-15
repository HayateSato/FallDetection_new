"""
Fall Dashboard database layer.
==============================

Uses the shared ORM models from shared.db so both the inference_server
(inference_log + feature_snapshot) and the fall_dashboard (fall_history +
participant_session) write to the same SQLite / Postgres instance.

Tables written by this module:
  fall_history         — one row per MQTT fall/alert event received
  participant_session  — one row per active patient session

Tables written by inference_server (read-only here):
  inference_log        — one row per /predict call
  feature_snapshot     — one row per feature per inference

Cross-reference: observation_id (UUID) is received in the MQTT alert payload
and stored in fall_history. It links back to inference_log for the retraining
JOIN without requiring a synchronous DB call in the inference server.

Defaults to SQLite (caregiver.db) so the client runs without Postgres. Set
DATABASE_URL in .env to point at Postgres in production.
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, List, Optional

from sqlalchemy import func, select

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session factory + models — shared with inference_server
# ---------------------------------------------------------------------------
from shared.db.session import SessionLocal, init_db as _shared_init_db
from shared.db.models import FallHistory, ParticipantSession


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

def record_fall(
    patient_id:        str,
    fall_detected:     bool,
    detection_time:    Optional[datetime] = None,
    patient_confirmed: str = "not_answered",
    observation_id:    Optional[str] = None,
    needs_help:        Optional[bool] = None,
) -> int:
    """
    Insert a fall_history row. Returns the new row id.

    observation_id — UUID from the /predict HTTP response, carried through
                     the MQTT alert payload. Links this row to inference_log
                     for the retraining JOIN:
                       SELECT * FROM inference_log il
                       JOIN fall_history fh ON fh.observation_id = il.observation_id
    needs_help     — patient response to the 'do you need help?' follow-up
                     question in the mobile app confirmation popup.
    """
    with session_scope() as db:
        row = FallHistory(
            observation_id    = observation_id,
            patient_id        = patient_id,
            fall_detected     = fall_detected,
            detection_time    = detection_time or datetime.utcnow(),
            patient_confirmed = patient_confirmed,
            needs_help        = needs_help,
        )
        db.add(row)
        db.flush()

        if fall_detected:
            sess = db.scalar(
                select(ParticipantSession)
                .where(ParticipantSession.participant_name == patient_id)
                .where(ParticipantSession.end_time.is_(None))
                .order_by(ParticipantSession.start_time.desc())
            )
            if sess is not None:
                sess.fall_count = (sess.fall_count or 0) + 1

        return row.id


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
    """Return one row per patient with fall counts."""
    with session_scope() as db:
        falls_by_patient = dict(
            db.execute(
                select(FallHistory.patient_id, func.count(FallHistory.id))
                .where(FallHistory.fall_detected.is_(True))
                .group_by(FallHistory.patient_id)
            ).all()
        )
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
                "fall_count":      falls_by_patient.get(s.participant_name, 0),
                "session_started": s.start_time.isoformat() if s.start_time else None,
                "session_active":  s.end_time is None,
            })
        for pid, count in falls_by_patient.items():
            if pid not in seen:
                out.append({
                    "patient_id":      pid,
                    "fall_count":      count,
                    "session_started": None,
                    "session_active":  False,
                })
        return out


def list_falls(
    patient_id: Optional[str] = None,
    only_falls: bool = True,
    limit: int = 200,
) -> List[dict]:
    """Return fall_history rows ordered by most recent first."""
    with session_scope() as db:
        stmt = select(FallHistory).order_by(FallHistory.detection_time.desc()).limit(limit)
        if patient_id:
            stmt = stmt.where(FallHistory.patient_id == patient_id)
        if only_falls:
            stmt = stmt.where(FallHistory.fall_detected.is_(True))
        rows = db.execute(stmt).scalars().all()
        return [
            {
                "id":                r.id,
                "observation_id":    r.observation_id,
                "patient_id":        r.patient_id,
                "fall_detected":     bool(r.fall_detected),
                "patient_confirmed": r.patient_confirmed,
                "needs_help":        r.needs_help,
                "detection_time":    r.detection_time.isoformat() if r.detection_time else None,
            }
            for r in rows
        ]


def update_fall_confirmation(fall_id: int, confirmed: str) -> bool:
    """Set patient_confirmed for a single fall_history row."""
    if confirmed not in ("yes", "no", "not_answered"):
        return False
    with session_scope() as db:
        row = db.get(FallHistory, fall_id)
        if row is None:
            return False
        row.patient_confirmed = confirmed
        return True
