"""
Caregiver client database layer.
================================

Two tables:

  participant_session  — one row per recording session for a patient
                         (reused from the full system's shared schema)

  fall_history         — one row per fall event observed by this client
                         (new — exact columns requested for the 6G integration)

Defaults to a local SQLite file (caregiver.db) so the client runs without
any external Postgres dependency. Set DATABASE_URL in .env to point at
Postgres in production, e.g.

    DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/caregiver
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, List, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Integer, String, create_engine, func, select
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine + session factory
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./caregiver.db")

# SQLite needs check_same_thread=False because the InfluxDB poller and
# the FastAPI request handlers run in different threads.
_engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, future=True, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ParticipantSession(Base):
    """One row per recording session — reused from full-system shared schema."""
    __tablename__ = "participant_session"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    participant_name = Column(String(100), nullable=False, index=True)
    gender           = Column(String(10))
    start_time       = Column(DateTime(timezone=True), server_default=func.now())
    end_time         = Column(DateTime(timezone=True), nullable=True)
    fall_count       = Column(Integer, default=0)


class FallHistory(Base):
    """
    One row per fall event observed by this caregiver client.

    Columns intentionally minimal — exactly what the 6G use case requires:
      patient_id        : FHIR patient identifier
      fall_detection    : binary, True if the model detected a fall
      patient_confirmed : 'yes' | 'no' | 'not_answered'  (manual override / future feedback)
      detection_time    : timestamp of the inference window
    """
    __tablename__ = "fall_history"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    patient_id        = Column(String(100), nullable=False, index=True)
    fall_detection    = Column(Boolean, nullable=False)
    patient_confirmed = Column(String(20), nullable=False, default="not_answered")
    detection_time    = Column(DateTime(timezone=True),
                               server_default=func.now(), nullable=False, index=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables on first run. Safe to call repeatedly."""
    Base.metadata.create_all(engine)
    logger.info(f"Database ready  url={DATABASE_URL}")


@contextmanager
def session_scope() -> Generator[Session, None, None]:
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


def record_fall(
    patient_id: str,
    fall_detected: bool,
    detection_time: Optional[datetime] = None,
    patient_confirmed: str = "not_answered",
) -> int:
    """Insert a fall_history row. Returns the new row id."""
    with session_scope() as db:
        row = FallHistory(
            patient_id=patient_id,
            fall_detection=fall_detected,
            detection_time=detection_time or datetime.utcnow(),
            patient_confirmed=patient_confirmed,
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


def list_patients() -> List[dict]:
    """Return one row per patient with fall counts."""
    with session_scope() as db:
        # Falls per patient
        falls_by_patient = dict(
            db.execute(
                select(FallHistory.patient_id, func.count(FallHistory.id))
                .where(FallHistory.fall_detection.is_(True))
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
                "patient_id":  s.participant_name,
                "fall_count":  falls_by_patient.get(s.participant_name, 0),
                "session_started": s.start_time.isoformat() if s.start_time else None,
                "session_active":  s.end_time is None,
            })
        # Patients that only show in fall_history (no session row yet)
        for pid, count in falls_by_patient.items():
            if pid not in seen:
                out.append({
                    "patient_id": pid,
                    "fall_count": count,
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
            stmt = stmt.where(FallHistory.fall_detection.is_(True))
        rows = db.execute(stmt).scalars().all()
        return [
            {
                "id":                r.id,
                "patient_id":        r.patient_id,
                "fall_detection":    bool(r.fall_detection),
                "patient_confirmed": r.patient_confirmed,
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
