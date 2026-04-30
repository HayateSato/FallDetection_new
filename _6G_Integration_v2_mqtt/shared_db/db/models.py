"""
SQLAlchemy ORM models — shared between inference_server and fall_dashboard.

Tables (all in the 'fall_detection' Postgres database):

  inference_log      — one row per /predict call; written by inference_server
  feature_snapshot   — one feature value per row; FK → inference_log; written by inference_server
  fall_history       — one row per confirmed alert; FK → inference_log via observation_id;
                       written by fall_dashboard on MQTT fall/alert arrival
  participant_session — one row per active patient session; written by fall_dashboard

Cross-reference key:
  observation_id (UUID string) is generated at the start of every /predict call.
  It is returned in the HTTP response so mock_app can include it in the MQTT alert
  payload. fall_dashboard stores it in fall_history, linking the two tables without
  needing a synchronous DB round-trip inside the HTTP handler.

Retraining query:
  SELECT il.*, fs.feature_name, fs.feature_value, fh.patient_confirmed, fh.needs_help
  FROM   inference_log il
  JOIN   feature_snapshot fs ON fs.inference_id = il.id
  JOIN   fall_history fh     ON fh.observation_id = il.observation_id
  WHERE  il.fall_detected = TRUE
    AND  fh.patient_confirmed = 'yes'
"""

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, func
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class InferenceLog(Base):
    """One row per /predict call. Written by inference_server via BackgroundTask."""
    __tablename__ = "inference_log"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    observation_id = Column(String(36), unique=True, nullable=False, index=True)  # UUID from /predict
    patient_id     = Column(String(100), nullable=False, index=True)
    device_id      = Column(String(100))
    model_version  = Column(String(64))
    fall_detected  = Column(Boolean, nullable=False)
    confidence     = Column(Float)
    window_size    = Column(Integer)                # number of ACC samples used
    latency_ms     = Column(Integer)
    detection_time = Column(DateTime(timezone=True), nullable=False)

    features     = relationship("FeatureSnapshot", back_populates="inference",
                                cascade="all, delete-orphan")
    fall_history = relationship("FallHistory", back_populates="inference",
                                uselist=False)


class FeatureSnapshot(Base):
    """One row per feature per inference. Enables retraining from stored data."""
    __tablename__ = "feature_snapshot"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    inference_id  = Column(Integer, ForeignKey("inference_log.id"),
                           nullable=False, index=True)
    feature_name  = Column(String(50), nullable=False)
    feature_value = Column(Float)

    inference = relationship("InferenceLog", back_populates="features")


class FallHistory(Base):
    """
    One row per confirmed fall alert. Written by fall_dashboard on MQTT arrival.

    observation_id links back to inference_log without requiring a synchronous
    DB round-trip in the inference server — the UUID is carried in the HTTP
    response → MQTT alert payload → fall_dashboard.
    """
    __tablename__ = "fall_history"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    observation_id    = Column(String(36),
                               ForeignKey("inference_log.observation_id"),
                               index=True)          # may be NULL if inference_log not yet written
    patient_id        = Column(String(100), nullable=False, index=True)
    fall_detected     = Column(Boolean, nullable=False)
    patient_confirmed = Column(String(20), default="not_answered")  # 'yes'/'no'/'not_answered'
    needs_help        = Column(Boolean)
    detection_time    = Column(DateTime(timezone=True), index=True)
    alert_time        = Column(DateTime(timezone=True), server_default=func.now())

    inference = relationship("InferenceLog", back_populates="fall_history")


class ParticipantSession(Base):
    """One row per patient session. Written by fall_dashboard on startup."""
    __tablename__ = "participant_session"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    participant_name = Column(String(100), nullable=False, index=True)
    gender           = Column(String(10))
    start_time       = Column(DateTime(timezone=True), server_default=func.now())
    end_time         = Column(DateTime(timezone=True), nullable=True)
    fall_count       = Column(Integer, default=0)
