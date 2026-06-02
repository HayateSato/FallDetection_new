"""
SQLAlchemy ORM models — shared between inference_server and retrain pipeline.

Tables (all in the 'fall_detection' Postgres database):

  inference_log      — one row per /predict call; written by inference_server
  feature_snapshot   — one feature value per row; FK -> inference_log; written by inference_server
  participant_session — one row per active patient session; written by fall_dashboard

Patient confirmation (patient_confirmed, needs_help) is stored directly on
inference_log. The mobile app calls POST /inference/{observation_id}/confirm
after the patient responds to the popup — no separate fall_history table needed.

Retraining query:
  SELECT il.*, fs.feature_name, fs.feature_value
  FROM   inference_log il
  JOIN   feature_snapshot fs ON fs.inference_id = il.id
  WHERE  il.fall_detected = TRUE
    AND  il.patient_confirmed = 'yes'
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

    id                = Column(Integer, primary_key=True, autoincrement=True)
    observation_id    = Column(String(36), unique=True, nullable=False, index=True)
    patient_id        = Column(String(100), nullable=False, index=True)
    device_id         = Column(String(100))
    model_version     = Column(String(64))
    fall_detected     = Column(Boolean, nullable=False)
    confidence        = Column(Float)
    window_size       = Column(Integer)
    latency_ms        = Column(Integer)
    detection_time    = Column(DateTime(timezone=True), nullable=False)
    patient_confirmed = Column(String(20), nullable=True)  # 'yes'/'no'/'not_answered'; set via /confirm
    needs_help        = Column(Boolean, nullable=True)

    features = relationship("FeatureSnapshot", back_populates="inference",
                            cascade="all, delete-orphan")


class FeatureSnapshot(Base):
    """One row per feature per inference. Enables retraining from stored data."""
    __tablename__ = "feature_snapshot"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    inference_id  = Column(Integer, ForeignKey("inference_log.id"),
                           nullable=False, index=True)
    feature_name  = Column(String(50), nullable=False)
    feature_value = Column(Float)

    inference = relationship("InferenceLog", back_populates="features")


class ParticipantSession(Base):
    """One row per patient session. Written by fall_dashboard on startup."""
    __tablename__ = "participant_session"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    participant_name = Column(String(100), nullable=False, index=True)
    gender           = Column(String(10))
    start_time       = Column(DateTime(timezone=True), server_default=func.now())
    end_time         = Column(DateTime(timezone=True), nullable=True)
    fall_count       = Column(Integer, default=0)
