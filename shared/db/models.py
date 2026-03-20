"""
SQLAlchemy ORM models for the fall detection system.

Tables:
  inference_log      — every prediction made by the ML server
  feature_snapshot   — individual feature values per prediction (for retraining / debugging)
  participant_session — recording sessions per patient
  api_request_log    — server-side request audit trail
"""

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, func
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class InferenceLog(Base):
    """One row per /predict call. Core audit table."""
    __tablename__ = "inference_log"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    timestamp      = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    model_version  = Column(String(20))
    fall_detected  = Column(Boolean)
    confidence     = Column(Float)
    window_size    = Column(Integer)          # number of ACC samples used
    inference_mode = Column(String(10))       # 'local' or 'remote'
    latency_ms     = Column(Integer)          # end-to-end inference pipeline latency
    participant    = Column(String(100))      # participant/patient name from recording session

    features = relationship("FeatureSnapshot", back_populates="inference",
                            cascade="all, delete-orphan")


class FeatureSnapshot(Base):
    """Individual feature values per inference — enables retraining from stored data."""
    __tablename__ = "feature_snapshot"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    inference_id = Column(Integer, ForeignKey("inference_log.id"), nullable=False)
    feature_name = Column(String(50), nullable=False)
    feature_value = Column(Float)

    inference = relationship("InferenceLog", back_populates="features")


class ParticipantSession(Base):
    """One row per recording session. Tracks participant metadata and fall count."""
    __tablename__ = "participant_session"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    participant_name = Column(String(100), nullable=False)
    gender           = Column(String(10))
    start_time       = Column(DateTime(timezone=True), server_default=func.now())
    end_time         = Column(DateTime(timezone=True), nullable=True)
    fall_count       = Column(Integer, default=0)


class ApiRequestLog(Base):
    """Server-side request audit log. Stores hashed API key — never the raw key."""
    __tablename__ = "api_request_log"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    timestamp       = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    client_ip       = Column(String(45))           # supports IPv6
    endpoint        = Column(String(100))
    status_code     = Column(Integer)
    response_time_ms = Column(Integer)
    api_key_hash    = Column(String(64))           # SHA-256 hex digest
