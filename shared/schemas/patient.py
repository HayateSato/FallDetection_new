"""Pydantic schemas for participant/patient session data."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class PatientSessionSchema(BaseModel):
    id: int
    participant_name: str
    gender: Optional[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    fall_count: int

    model_config = {"from_attributes": True}


class PatientSummary(BaseModel):
    """Lightweight patient view for the caregiver dashboard list."""
    id: int
    participant_name: str
    fall_count: int
    last_seen: Optional[datetime]       # from latest inference_log timestamp
    last_confidence: Optional[float]    # from latest inference_log
    active: bool                        # end_time is None
