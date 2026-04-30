"""Pydantic schema for fall events — used in SSE streams and Redis pub/sub."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class FallEvent(BaseModel):
    """Emitted by ml_server to Redis when a fall is detected.
    Consumed by caregiver/api SSE endpoint and emergency notification service.
    """
    patient_id: str                  # participant name used as identifier
    fall_detected: bool
    confidence: float
    model_version: str
    timestamp: datetime
    inference_id: Optional[int] = None   # inference_log.id for DB cross-reference
