"""Pydantic schemas for inference data (read from Postgres)."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class FeatureSnapshotSchema(BaseModel):
    feature_name: str
    feature_value: Optional[float]

    model_config = {"from_attributes": True}


class InferenceLogSchema(BaseModel):
    id: int
    timestamp: datetime
    model_version: Optional[str]
    fall_detected: Optional[bool]
    confidence: Optional[float]
    window_size: Optional[int]
    inference_mode: Optional[str]
    latency_ms: Optional[int]
    participant: Optional[str]

    model_config = {"from_attributes": True}


class InferenceLogWithFeatures(InferenceLogSchema):
    features: list[FeatureSnapshotSchema] = []
