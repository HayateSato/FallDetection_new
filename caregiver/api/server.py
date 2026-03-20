"""
Caregiver API Server

FastAPI REST API for the care-giver dashboard.
Reads inference history from PostgreSQL and streams live fall events from Redis.

Endpoints:
    POST /auth/login                  — username/password → JWT token
    GET  /health                      — health check
    GET  /patients                    — list all participant sessions
    GET  /patients/{name}/falls       — fall history for one patient
    GET  /patients/{name}/stream      — SSE stream of live fall events for one patient
    GET  /patients/stream             — SSE stream of ALL fall events (any patient)
    GET  /stats/summary               — aggregate stats for today

Run from project root:
    uvicorn caregiver.api.server:app --host 0.0.0.0 --port 8002
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auth models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


# ---------------------------------------------------------------------------
# Simple in-memory user store (replace with DB table in production)
# ---------------------------------------------------------------------------
# Users are configured via environment variables:
#   CAREGIVER_USERS=alice:hashed_password,bob:hashed_password
# Generate a hash: python -c "from shared.auth.jwt_utils import hash_password; print(hash_password('mypassword'))"

def _load_users() -> dict:
    raw = os.environ.get("CAREGIVER_USERS", "")
    users = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if ":" in entry:
            username, pw_hash = entry.split(":", 1)
            users[username.strip()] = pw_hash.strip()
    return users


_USERS = _load_users()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Caregiver API",
    description="Patient monitoring API for care-givers. Reads from PostgreSQL, streams from Redis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to dashboard origin in production
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------

def get_db():
    from shared.db.session import SessionLocal
    if SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "caregiver_api"}


@app.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    """Authenticate a care-giver and return a JWT token."""
    from shared.auth.jwt_utils import create_token, verify_password

    user_hash = _USERS.get(req.username)
    if not user_hash:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(req.password, user_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token({"sub": req.username, "role": "caregiver"})
    logger.info(f"Caregiver login: {req.username}")
    return LoginResponse(access_token=token, role="caregiver")


@app.get("/patients")
async def list_patients(db: Session = Depends(get_db)):
    """
    List all participant sessions with summary stats.
    Returns patients ordered by most recent session first.
    """
    from shared.db.models import ParticipantSession, InferenceLog

    sessions = db.query(ParticipantSession).order_by(desc(ParticipantSession.start_time)).all()

    result = []
    for s in sessions:
        # Get latest inference for this patient
        latest = (
            db.query(InferenceLog)
            .filter(InferenceLog.participant == s.participant_name)
            .order_by(desc(InferenceLog.timestamp))
            .first()
        )
        result.append({
            "id": s.id,
            "participant_name": s.participant_name,
            "gender": s.gender,
            "fall_count": s.fall_count,
            "start_time": s.start_time.isoformat() if s.start_time else None,
            "end_time": s.end_time.isoformat() if s.end_time else None,
            "active": s.end_time is None,
            "last_seen": latest.timestamp.isoformat() if latest else None,
            "last_confidence": latest.confidence if latest else None,
        })

    return {"patients": result}


@app.get("/patients/{patient_name}/falls")
async def patient_falls(
    patient_name: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Paginated fall history for one patient."""
    from shared.db.models import InferenceLog

    total = (
        db.query(func.count(InferenceLog.id))
        .filter(InferenceLog.participant == patient_name, InferenceLog.fall_detected == True)
        .scalar()
    )
    falls = (
        db.query(InferenceLog)
        .filter(InferenceLog.participant == patient_name, InferenceLog.fall_detected == True)
        .order_by(desc(InferenceLog.timestamp))
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "patient_name": patient_name,
        "total": total,
        "limit": limit,
        "offset": offset,
        "falls": [
            {
                "id": f.id,
                "timestamp": f.timestamp.isoformat(),
                "confidence": f.confidence,
                "model_version": f.model_version,
                "latency_ms": f.latency_ms,
            }
            for f in falls
        ],
    }


@app.get("/patients/{patient_name}/stream")
async def patient_event_stream(patient_name: str):
    """
    SSE stream of live fall events for a specific patient.
    Browser connects with EventSource; auto-reconnects on disconnect.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        from shared.redis_client import subscribe_fall_events
        try:
            async for event in subscribe_fall_events():
                if event.get("patient_id") != patient_name:
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # tells nginx not to buffer SSE
        },
    )


@app.get("/patients/stream")
async def all_events_stream():
    """
    SSE stream of ALL fall events (any patient).
    Used by the caregiver dashboard alert banner.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        from shared.redis_client import subscribe_fall_events
        try:
            async for event in subscribe_fall_events():
                if event.get("fall_detected"):   # only push actual falls
                    yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/stats/summary")
async def stats_summary(db: Session = Depends(get_db)):
    """Aggregate stats for today — used by the dashboard header."""
    from shared.db.models import InferenceLog

    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    falls_today = (
        db.query(func.count(InferenceLog.id))
        .filter(InferenceLog.fall_detected == True, InferenceLog.timestamp >= today_start)
        .scalar()
    )
    total_today = (
        db.query(func.count(InferenceLog.id))
        .filter(InferenceLog.timestamp >= today_start)
        .scalar()
    )
    avg_confidence = (
        db.query(func.avg(InferenceLog.confidence))
        .filter(InferenceLog.timestamp >= today_start)
        .scalar()
    )

    return {
        "falls_today": falls_today,
        "predictions_today": total_today,
        "avg_confidence_today": round(float(avg_confidence), 3) if avg_confidence else None,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("CAREGIVER_API_PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
