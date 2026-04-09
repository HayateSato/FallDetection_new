"""
Caregiver dashboard web layer (FastAPI).

  GET  /                          → dashboard index.html
  GET  /api/patients              → list of patients with fall counts
  GET  /api/falls                 → fall_history rows  (?patient_id=&only_falls=&limit=)
  POST /api/falls/{id}/confirm    → set patient_confirmed = yes | no | not_answered
  GET  /api/stream                → Server-Sent Events feed of live fall events

The fall stream is fed by FallEventBroker (subscribes to Redis fall_events).
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from caregiver_client import db as cdb
from caregiver_client.redis_listener import FallEventBroker

logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(__file__).parent / "dashboard"

broker = FallEventBroker()
app    = FastAPI(title="Caregiver Dashboard — 6G Integration", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    cdb.init_db()
    await broker.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await broker.stop()


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

# Populated by client.py startup — maps patient_id → MAC address
mac_map: dict = {}


@app.get("/api/patients")
def api_patients():
    patients = cdb.list_patients()
    for p in patients:
        p["mac_id"] = mac_map.get(p["patient_id"], "")
    return {"patients": patients}


@app.get("/api/falls")
def api_falls(
    patient_id: Optional[str] = Query(None),
    only_falls: bool          = Query(True),
    limit:      int           = Query(200, ge=1, le=2000),
):
    falls = cdb.list_falls(patient_id=patient_id, only_falls=only_falls, limit=limit)
    for f in falls:
        f["mac_id"] = mac_map.get(f["patient_id"], "")
    return {"falls": falls}


@app.post("/api/falls/{fall_id}/confirm")
def api_confirm(fall_id: int, confirmed: str = Query(...)):
    ok = cdb.update_fall_confirmation(fall_id, confirmed)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid fall_id or confirmed value.")
    return {"ok": True, "fall_id": fall_id, "patient_confirmed": confirmed}


# ---------------------------------------------------------------------------
# Server-Sent Events stream of live fall events
# ---------------------------------------------------------------------------

@app.get("/api/stream")
async def api_stream():
    """Long-lived SSE connection — emits one `data: {json}` line per fall event."""
    queue = broker.subscribe()

    async def event_generator():
        try:
            # initial hello so the browser knows we're connected
            yield "event: connected\ndata: {}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                except asyncio.TimeoutError:
                    # keepalive comment so proxies don't drop the connection
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            broker.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Static dashboard files (mounted last so /api/* takes precedence)
# ---------------------------------------------------------------------------

if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")
else:
    logger.warning(f"Dashboard directory not found at {DASHBOARD_DIR}")
