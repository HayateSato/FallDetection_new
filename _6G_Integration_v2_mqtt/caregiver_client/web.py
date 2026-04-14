"""
Caregiver dashboard web layer (FastAPI).

  GET  /                          → dashboard index.html (caregiver view)
  GET  /patient/                  → patient.html (patient feedback popup)
  GET  /api/patients              → list of patients with fall counts
  GET  /api/falls                 → fall_history rows  (?patient_id=&only_falls=&limit=)
  POST /api/falls/{id}/confirm    → set patient_confirmed = yes | no | not_answered
  GET  /api/stream                → Server-Sent Events feed of live fall events

The fall stream is fed by FallEventBroker (subscribes to MQTT broker).
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from caregiver_client import db as cdb
from caregiver_client.mqtt_listener import FallEventBroker

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

# ---------------------------------------------------------------------------
# Server-side auto-confirm timer (10 seconds)
# If no patient feedback arrives within 10s, auto-set 'not_answered'.
# ---------------------------------------------------------------------------
AUTO_CONFIRM_SECONDS = 10
_pending_timers: Dict[int, asyncio.Task] = {}


async def _auto_confirm_after(fall_id: int, seconds: int) -> None:
    """Wait `seconds`, then set patient_confirmed='not_answered' if still pending."""
    try:
        await asyncio.sleep(seconds)
        with cdb.session_scope() as db:
            row = db.get(cdb.FallHistory, fall_id)
            if row and row.patient_confirmed == "not_answered":
                logger.info(f"Auto-confirm timer expired  fall_id={fall_id} → not_answered (treated as fall)")
        _pending_timers.pop(fall_id, None)
    except asyncio.CancelledError:
        pass


def start_auto_confirm_timer(fall_id: int) -> None:
    """Start a 10s timer for a new fall. Cancelled if patient responds in time."""
    if fall_id in _pending_timers:
        return
    task = asyncio.ensure_future(_auto_confirm_after(fall_id, AUTO_CONFIRM_SECONDS))
    _pending_timers[fall_id] = task
    logger.info(f"Auto-confirm timer started  fall_id={fall_id}  timeout={AUTO_CONFIRM_SECONDS}s")


def cancel_auto_confirm_timer(fall_id: int) -> None:
    """Cancel the timer when the patient responds."""
    task = _pending_timers.pop(fall_id, None)
    if task and not task.done():
        task.cancel()
        logger.info(f"Auto-confirm timer cancelled  fall_id={fall_id}")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup() -> None:
    cdb.init_db()
    # broker.start() is called from client.py after wiring the on_fall callback


@app.on_event("shutdown")
async def _shutdown() -> None:
    await broker.stop()
    for task in _pending_timers.values():
        task.cancel()


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
    cancel_auto_confirm_timer(fall_id)
    logger.info(f"Patient feedback received  fall_id={fall_id}  confirmed={confirmed}")
    return {"ok": True, "fall_id": fall_id, "patient_confirmed": confirmed}


# ---------------------------------------------------------------------------
# Patient page (served as a standalone route so /patient/ works)
# ---------------------------------------------------------------------------

@app.get("/patient")
@app.get("/patient/")
async def patient_page():
    return FileResponse(DASHBOARD_DIR / "patient.html")


@app.get("/patient/patient.js")
async def patient_js():
    return FileResponse(DASHBOARD_DIR / "patient.js", media_type="application/javascript")


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
# Static dashboard files (mounted last so /api/* and /patient/* take precedence)
# ---------------------------------------------------------------------------

if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")
else:
    logger.warning(f"Dashboard directory not found at {DASHBOARD_DIR}")
