"""
Fall Dashboard web layer (FastAPI).

  GET  /                          → dashboard index.html (caregiver view)
  GET  /api/patients              → list of patients with fall counts
  GET  /api/falls                 → fall history (?patient_id=&only_falls=&limit=)  [stub — needs InfluxDB]
  GET  /api/stream                → Server-Sent Events feed of live fall events

The fall stream is fed by FallEventBroker (subscribes to MQTT broker).
Patient confirmation is handled by the mobile app (real) or mock_app/patient_server.py (dev).
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
from pydantic import BaseModel

from fall_dashboard import db as cdb
from fall_dashboard import patient_store
from fall_dashboard.mqtt_listener import FallEventBroker


logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(__file__).parent / "dashboard"

broker = FallEventBroker()
app    = FastAPI(title="Fall Dashboard — 6G Integration", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("shutdown")
async def _shutdown() -> None:
    await broker.stop()


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

# Populated by client.py startup — maps patient_id → MAC address
mac_map: dict = {}


class PatientCreate(BaseModel):
    patient_id: str
    name: Optional[str] = None
    mac_id: Optional[str] = None


@app.get("/api/patients")
def api_patients():
    # list_patients() already carries mac_id from the SQLite store; fall back to
    # the mac_map only when the store has no MAC for that patient.
    patients = cdb.list_patients()
    for p in patients:
        if not p.get("mac_id"):
            p["mac_id"] = mac_map.get(p["patient_id"], "")
    return {"patients": patients}


@app.post("/api/patients", status_code=201)
def api_add_patient(body: PatientCreate):
    pid = body.patient_id.strip()
    if not pid:
        raise HTTPException(status_code=400, detail="patient_id is required")
    patient_store.upsert_patient(patient_id=pid, name=body.name, mac_id=body.mac_id)
    return {"patient_id": pid, "created": True}


@app.delete("/api/patients/{patient_id}")
def api_delete_patient(patient_id: str):
    deleted = patient_store.delete_patient(patient_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"patient_id": patient_id, "deleted": True}


@app.get("/api/falls")
def api_falls(
    patient_id: Optional[str] = Query(None),
    only_falls: bool          = Query(True),
    limit:      int           = Query(200, ge=1, le=2000),
    hours:      int           = Query(720, ge=1, le=8760),
):
    falls = cdb.list_falls(patient_id=patient_id, only_falls=only_falls, limit=limit, hours=hours)
    for f in falls:
        f["mac_id"] = mac_map.get(f["patient_id"], "")
    return {"falls": falls}


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
# Static dashboard files (mounted last so /api/* routes take precedence)
# ---------------------------------------------------------------------------

if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")
else:
    logger.warning(f"Dashboard directory not found at {DASHBOARD_DIR}")
