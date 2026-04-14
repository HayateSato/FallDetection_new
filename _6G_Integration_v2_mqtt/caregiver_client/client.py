"""
Caregiver client entry point.

Starts two things in one process:
  1. The MQTT subscriber (started inside FastAPI's startup hook)
  2. The FastAPI web server (uvicorn) serving the dashboard + JSON API

The mock_app (or eventually the real mobile app) is now responsible for
fetching sensor data and calling the inference server. This client only:
  - Listens to MQTT for fall events published by the inference server
  - Writes each fall to the local DB
  - Fans the event out to connected dashboard browsers via SSE
  - Manages the 10-second auto-confirm timer for patient feedback

Run from _6G_Integration_v2_mqtt/ as working directory:

    python -m caregiver_client.client
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from caregiver_client import db as cdb
from caregiver_client import web as cweb
from caregiver_client.web import app, broker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_dir = os.path.join("results", "logs")
os.makedirs(log_dir, exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(log_dir, f"6g_caregiver_{_ts}.log"),
            mode="w", encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("caregiver_client")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PATIENT_IDS = [p.strip() for p in os.getenv("PATIENT_IDS", "").split(",") if p.strip()]
_mac_list   = [m.strip() for m in os.getenv("MAC_IDS", "").split(",")     if m.strip()]
MAC_MAP     = {pid: _mac_list[i] for i, pid in enumerate(PATIENT_IDS) if i < len(_mac_list)}

WEB_HOST = os.getenv("CAREGIVER_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("CAREGIVER_PORT", "8002"))

# Share the MAC map with the web layer so API responses include MAC addresses
cweb.mac_map = MAC_MAP

# ---------------------------------------------------------------------------
# Wire MQTT → DB + SSE + auto-confirm timer
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _start_mqtt_with_callback() -> None:
    """
    Set the on_fall callback on the broker BEFORE starting it, so every fall
    event received via MQTT triggers a DB write and SSE fan-out.
    """
    # Pre-create participant_session rows so patients appear on the dashboard immediately
    for pid in PATIENT_IDS:
        cdb.ensure_session(pid)

    _loop = asyncio.get_running_loop()

    def _on_fall_mqtt(event: dict) -> None:
        """
        Called from paho's background thread on every fall event.
        Writes the fall to the local DB, adds fall_id to the event dict,
        fans out to SSE clients, and starts the 10-second auto-confirm timer.
        """
        patient_id = event.get("patient_id", "unknown")
        ts         = event.get("timestamp")
        try:
            det_time = datetime.fromisoformat(ts) if ts else datetime.now(timezone.utc)
        except Exception:
            det_time = datetime.now(timezone.utc)

        cdb.ensure_session(patient_id)
        fall_id = cdb.record_fall(
            patient_id        = patient_id,
            fall_detected     = True,
            detection_time    = det_time,
            patient_confirmed = "not_answered",
        )
        event["fall_id"] = fall_id

        asyncio.run_coroutine_threadsafe(broker.publish_local(event), _loop)
        if fall_id is not None:
            _loop.call_soon_threadsafe(cweb.start_auto_confirm_timer, fall_id)

        logger.info(f"Fall recorded  patient={patient_id}  fall_id={fall_id}")

    broker.on_fall = _on_fall_mqtt
    await broker.start()


@app.on_event("shutdown")
async def _stop_poller() -> None:
    pass  # broker.stop() is already called from web.py's shutdown hook


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _banner() -> None:
    _mqtt_host = os.getenv("MQTT_BROKER_HOST", "")
    _mqtt_info = (f"{_mqtt_host}:{os.getenv('MQTT_BROKER_PORT', '1883')}"
                  if _mqtt_host else "(not set — no live SSE)")
    print("=" * 64)
    print("  Caregiver Dashboard — 6G / Charite Integration")
    print(f"  Web UI:           http://{WEB_HOST}:{WEB_PORT}/")
    print(f"  Patients:         {PATIENT_IDS or '(none configured)'}")
    print(f"  DB URL:           {os.getenv('DATABASE_URL', 'sqlite:///./caregiver.db')}")
    print(f"  MQTT broker:      {_mqtt_info}")
    print()
    print("  NOTE: sensor data is fetched by mock_app (run separately)")
    print("        python -m mock_app.client")
    print("=" * 64)


def main() -> None:
    _banner()

    def _graceful(*_):
        logger.info("Shutting down caregiver client...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    uvicorn.run(
        app,
        host=WEB_HOST,
        port=WEB_PORT,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
