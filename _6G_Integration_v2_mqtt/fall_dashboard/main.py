"""
Fall Dashboard entry point.

Starts two things in one process:
  1. The MQTT subscriber (started inside FastAPI's startup hook)
  2. The FastAPI web server (uvicorn) serving the dashboard + JSON API

The mock_app (or eventually the real mobile app) is responsible for:
  - Fetching sensor data and calling the inference server
  - Showing the patient confirmation popup (10s timeout)
  - Publishing fall/alert/<patient_id> to MQTT after patient confirms or times out

This service only:
  - Listens to MQTT fall/alert/# (confirmed alerts from the mobile app)
  - Fans the event out to connected dashboard browsers via SSE
  - No local database — patient list from PATIENT_IDS env var, history from InfluxDB

Run from _6G_Integration_v2_mqtt/ as working directory:

    python -m fall_dashboard.main
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

from fall_dashboard import web as cweb
from fall_dashboard.web import app, broker

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
logger = logging.getLogger("fall_dashboard")

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
# Wire MQTT → DB + SSE
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _start_mqtt_with_callback() -> None:
    """
    Set the on_fall callback on the broker BEFORE starting it, so every fall
    event received via MQTT triggers the SSE fan-out to the caregiver dashboard.
    """
    _loop = asyncio.get_running_loop()

    def _on_fall_mqtt(event: dict) -> None:  # noqa: C901
        """
        Called from paho's background thread on every confirmed fall alert
        (published by mock_app after patient confirmation / timeout).

        The patient confirmation step already happened in the mobile app before
        this alert was published — no auto-confirm timer needed here.
        """
        patient_id = event.get("patient_id", "unknown")
        needs_help = event.get("needs_help")

        # patient_confirmed encoding boundary:
        #   Postgres / MQTT payload  →  string:  'yes' | 'no' | 'not_answered' | None
        #   InfluxDB / SSE browser   →  int:       1   |   0  |      -1        | (excluded)
        # This is the single conversion point for the caregiver layer.
        # The same mapping lives in influx_writer._CONFIRMED_TO_INT for InfluxDB writes.
        pc_raw = event.get("patient_confirmed", "not_answered")
        if isinstance(pc_raw, str):
            pc_int = 1 if pc_raw == "yes" else (0 if pc_raw == "no" else -1)
            event["patient_confirmed"] = pc_int
        else:
            pc_int = int(pc_raw) if pc_raw is not None else -1

        # Caregiver alert conditions (SSE fan-out -> dashboard):
        #   -1 (not_answered) -> patient could not respond at all -> treat as serious fall
        #    1 (yes) + needs_help=True -> confirmed fall, patient asked for help
        # Silent (stored in InfluxDB for history; no banner):
        #    0 (no) -> patient says they didn't fall (false positive)
        #    1 (yes) + needs_help=False -> confirmed fall but patient says they are okay
        should_alert = (pc_int == -1 or (pc_int == 1 and needs_help is True))
        if should_alert:
            asyncio.run_coroutine_threadsafe(broker.publish_local(event), _loop)
            logger.info(
                f"Fall ALERT -> caregiver  patient={patient_id}  "
                f"confirmed={pc_int}  needs_help={needs_help}"
            )
        else:
            logger.info(
                f"Fall recorded (no caregiver alert)  patient={patient_id}  "
                f"confirmed={pc_int}  needs_help={needs_help}"
            )

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
    print(f"  MQTT broker:      {_mqtt_info}")
    print()
    print("  NOTE: sensor data is fetched by mock_app (run separately)")
    print("        python -m mock_app.main")
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
