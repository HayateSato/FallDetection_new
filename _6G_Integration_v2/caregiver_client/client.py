"""
Caregiver client entry point.

Starts three things in one process:
  1. The InfluxDB poller (background thread)
  2. The FastAPI web server (uvicorn) serving the dashboard + JSON API
  3. The Redis subscriber (started inside FastAPI's startup hook)

Run from `_6G_Integration/` as the working directory:

    python -m caregiver_client.client

Configuration is read from the project root .env file (the same one
the inference server uses). Environment variables prefixed with CLIENT_*
can override defaults — see .env for the full list.
"""

import logging
import os
import signal
import sys
from datetime import datetime

import uvicorn
from dotenv import load_dotenv

# Load .env from current working directory (must be _6G_Integration/)
load_dotenv()

from caregiver_client import db as cdb           # noqa: E402
from caregiver_client.influx_poller import InfluxPoller  # noqa: E402
from caregiver_client.inference_client import InferenceServerClient  # noqa: E402
from caregiver_client.web import app, broker     # noqa: E402

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
# Wire poller into the FastAPI lifecycle
# ---------------------------------------------------------------------------

PATIENT_IDS = [p.strip() for p in os.getenv("PATIENT_IDS", "").split(",") if p.strip()]

# Optional comma-separated patient_id:device_id pairs
DEVICE_MAP = {}
for pair in os.getenv("PATIENT_DEVICE_MAP", "").split(","):
    if ":" in pair:
        pid, did = pair.split(":", 1)
        DEVICE_MAP[pid.strip()] = did.strip()

POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))
POLL_LOOKBACK_SECONDS = int(os.getenv("POLL_LOOKBACK_SECONDS", "15"))
INFLUX_PATIENT_TAG    = os.getenv("INFLUX_PATIENT_TAG", "").strip() or None

WEB_HOST = os.getenv("CAREGIVER_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("CAREGIVER_PORT", "8002"))

_poller: "InfluxPoller | None" = None


@app.on_event("startup")
async def _start_poller() -> None:
    """Spawn the InfluxDB poller after the broker is up."""
    global _poller
    if not PATIENT_IDS:
        logger.warning("PATIENT_IDS is empty — poller will NOT start. "
                       "Set PATIENT_IDS in .env to enable polling.")
        return

    inference = InferenceServerClient()
    logger.info(f"Inference server: {inference.server_url}")

    # Pre-create participant_session rows so patients show on the dashboard immediately
    for pid in PATIENT_IDS:
        cdb.ensure_session(pid)

    async def _on_fall(event: dict) -> None:
        await broker.publish_local(event)

    def _on_fall_sync(event: dict) -> None:
        # Called from the poller thread — schedule onto the event loop
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(_on_fall(event), loop)
        except Exception as exc:
            logger.warning(f"Could not forward poller fall to broker: {exc}")

    _poller = InfluxPoller(
        inference_client = inference,
        patient_ids      = PATIENT_IDS,
        device_map       = DEVICE_MAP,
        poll_interval    = POLL_INTERVAL_SECONDS,
        lookback_seconds = POLL_LOOKBACK_SECONDS,
        patient_tag      = INFLUX_PATIENT_TAG,
        on_fall          = _on_fall_sync,
    )
    _poller.start()


@app.on_event("shutdown")
async def _stop_poller() -> None:
    if _poller is not None:
        _poller.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _banner() -> None:
    print("=" * 64)
    print("  Caregiver Dashboard — 6G / Charite Integration")
    print(f"  Web UI:           http://{WEB_HOST}:{WEB_PORT}/")
    print(f"  Inference server: {os.getenv('INFERENCE_SERVER_URL', 'http://localhost:8001')}")
    print(f"  Patients:         {PATIENT_IDS or '(none configured)'}")
    print(f"  Poll interval:    {POLL_INTERVAL_SECONDS}s")
    print(f"  DB URL:           {os.getenv('DATABASE_URL', 'sqlite:///./caregiver.db')}")
    print(f"  Redis URL:        {os.getenv('REDIS_URL', '(not set — no live SSE)')}")
    print("=" * 64)


def main() -> None:
    _banner()

    def _graceful(*_):
        logger.info("Shutting down caregiver client...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    # Pass the app object directly (not a "module:attr" string) so the
    # startup hooks registered in this file stay attached to the same instance.
    uvicorn.run(
        app,
        host=WEB_HOST,
        port=WEB_PORT,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
