"""
Mock Mobile App — entry point.

Simulates a mobile app that reads sensor data from a SmarKo wearable and
sends it to the inference server for fall detection.

In this mock, InfluxDB replaces the BLE wearable — it fetches the same raw
LSB arrays that the wearable would normally stream directly to the app.

Run from _6G_Integration_v2_mqtt/ as working directory:

    python -m mock_app.client

Configuration is read from the project root .env file (shared with the
inference server and caregiver client).
"""

import logging
import os
import signal
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from mock_app.api_caller import InferenceServerClient
from mock_app.poller import MockAppPoller

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
            os.path.join(log_dir, f"mock_app_{_ts}.log"),
            mode="w", encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("mock_app")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PATIENT_IDS = [p.strip() for p in os.getenv("PATIENT_IDS", "").split(",") if p.strip()]
_mac_list   = [m.strip() for m in os.getenv("MAC_IDS", "").split(",")     if m.strip()]
MAC_MAP     = {pid: _mac_list[i] for i, pid in enumerate(PATIENT_IDS) if i < len(_mac_list)}

POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))
POLL_LOOKBACK_SECONDS = int(os.getenv("POLL_LOOKBACK_SECONDS", "15"))

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
def _banner() -> None:
    print("=" * 64)
    print("  Mock Mobile App — Fall Detection")
    print(f"  Inference server: {os.getenv('INFERENCE_SERVER_URL', 'http://localhost:8001')}")
    print(f"  Patients:         {PATIENT_IDS or '(none configured)'}")
    print(f"  MAC map:          {MAC_MAP or '(none)'}")
    print(f"  Poll interval:    {POLL_INTERVAL_SECONDS}s")
    print(f"  Lookback:         {POLL_LOOKBACK_SECONDS}s")
    print()
    print("  NOTE: fetching sensor data from InfluxDB (simulating wearable BLE stream)")
    print("  Falls will be published to MQTT by the inference server.")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    _banner()

    if not PATIENT_IDS:
        logger.error("PATIENT_IDS is not set in .env — nothing to poll.")
        sys.exit(1)

    def _graceful(*_):
        logger.info("Shutting down mock app...")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    inference = InferenceServerClient()
    poller    = MockAppPoller(
        inference_client = inference,
        patient_ids      = PATIENT_IDS,
        mac_map          = MAC_MAP,
        poll_interval    = POLL_INTERVAL_SECONDS,
        lookback_seconds = POLL_LOOKBACK_SECONDS,
    )
    poller.start()
    poller.join()  # block main thread; Ctrl-C triggers _graceful


if __name__ == "__main__":
    main()
