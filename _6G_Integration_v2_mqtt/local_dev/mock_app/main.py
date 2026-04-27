"""
Mock Mobile App — entry point.

Simulates a mobile app that:
  1. Reads sensor data from InfluxDB (real app: reads from SmarKo wearable via BLE)
  2. POSTs to the inference server's /predict endpoint
  3. On fall_detected=True: opens a patient confirmation window (10s timeout)
  4. After timeout or confirmation: publishes fall/alert/<patient_id> to MQTT broker
  5. Caregiver dashboard receives the alert via MQTT

Run from _6G_Integration_v2_mqtt/ as working directory:

    python -m local_dev.mock_app.main
"""

import logging
import os
import signal
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from local_dev.mock_app.api_caller import InferenceServerClient
from local_dev.mock_app.poller import MockAppPoller

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

POLL_INTERVAL_SECONDS        = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))
POLL_LOOKBACK_SECONDS        = int(os.getenv("POLL_LOOKBACK_SECONDS", "15"))
MQTT_BROKER_HOST             = os.getenv("MQTT_BROKER_HOST", "").strip()
MQTT_BROKER_PORT             = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_USERNAME                = os.getenv("MQTT_USERNAME", "").strip()
MQTT_PASSWORD                = os.getenv("MQTT_PASSWORD", "").strip()
MQTT_ALERT_TOPIC             = os.getenv("MQTT_ALERT_TOPIC", "fall/alert")
MOCK_PATIENT_RESPONSE_TIMEOUT = int(os.getenv("MOCK_PATIENT_RESPONSE_TIMEOUT", "10"))


# ---------------------------------------------------------------------------
# MQTT publisher (for sending fall/alert after patient confirmation)
# ---------------------------------------------------------------------------
def _create_mqtt_publisher():
    """
    Create and connect a paho MQTT client used for publishing fall alerts.
    Returns None if MQTT_BROKER_HOST is not configured (alerts won't be sent).
    """
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        logger.warning("paho-mqtt not installed — fall alerts will not be published. "
                       "`pip install paho-mqtt`")
        return None

    if not MQTT_BROKER_HOST:
        logger.warning("MQTT_BROKER_HOST not set — fall alerts will not be published.")
        return None

    client = mqtt.Client(client_id="mock-app-publisher")
    if MQTT_USERNAME:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD or None)
    try:
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
        client.loop_start()
        logger.info(f"MQTT publisher connected  {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}"
                    f"  alert_topic={MQTT_ALERT_TOPIC}/<patient_id>")
        return client
    except Exception as exc:
        logger.warning(f"MQTT publisher connection failed ({exc}) — "
                       "fall alerts will not reach caregiver.")
        return None


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
def _banner() -> None:
    _mqtt_info = (f"{MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}"
                  if MQTT_BROKER_HOST else "(not set — alerts won't reach caregiver)")
    print("=" * 64)
    print("  Mock Mobile App — Fall Detection")
    print(f"  Inference server:     {os.getenv('INFERENCE_SERVER_URL', 'http://localhost:8001')}")
    print(f"  Patients:             {PATIENT_IDS or '(none configured)'}")
    print(f"  MAC map:              {MAC_MAP or '(none)'}")
    print(f"  Poll interval:        {POLL_INTERVAL_SECONDS}s")
    print(f"  MQTT broker:          {_mqtt_info}")
    print(f"  Alert topic:          {MQTT_ALERT_TOPIC}/<patient_id>")
    print(f"  Confirmation timeout: {MOCK_PATIENT_RESPONSE_TIMEOUT}s (simulated patient response)")
    print()
    print("  FLOW: InfluxDB → /predict → [wait for patient] → MQTT alert → caregiver")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    _banner()

    if not PATIENT_IDS:
        logger.error("PATIENT_IDS is not set in .env — nothing to poll.")
        sys.exit(1)

    mqtt_publisher = _create_mqtt_publisher()

    def _graceful(*_):
        logger.info("Shutting down mock app...")
        if mqtt_publisher is not None:
            try:
                mqtt_publisher.loop_stop()
                mqtt_publisher.disconnect()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    inference = InferenceServerClient()
    poller    = MockAppPoller(
        inference_client     = inference,
        patient_ids          = PATIENT_IDS,
        mqtt_publisher       = mqtt_publisher,
        mac_map              = MAC_MAP,
        poll_interval        = POLL_INTERVAL_SECONDS,
        lookback_seconds     = POLL_LOOKBACK_SECONDS,
        alert_topic          = MQTT_ALERT_TOPIC,
        confirmation_timeout = MOCK_PATIENT_RESPONSE_TIMEOUT,
    )
    poller.start()
    poller.join()


if __name__ == "__main__":
    main()
