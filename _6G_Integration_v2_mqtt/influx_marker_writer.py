"""
InfluxDB Fall Marker Writer — MQTT Subscriber
==============================================

Standalone script that subscribes to the MQTT fall topic and writes a
`fall_marker=1` point to InfluxDB whenever a fall is detected.

The marker is written to the same measurement (SMART_DATA) and tagged with
the same macAddress so it appears alongside the sensor data in InfluxDB
queries and dashboards.

Usage:
    python influx_marker_writer.py

Requires MQTT_BROKER_HOST, INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG,
INFLUXDB_BUCKET to be set in .env (same file as the other components).

Topic subscription:
    Subscribes to  <MQTT_FALL_TOPIC>/#  (default: fall/events/#)
    The inference server publishes to   fall/events/<patient_id>
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("influx_marker_writer")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "").strip()
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_FALL_TOPIC  = os.getenv("MQTT_FALL_TOPIC", "fall/events")
MQTT_USERNAME    = os.getenv("MQTT_USERNAME", "").strip()
MQTT_PASSWORD    = os.getenv("MQTT_PASSWORD", "").strip()
INFLUXDB_URL     = os.getenv("INFLUXDB_URL", "")
INFLUXDB_TOKEN   = os.getenv("INFLUXDB_TOKEN", "")
INFLUXDB_ORG     = os.getenv("INFLUXDB_ORG", "")
INFLUXDB_BUCKET  = os.getenv("INFLUXDB_BUCKET", "")

if not MQTT_BROKER_HOST:
    logger.error("MQTT_BROKER_HOST is not set in .env — cannot subscribe.")
    sys.exit(1)
if not INFLUXDB_URL:
    logger.error("INFLUXDB_URL is not set in .env — cannot write markers.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# InfluxDB writer
# ---------------------------------------------------------------------------
_influx_client = InfluxDBClient(
    url=INFLUXDB_URL,
    token=INFLUXDB_TOKEN,
    org=INFLUXDB_ORG,
    timeout=30_000,
    verify_ssl=True,
)
_write_api = _influx_client.write_api(write_options=SYNCHRONOUS)


def write_fall_marker(event: dict) -> None:
    """Write a fall_marker=1 point to InfluxDB."""
    mac_id     = event.get("mac_id") or event.get("device_id") or ""
    patient_id = event.get("patient_id", "unknown")
    confidence = event.get("confidence", 0.0)
    timestamp  = event.get("timestamp") or event.get("detection_time")

    # Parse timestamp
    try:
        if isinstance(timestamp, str):
            ts = datetime.fromisoformat(timestamp)
        else:
            ts = datetime.now(timezone.utc)
    except Exception:
        ts = datetime.now(timezone.utc)

    point = (
        Point("SMART_DATA")
        .tag("macAddress", mac_id)
        .tag("patient_id", patient_id)
        .field("fall_marker", 1)
        .field("fall_confidence", float(confidence))
        .time(ts, WritePrecision.MS)
    )

    try:
        _write_api.write(bucket=INFLUXDB_BUCKET, record=point)
        logger.info(f"Marker written  mac={mac_id}  patient={patient_id}  "
                    f"confidence={confidence}  time={ts.isoformat()}")
    except Exception as exc:
        logger.error(f"Failed to write marker: {exc}")


# ---------------------------------------------------------------------------
# MQTT subscriber
# ---------------------------------------------------------------------------
def main() -> None:
    import paho.mqtt.client as mqtt

    subscribe_topic = f"{MQTT_FALL_TOPIC}/#"

    logger.info("=" * 60)
    logger.info("  InfluxDB Fall Marker Writer")
    logger.info(f"  MQTT:     {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}  topic={subscribe_topic}")
    logger.info(f"  InfluxDB: {INFLUXDB_URL}  bucket={INFLUXDB_BUCKET}")
    logger.info("=" * 60)

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(subscribe_topic)
            logger.info(f"Subscribed to '{subscribe_topic}' — waiting for fall events...")
        else:
            logger.error(f"MQTT connect failed  rc={rc}")

    def on_message(client, userdata, msg):
        try:
            event = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            logger.warning(f"Bad message on {msg.topic!r}: {msg.payload!r}")
            return

        if not event.get("fall_detected", False):
            return

        logger.info(f"Fall event received: patient={event.get('patient_id')}  "
                    f"confidence={event.get('confidence')}")
        write_fall_marker(event)

    client = mqtt.Client(client_id="influx-marker-writer")
    if MQTT_USERNAME:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD or None)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
        client.loop_forever()  # blocks; handles reconnects automatically
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        client.disconnect()
        _influx_client.close()


if __name__ == "__main__":
    main()
