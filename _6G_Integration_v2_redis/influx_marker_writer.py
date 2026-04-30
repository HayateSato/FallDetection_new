"""
InfluxDB Fall Marker Writer — Redis Subscriber
================================================

Standalone script that subscribes to the Redis `fall_events` channel and
writes a `fall_marker=1` point to InfluxDB whenever a fall is detected.

The marker is written to the same measurement (SMART_DATA) and tagged with
the same macAddress so it appears alongside the sensor data in InfluxDB
queries and dashboards.

Usage:
    python influx_marker_writer.py

Requires REDIS_URL, INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET
to be set in .env (same file as the other components).
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
REDIS_URL      = os.getenv("REDIS_URL", "").strip()
REDIS_CHANNEL  = os.getenv("REDIS_FALL_CHANNEL", "fall_events")
INFLUXDB_URL   = os.getenv("INFLUXDB_URL", "")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "")
INFLUXDB_ORG   = os.getenv("INFLUXDB_ORG", "")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "")

if not REDIS_URL:
    logger.error("REDIS_URL is not set in .env — cannot subscribe.")
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
    mac_id      = event.get("mac_id") or event.get("device_id") or ""
    patient_id  = event.get("patient_id", "unknown")
    confidence  = event.get("confidence", 0.0)
    timestamp   = event.get("timestamp") or event.get("detection_time")

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
# Redis subscriber loop
# ---------------------------------------------------------------------------
def main() -> None:
    import redis

    logger.info("=" * 60)
    logger.info("  InfluxDB Fall Marker Writer")
    logger.info(f"  Redis:    {REDIS_URL}  channel={REDIS_CHANNEL}")
    logger.info(f"  InfluxDB: {INFLUXDB_URL}  bucket={INFLUXDB_BUCKET}")
    logger.info("=" * 60)

    r = redis.from_url(REDIS_URL, decode_responses=True)
    pubsub = r.pubsub()
    pubsub.subscribe(REDIS_CHANNEL)
    logger.info(f"Subscribed to '{REDIS_CHANNEL}' — waiting for fall events...")

    try:
        for msg in pubsub.listen():
            if msg.get("type") != "message":
                continue
            try:
                event = json.loads(msg["data"])
            except Exception:
                logger.warning(f"Bad message: {msg['data']!r}")
                continue

            if not event.get("fall_detected", False):
                continue

            logger.info(f"Fall event received: patient={event.get('patient_id')}  "
                        f"confidence={event.get('confidence')}")
            write_fall_marker(event)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        pubsub.close()
        r.close()
        _influx_client.close()


if __name__ == "__main__":
    main()
