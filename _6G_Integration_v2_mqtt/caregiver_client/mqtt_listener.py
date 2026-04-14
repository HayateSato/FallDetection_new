"""
MQTT subscriber — listens on the inference server's fall topic and forwards
each message to in-process SSE subscribers (the dashboard).

The poller already writes fall_history rows directly when /predict returns.
MQTT is used purely as the live notification path: as soon as the server
detects a fall it publishes; this listener picks it up and pushes the event
to every connected dashboard browser via Server-Sent Events.

Why both paths (DB write from poller AND MQTT live push)?
  - DB write happens whenever the response comes back from /predict (durable history)
  - MQTT push is the real-time channel — decoupled from the HTTP roundtrip,
    so future producers (e.g. the mobile app) can also feed the dashboard

Topic structure:
  MQTT_FALL_TOPIC (default: fall/events) acts as a prefix.
  The inference server publishes to  fall/events/<patient_id>
  This listener subscribes to        fall/events/#   (wildcard — all patients)
"""

import asyncio
import json
import logging
import os
from typing import Optional, Set

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

logger = logging.getLogger(__name__)

MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "").strip()
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_FALL_TOPIC  = os.getenv("MQTT_FALL_TOPIC", "fall/events")
MQTT_USERNAME    = os.getenv("MQTT_USERNAME", "").strip()
MQTT_PASSWORD    = os.getenv("MQTT_PASSWORD", "").strip()


class FallEventBroker:
    """
    In-process pub/sub bridge between MQTT and dashboard SSE clients.
    Keeps a set of asyncio.Queue subscribers; each /api/stream connection
    registers a queue and gets every fall event pushed onto it.

    Public interface is identical to the previous Redis-based FallEventBroker
    so web.py requires no changes beyond the import.
    """

    def __init__(self) -> None:
        self._subscribers: Set[asyncio.Queue] = set()
        self._client: Optional[object] = None   # paho mqtt.Client
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    async def start(self) -> None:
        if not MQTT_BROKER_HOST:
            logger.info("MQTT_BROKER_HOST not set — fall events will not stream live.")
            return
        if mqtt is None:
            logger.warning("paho-mqtt package not installed — `pip install paho-mqtt`.")
            return

        # Capture the running event loop so the paho callback can schedule
        # coroutines back onto it from its background thread.
        self._loop = asyncio.get_running_loop()

        client = mqtt.Client(client_id="fall-detection-caregiver")
        if MQTT_USERNAME:
            client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD or None)

        client.on_connect    = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message    = self._on_message

        try:
            client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
            client.loop_start()  # paho background thread
            self._client = client
            logger.info(f"FallEventBroker connecting to MQTT "
                        f"{MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}"
                        f"  subscribing to {MQTT_FALL_TOPIC}/#")
        except Exception as exc:
            logger.warning(f"MQTT broker connection failed ({exc}) — "
                           "live dashboard alerts disabled.")
            self._client = None

    async def stop(self) -> None:
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._client = None

    # ------------------------------------------------------------------
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    # ------------------------------------------------------------------
    async def publish_local(self, event: dict) -> None:
        """Direct fan-out to all connected SSE clients."""
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("SSE subscriber queue full — dropping event")

    # ------------------------------------------------------------------
    # paho callbacks (called from paho's background thread)
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            topic = f"{MQTT_FALL_TOPIC}/#"
            client.subscribe(topic)
            logger.info(f"MQTT connected — subscribed to '{topic}'")
        else:
            logger.warning(f"MQTT connect failed  rc={rc}")

    def _on_disconnect(self, client, userdata, rc) -> None:
        if rc != 0:
            logger.warning(f"MQTT disconnected unexpectedly  rc={rc} — paho will auto-reconnect")

    def _on_message(self, client, userdata, msg) -> None:
        """
        Paho calls this in its own background thread.
        Parse the payload and schedule publish_local() on the asyncio event loop.
        """
        try:
            payload = msg.payload.decode("utf-8")
            event   = json.loads(payload)
        except Exception as exc:
            logger.warning(f"Could not parse MQTT message on {msg.topic!r}: {exc}")
            return

        if not event.get("fall_detected", False):
            return

        if self._loop is None or not self._loop.is_running():
            logger.warning("Event loop not available — cannot forward MQTT event to SSE")
            return

        asyncio.run_coroutine_threadsafe(self.publish_local(event), self._loop)
