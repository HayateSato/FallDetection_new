"""
MQTT subscriber — listens for confirmed fall alerts from the mobile app
and forwards them to in-process SSE subscribers (the caregiver dashboard).

Topic structure:
  fall/events/<patient_id>  — inference server publishes (fall detected)
                              NOT subscribed here — that's the mobile app's concern
  fall/alert/<patient_id>   — mobile app publishes AFTER patient confirmation/timeout
                              THIS listener subscribes to fall/alert/#

The caregiver dashboard only receives events that have passed through the
patient confirmation step. Raw fall detections (fall/events) are handled
by the mobile app, which decides whether to escalate.
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
MQTT_ALERT_TOPIC = os.getenv("MQTT_ALERT_TOPIC", "fall/alert")  # confirmed alerts from mobile app
MQTT_USERNAME    = os.getenv("MQTT_USERNAME", "").strip()
MQTT_PASSWORD    = os.getenv("MQTT_PASSWORD", "").strip()


class FallEventBroker:
    """
    In-process pub/sub bridge between MQTT and dashboard SSE clients.
    Keeps a set of asyncio.Queue subscribers; each /api/stream connection
    registers a queue and gets every fall event pushed onto it.

    on_fall (optional):
        Sync callback called from paho's thread on each confirmed alert BEFORE
        fan-out to SSE clients.  Signature: (event: dict) -> None.
        Used by the caregiver client to write the fall to the local DB.
        The patient confirmation step already happened in the mobile app before
        this alert was published — no auto-confirm timer needed here.
        If not set, events are forwarded to SSE only (no DB write).
    """

    def __init__(self) -> None:
        self._subscribers: Set[asyncio.Queue] = set()
        self._client: Optional[object] = None   # paho mqtt.Client
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.on_fall = None   # set by client.py before broker.start()

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
                        f"  subscribing to {MQTT_ALERT_TOPIC}/#")
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
            topic = f"{MQTT_ALERT_TOPIC}/#"
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

        If on_fall is set (caregiver mode):
          - on_fall() handles DB write, auto-confirm timer, AND publish_local()
        Otherwise:
          - publish_local() is scheduled directly (SSE fan-out only, no DB write)
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

        if self.on_fall is not None:
            # on_fall is responsible for publish_local + DB + timer
            try:
                self.on_fall(event)
            except Exception as exc:
                logger.warning(f"on_fall callback raised: {exc}")
        else:
            asyncio.run_coroutine_threadsafe(self.publish_local(event), self._loop)
