"""
Redis subscriber — listens on the inference server's `fall_events` channel
and forwards each message to in-process SSE subscribers (the dashboard).

The poller already writes fall_history rows directly when /predict returns.
Redis is used purely as the live notification path: as soon as the server
detects a fall it publishes; this listener picks it up and pushes the event
to every connected dashboard browser via Server-Sent Events.

Why both paths (DB write from poller AND Redis live push)?
  - DB write happens whenever the response comes back from /predict (durable history)
  - Redis push is the real-time channel — decoupled from the HTTP roundtrip,
    so future producers (e.g. a separate trigger client) can also feed the dashboard
"""

import asyncio
import json
import logging
import os
from typing import Optional, Set

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

logger = logging.getLogger(__name__)

REDIS_URL          = os.getenv("REDIS_URL", "").strip()
REDIS_FALL_CHANNEL = os.getenv("REDIS_FALL_CHANNEL", "fall_events")


class FallEventBroker:
    """
    In-process pub/sub bridge between Redis and dashboard SSE clients.
    Keeps a set of asyncio.Queue subscribers; each /api/stream connection
    registers a queue and gets every fall event pushed onto it.
    """

    def __init__(self) -> None:
        self._subscribers: Set[asyncio.Queue] = set()
        self._task: Optional[asyncio.Task] = None
        self._stop_flag = False

    # ------------------------------------------------------------------
    async def start(self) -> None:
        if not REDIS_URL:
            logger.info("REDIS_URL not set — fall events will not stream live.")
            return
        if aioredis is None:
            logger.warning("redis package not installed — `pip install redis`.")
            return
        self._task = asyncio.create_task(self._listen_loop())
        logger.info(f"FallEventBroker subscribing to {REDIS_URL} channel={REDIS_FALL_CHANNEL}")

    async def stop(self) -> None:
        self._stop_flag = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    # ------------------------------------------------------------------
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    # ------------------------------------------------------------------
    async def publish_local(self, event: dict) -> None:
        """Direct fan-out (used when bypassing Redis, e.g. local fallback)."""
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("SSE subscriber queue full — dropping event")

    # ------------------------------------------------------------------
    async def _listen_loop(self) -> None:
        backoff = 1
        while not self._stop_flag:
            try:
                client = aioredis.from_url(REDIS_URL, decode_responses=True)
                pubsub = client.pubsub()
                await pubsub.subscribe(REDIS_FALL_CHANNEL)
                logger.info(f"Connected to Redis channel '{REDIS_FALL_CHANNEL}'")
                backoff = 1

                async for msg in pubsub.listen():
                    if self._stop_flag:
                        break
                    if msg.get("type") != "message":
                        continue
                    data = msg.get("data")
                    try:
                        event = json.loads(data) if isinstance(data, str) else data
                    except Exception:
                        logger.warning(f"Could not parse fall_events payload: {data!r}")
                        continue
                    await self.publish_local(event)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"Redis listener error ({exc}) — retrying in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
            finally:
                try:
                    await pubsub.close()
                    await client.close()
                except Exception:
                    pass
