"""
Redis connection helper for async and sync usage.

The fall event pub/sub channel is named: "fall_events"

Payload format (JSON string):
    {
        "patient_id": "alice",
        "fall_detected": true,
        "confidence": 0.82,
        "model_version": "v3",
        "timestamp": "2026-03-20T12:34:56+00:00",
        "inference_id": 42
    }

Async usage (FastAPI SSE endpoints):
    from shared.redis_client import get_async_redis
    async with get_async_redis() as r:
        pubsub = r.pubsub()
        await pubsub.subscribe("fall_events")
        async for message in pubsub.listen():
            ...

Sync usage (background tasks, non-async context):
    from shared.redis_client import get_sync_redis
    r = get_sync_redis()
    r.publish("fall_events", json_payload)
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
FALL_EVENTS_CHANNEL   = "fall_events"     # emergency alerts — published after patient feedback decision
PATIENT_ALERTS_CHANNEL = "patient_alerts"  # fall popup for the patient dashboard


def get_sync_redis():
    """Return a synchronous Redis client. Caller is responsible for closing it."""
    try:
        import redis
        return redis.from_url(REDIS_URL, decode_responses=True)
    except ImportError:
        raise RuntimeError("redis package not installed. Run: pip install redis")


@asynccontextmanager
async def get_async_redis():
    """Async context manager for an async Redis client."""
    try:
        import redis.asyncio as aioredis
    except ImportError:
        raise RuntimeError("redis package not installed. Run: pip install redis")
    client = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


async def subscribe_channel(channel: str):
    """
    Generic async generator that yields decoded JSON dicts from any Redis channel.
    Used internally by subscribe_fall_events() and subscribe_patient_alerts().
    """
    import json
    import redis.asyncio as aioredis

    client = aioredis.from_url(REDIS_URL, decode_responses=True)
    pubsub = client.pubsub()
    await pubsub.subscribe(channel)
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                yield json.loads(message["data"])
            except (ValueError, KeyError):
                continue
    finally:
        await pubsub.unsubscribe(channel)
        await client.aclose()


async def subscribe_patient_alerts(participant: str = None):
    """
    Async generator yielding patient alert events from the 'patient_alerts' channel.
    If participant is given, only events for that participant are yielded.

    Payload shape:
        {"patient_id": "alice", "fall_detected": true, "confidence": 0.9,
         "model_version": "v3", "timestamp": "...", "inference_id": 42}
    """
    async for event in subscribe_channel(PATIENT_ALERTS_CHANNEL):
        if participant and event.get("patient_id") != participant:
            continue
        yield event


async def subscribe_fall_events():
    """
    Async generator yielding decoded fall event dicts from the 'fall_events' channel.
    These are published only after patient feedback decision:
      - patient did not respond in 12 s  (no_answer)
      - patient confirmed fall AND requested help
    """
    async for event in subscribe_channel(FALL_EVENTS_CHANNEL):
        yield event
