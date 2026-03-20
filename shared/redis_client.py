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
FALL_EVENTS_CHANNEL = "fall_events"


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


async def subscribe_fall_events():
    """
    Async generator that yields decoded fall event dicts from the Redis channel.
    Filters out 'subscribe' confirmation messages automatically.

    Usage:
        async for event in subscribe_fall_events():
            print(event["patient_id"], event["fall_detected"])
    """
    import json
    import redis.asyncio as aioredis

    client = aioredis.from_url(REDIS_URL, decode_responses=True)
    pubsub = client.pubsub()
    await pubsub.subscribe(FALL_EVENTS_CHANNEL)
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                yield json.loads(message["data"])
            except (ValueError, KeyError):
                continue
    finally:
        await pubsub.unsubscribe(FALL_EVENTS_CHANNEL)
        await client.aclose()
