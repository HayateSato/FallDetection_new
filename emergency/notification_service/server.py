"""
Emergency Notification Service

Subscribes to the Redis 'fall_events' channel and fans out fall alerts to:
  1. All connected SSE clients (emergency tablets via browser EventSource)
  2. Optional webhook (hospital system, pager service)

Endpoints:
    GET  /health   — health check
    GET  /stream   — SSE stream; each connected tablet receives fall events

Run from project root:
    uvicorn emergency.notification_service.server:app --host 0.0.0.0 --port 8003
"""

import asyncio
import json
import logging
import os
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from emergency.notification_service.channels.sse import manager
from emergency.notification_service.channels.webhook import send_webhook

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Emergency Notification Service",
    description="Fans out fall events to tablets and optional webhooks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app)
except ImportError:
    pass


@app.on_event("startup")
async def start_redis_listener():
    """Start a background task that continuously reads from Redis and broadcasts."""
    asyncio.create_task(_redis_listener())


async def _redis_listener() -> None:
    """
    Continuously subscribe to Redis fall_events channel.
    On each message: broadcast to SSE clients and send webhook.
    Auto-reconnects if Redis connection drops.
    """
    from shared.redis_client import subscribe_fall_events
    logger.info("Redis listener started — subscribed to 'fall_events'")
    while True:
        try:
            async for event in subscribe_fall_events():
                logger.info(
                    f"Fall event received: patient={event.get('patient_id')}, "
                    f"fall={event.get('fall_detected')}, conf={event.get('confidence')}"
                )
                await manager.broadcast(event)
                if event.get("fall_detected"):
                    await send_webhook(event)
        except Exception as e:
            logger.error(f"Redis listener error: {e} — reconnecting in 5s")
            await asyncio.sleep(5)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "emergency_notification_service",
        "connected_tablets": manager.connection_count,
    }


@app.get("/stream")
async def event_stream():
    """
    SSE endpoint for emergency tablets.

    Browser usage:
        const es = new EventSource("/api/emergency/stream");
        es.onmessage = (e) => {
            const event = JSON.parse(e.data);
            if (event.fall_detected) showAlert(event);
        };

    The EventSource API auto-reconnects on disconnect — no manual retry needed.
    """
    q = manager.connect()

    async def generate() -> AsyncGenerator[str, None]:
        try:
            while True:
                try:
                    # Wait for next event with a periodic keep-alive ping
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send a comment keep-alive so nginx doesn't close idle connections
                    yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            manager.disconnect(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    port = int(os.getenv("EMERGENCY_PORT", "8003"))
    uvicorn.run(app, host="0.0.0.0", port=port)
