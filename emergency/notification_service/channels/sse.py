"""
SSE Connection Manager for the emergency notification service.

Each connected tablet/browser gets its own asyncio.Queue.
When a fall event arrives from Redis, it is broadcast to all active queues.
"""

import asyncio
from typing import List


class ConnectionManager:
    """Manages a set of active SSE client connections."""

    def __init__(self):
        self._queues: List[asyncio.Queue] = []

    def connect(self) -> asyncio.Queue:
        """Register a new SSE client. Returns its dedicated queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._queues.append(q)
        return q

    def disconnect(self, q: asyncio.Queue) -> None:
        """Remove a client queue when they disconnect."""
        try:
            self._queues.remove(q)
        except ValueError:
            pass

    async def broadcast(self, message: dict) -> None:
        """Push a message to all connected clients."""
        for q in list(self._queues):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                pass  # slow client — skip rather than block

    @property
    def connection_count(self) -> int:
        return len(self._queues)


# Singleton instance used by server.py
manager = ConnectionManager()
