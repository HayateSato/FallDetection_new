"""
Optional webhook channel for emergency notifications.

When WEBHOOK_URL is set in the environment, fall events are also POSTed
to that URL. This enables integration with:
  - PagerDuty / OpsGenie
  - Custom SMS gateways
  - Hospital notification systems
"""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)
WEBHOOK_URL: Optional[str] = os.environ.get("WEBHOOK_URL", "").strip() or None


async def send_webhook(event: dict) -> None:
    """POST fall event to configured webhook URL. Never raises — errors are logged."""
    if not WEBHOOK_URL:
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(WEBHOOK_URL, json=event)
            resp.raise_for_status()
            logger.info(f"Webhook sent: {WEBHOOK_URL} → {resp.status_code}")
    except Exception as e:
        logger.error(f"Webhook failed (non-fatal): {e}")
