"""
server_health entry point.

Run from _6G_Integration_v2_mqtt/ as cwd:

    python -m server_health.main

Default port 8006 (8004 = ml_dashboard, 8005 = mock_app patient popup).
"""

import logging
import os
import signal
import sys

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from server_health.web import app

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("server_health")

HOST = os.getenv("SERVER_HEALTH_HOST", "0.0.0.0")
PORT = int(os.getenv("SERVER_HEALTH_PORT", "8006"))


def _banner() -> None:
    print("=" * 64)
    print("  server_health — admin status dashboard")
    print(f"  Web UI:  http://localhost:{PORT}/")
    print()
    print("  WARNING: auth not yet enforced — local / cluster-internal use only.")
    print("=" * 64)


def main() -> None:
    _banner()

    def _graceful(*_):
        logger.info("Shutting down server_health...")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    uvicorn.run(app, host=HOST, port=PORT, workers=1, log_level="info")


if __name__ == "__main__":
    main()
