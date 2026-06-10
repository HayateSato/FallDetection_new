"""
ml_dashboard entry point.

Run from _6G_Integration_v2_mqtt/ as working directory:

    python -m ml_dashboard.main

Endpoints (default port 8004):
    GET  /                  → dashboard UI
    GET  /api/status
    GET  /api/versions
    POST /api/retrain
    GET  /api/retrain/{job_id}
    POST /api/promote
    POST /api/switch
"""

import logging
import os
import signal
import sys

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from ml_dashboard.web import app

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("ml_dashboard")


HOST = os.getenv("ML_DASHBOARD_HOST", "0.0.0.0")
PORT = int(os.getenv("ML_DASHBOARD_PORT", "8004"))


def _banner() -> None:
    print("=" * 64)
    print("  ml_dashboard — admin UI for retrain + hot-swap")
    print(f"  Web UI:           http://localhost:{PORT}/")
    print(f"  MLflow tracking:  {os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")
    print(f"  Inference server: {os.getenv('INFERENCE_SERVER_URL', 'http://localhost:8001')}")
    print()
    print("  WARNING: this UI controls the live model. Auth is NOT yet enforced —")
    print("           do not expose beyond localhost / cluster internal until")
    print("           todo.md Step 11.5.4 (auth gate) is implemented.")
    print("=" * 64)


def main() -> None:
    _banner()

    def _graceful(*_):
        logger.info("Shutting down ml_dashboard...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    uvicorn.run(app, host=HOST, port=PORT, workers=1, log_level="info")


if __name__ == "__main__":
    main()
