"""
server_health — admin-only dashboard for plain-language service status.

Runs probes against:
  inference_server, fall_dashboard, postgres, mqtt_broker, mlflow, minio

Returns one consolidated status:
  - healthy  → all 6 services are up
  - degraded → some are up, none are clearly down
  - down     → at least one service is unreachable

Endpoint:
  GET  /            → dashboard index.html
  GET  /api/status  → JSON status payload (used by the page + monitoring)

Auth: NOT yet enforced — see todo.md Step 11.5.4 (JWT/role gate is shared with
ml_dashboard once that work lands). Until then, do not expose beyond localhost
or cluster-internal.
"""

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server_health.checks import run_all

logger = logging.getLogger(__name__)
DASHBOARD_DIR = Path(__file__).parent / "dashboard"

app = FastAPI(title="server_health — admin status dashboard", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/status")
async def api_status():
    return await run_all()


# Static dashboard files mounted last so /api/status takes precedence
if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")
else:
    logger.warning(f"Dashboard directory not found at {DASHBOARD_DIR}")
