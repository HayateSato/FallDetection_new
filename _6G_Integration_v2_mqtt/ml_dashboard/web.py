"""
ml_dashboard — admin UI for retrain, register, promote, and hot-swap.

Endpoints:
  GET  /                     → dashboard index.html
  GET  /api/status           → current loaded model on inference_server + Production version
  GET  /api/versions         → list of registered model versions with aliases
  POST /api/retrain          → start retrain.retrain as a background subprocess
  GET  /api/retrain/{job_id} → poll job status + accumulated stdout
  POST /api/promote          → set alias on a registered version
  POST /api/switch           → hot-swap inference_server to a registered version

This UI controls the live model serving real patients. In production it MUST
be auth-gated (todo.md Step 11.5.4). The current build runs without auth — do
not expose it beyond localhost / cluster-internal until auth is added.
"""

import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

import httpx
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(__file__).parent / "dashboard"

# ---------------------------------------------------------------------------
# Config (read from .env via main.py's load_dotenv())
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
INFERENCE_SERVER_URL = os.getenv("INFERENCE_SERVER_URL", "http://localhost:8001")
INFERENCE_API_KEY    = os.getenv("INFERENCE_API_KEY", "")
REGISTERED_MODEL     = os.getenv("MLFLOW_REGISTERED_MODEL", "fall-detection-xgboost")
PROJECT_ROOT         = Path(__file__).resolve().parents[1]   # _6G_Integration_v2_mqtt/

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


# ---------------------------------------------------------------------------
# In-memory job registry for retrain subprocesses
# Job state survives only while ml_dashboard is running. Adequate for MVP.
# ---------------------------------------------------------------------------
_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _spawn_retrain(model_version: str, dataset: str, do_register: bool) -> str:
    """Start retrain.retrain as a subprocess; return a job_id for polling."""
    job_id = uuid.uuid4().hex[:8]
    args = [
        sys.executable, "-m", "retrain.retrain",
        "--model-version", model_version,
        "--dataset", dataset,
    ]
    if do_register:
        args.append("--register")

    with _jobs_lock:
        _jobs[job_id] = {
            "status":     "running",
            "started_at": time.time(),
            "args":       " ".join(args),
            "output":     [],
            "exit_code":  None,
        }

    def _run():
        try:
            proc = subprocess.Popen(
                args,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                with _jobs_lock:
                    _jobs[job_id]["output"].append(line.rstrip())
            proc.wait()
            with _jobs_lock:
                _jobs[job_id]["status"]    = "ok" if proc.returncode == 0 else "failed"
                _jobs[job_id]["exit_code"] = proc.returncode
        except Exception as exc:
            with _jobs_lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["output"].append(f"Subprocess crashed: {exc}")

    threading.Thread(target=_run, daemon=True, name=f"Retrain-{job_id}").start()
    return job_id


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="ml_dashboard — admin UI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production (Step 11.5.4)
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic request bodies
# ---------------------------------------------------------------------------

class RetrainRequest(BaseModel):
    model_version: str = "v0"
    dataset:       str = "our_data"
    do_register:   bool = True   # named do_register; 'register' shadows BaseModel.register


class PromoteRequest(BaseModel):
    version: int
    alias:   str = "Production"


class SwitchRequest(BaseModel):
    # one of these must be set
    mlflow_alias: Optional[str] = None    # e.g. "Production"
    version:      Optional[str] = None    # e.g. "v0_retrained"


# ---------------------------------------------------------------------------
# Status — what is currently loaded vs what alias points to
# ---------------------------------------------------------------------------

@app.get("/api/status")
def api_status():
    """Return current loaded model on inference_server + Production version in registry."""
    info = {"inference_server": None, "registry": None, "alias_matches_loaded": None}

    # Currently loaded model on inference_server
    try:
        r = httpx.get(f"{INFERENCE_SERVER_URL}/model/info", timeout=3.0)
        info["inference_server"] = r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as exc:
        info["inference_server"] = {"error": str(exc)}

    # Production-aliased version in the registry
    try:
        mv = mlflow_client.get_model_version_by_alias(REGISTERED_MODEL, "Production")
        info["registry"] = {
            "version":      int(mv.version),
            "run_id":       mv.run_id,
            "creation_ts":  mv.creation_timestamp,
            "tags":         dict(mv.tags) if mv.tags else {},
        }
    except Exception as exc:
        info["registry"] = {"error": str(exc)}

    # Drift warning: alias moved but server not yet swapped
    loaded_as = (info["inference_server"] or {}).get("loaded_as", "")
    prod_ver  = (info["registry"] or {}).get("version")
    if prod_ver is not None and loaded_as:
        info["alias_matches_loaded"] = f"v{prod_ver}" in loaded_as

    return info


@app.get("/api/versions")
def api_versions():
    """List all registered versions of fall-detection-xgboost with their aliases."""
    try:
        versions = mlflow_client.search_model_versions(f"name = '{REGISTERED_MODEL}'")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MLflow unreachable: {exc}")

    out = []
    for v in versions:
        out.append({
            "version":       int(v.version),
            "run_id":        v.run_id,
            "creation_time": v.creation_timestamp,
            "aliases":       list(v.aliases or []),
            "description":   v.description or "",
        })
    out.sort(key=lambda r: r["version"], reverse=True)
    return {"model": REGISTERED_MODEL, "versions": out}


# ---------------------------------------------------------------------------
# Retrain — spawn subprocess and stream output
# ---------------------------------------------------------------------------

@app.post("/api/retrain")
def api_retrain(req: RetrainRequest):
    job_id = _spawn_retrain(req.model_version, req.dataset, req.do_register)
    logger.info(f"Retrain job started  job_id={job_id}  args={req.dict()}")
    return {"job_id": job_id}


@app.get("/api/retrain/{job_id}")
def api_retrain_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job_id not found")
        return {
            "job_id":    job_id,
            "status":    job["status"],
            "exit_code": job["exit_code"],
            "args":      job["args"],
            "output":    job["output"][-500:],   # last 500 lines
            "duration":  round(time.time() - job["started_at"], 1),
        }


# ---------------------------------------------------------------------------
# Promote — set alias on a version
# ---------------------------------------------------------------------------

@app.post("/api/promote")
def api_promote(req: PromoteRequest):
    if not req.alias:
        raise HTTPException(status_code=400, detail="alias is required")
    try:
        mlflow_client.set_registered_model_alias(REGISTERED_MODEL, req.alias, req.version)
        logger.info(f"Promote  alias={req.alias} → version={req.version}")
        return {"ok": True, "alias": req.alias, "version": req.version}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Hot-swap — call inference_server's /model/switch
# ---------------------------------------------------------------------------

@app.post("/api/switch")
def api_switch(req: SwitchRequest):
    if not (req.mlflow_alias or req.version):
        raise HTTPException(status_code=400, detail="must provide mlflow_alias or version")

    body = {}
    if req.mlflow_alias:
        body["mlflow_stage"] = req.mlflow_alias    # server uses key 'mlflow_stage' for alias name
    if req.version:
        body["version"] = req.version

    headers = {"Content-Type": "application/json"}
    if INFERENCE_API_KEY:
        headers["X-API-Key"] = INFERENCE_API_KEY

    try:
        r = httpx.post(
            f"{INFERENCE_SERVER_URL}/model/switch",
            json=body,
            headers=headers,
            timeout=30.0,
        )
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        logger.info(f"Hot-swap requested  body={body}  response={r.text[:200]}")
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"inference_server unreachable: {exc}")


# ---------------------------------------------------------------------------
# Static dashboard files (mounted last so /api/* takes precedence)
# ---------------------------------------------------------------------------

if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")
else:
    logger.warning(f"Dashboard directory not found at {DASHBOARD_DIR}")
