"""
Per-service health probes.

Each check returns a dict with the same shape:
    {
        "name":    str,         # the service name (used as a UI label)
        "status":  str,         # "healthy" | "degraded" | "down"
        "url":     str | None,  # endpoint we probed
        "details": str,         # plain-language note for the admin
        "latency_ms": int       # time the probe took
    }

Errors never raise — they are translated into a "down" result. The dashboard
must always render even when half the stack is offline.
"""

import asyncio
import os
import socket
import time
from typing import Optional

import httpx
from sqlalchemy import create_engine, text


HTTP_TIMEOUT = 3.0


def _result(name: str, status: str, url: Optional[str], details: str, t_start: float) -> dict:
    return {
        "name":       name,
        "status":     status,
        "url":        url,
        "details":    details,
        "latency_ms": int((time.perf_counter() - t_start) * 1000),
    }


# ---------------------------------------------------------------------------
# HTTP probes — parse the response body when it carries useful info
# ---------------------------------------------------------------------------

async def check_inference_server() -> dict:
    name = "inference_server"
    url  = os.getenv("INFERENCE_SERVER_URL", "http://localhost:8001") + "/health"
    t0   = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.get(url)
        if r.status_code != 200:
            return _result(name, "down", url, f"HTTP {r.status_code}", t0)
        body  = r.json()
        model = body.get("model_version", "?")
        up    = body.get("uptime_seconds")
        up_s  = f"{int(up)}s" if up is not None else "—"
        return _result(name, "healthy", url, f"model={model}, uptime={up_s}", t0)
    except Exception as exc:
        return _result(name, "down", url, f"unreachable ({exc.__class__.__name__})", t0)


async def check_fall_dashboard() -> dict:
    name = "fall_dashboard"
    base = os.getenv("FALL_DASHBOARD_URL", "http://localhost:8002")
    url  = base + "/api/patients"
    t0   = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.get(url)
        if r.status_code != 200:
            return _result(name, "down", url, f"HTTP {r.status_code}", t0)
        n = len((r.json() or {}).get("patients", []))
        return _result(name, "healthy", url, f"{n} registered patient(s)", t0)
    except Exception as exc:
        return _result(name, "down", url, f"unreachable ({exc.__class__.__name__})", t0)


async def check_mlflow() -> dict:
    name = "mlflow"
    base = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if not base.startswith("http"):
        # SQLite mode — no tracking server to probe
        return _result(name, "degraded", None,
                       "file-based SQLite tracking (no server)", time.perf_counter())
    url = base + "/health"
    t0  = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.get(url)
        if r.status_code != 200:
            return _result(name, "down", url, f"HTTP {r.status_code}", t0)
        return _result(name, "healthy", url, r.text.strip()[:80] or "ok", t0)
    except Exception as exc:
        return _result(name, "down", url, f"unreachable ({exc.__class__.__name__})", t0)


async def check_minio() -> dict:
    name = "minio"
    base = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    url  = base + "/minio/health/live"
    t0   = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.get(url)
        if r.status_code != 200:
            return _result(name, "down", url, f"HTTP {r.status_code}", t0)
        return _result(name, "healthy", url, "S3 endpoint live", t0)
    except Exception as exc:
        return _result(name, "down", url, f"unreachable ({exc.__class__.__name__})", t0)


# ---------------------------------------------------------------------------
# TCP probe (MQTT broker has no HTTP health endpoint)
# ---------------------------------------------------------------------------

async def check_mqtt_broker() -> dict:
    name = "mqtt_broker"
    host = os.getenv("MQTT_BROKER_HOST", "localhost")
    port = int(os.getenv("MQTT_BROKER_PORT", "1883"))
    url  = f"tcp://{host}:{port}"
    t0   = time.perf_counter()
    try:
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: socket.create_connection((host, port), timeout=HTTP_TIMEOUT).close(),
        )
        return _result(name, "healthy", url, "TCP port reachable", t0)
    except Exception as exc:
        return _result(name, "down", url, f"connection failed ({exc.__class__.__name__})", t0)


# ---------------------------------------------------------------------------
# Postgres — a real SQL round-trip
# ---------------------------------------------------------------------------

async def check_postgres() -> dict:
    name = "postgres"
    url  = os.getenv("DATABASE_URL", "")
    safe_url = url.split("@")[-1] if "@" in url else url   # strip credentials for display
    t0   = time.perf_counter()

    def _sql_probe():
        engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    try:
        await asyncio.get_running_loop().run_in_executor(None, _sql_probe)
        return _result(name, "healthy", safe_url, "SELECT 1 succeeded", t0)
    except Exception as exc:
        return _result(name, "down", safe_url, f"connection failed ({exc.__class__.__name__})", t0)


# ---------------------------------------------------------------------------
# Run all in parallel
# ---------------------------------------------------------------------------

CHECKS = [
    check_inference_server,
    check_fall_dashboard,
    check_postgres,
    check_mqtt_broker,
    check_mlflow,
    check_minio,
]


async def run_all() -> dict:
    results = await asyncio.gather(*[c() for c in CHECKS], return_exceptions=False)

    statuses = [r["status"] for r in results]
    if all(s == "healthy" for s in statuses):
        overall = "healthy"
    elif any(s == "down" for s in statuses):
        overall = "down"
    else:
        overall = "degraded"

    return {"overall": overall, "services": results}
