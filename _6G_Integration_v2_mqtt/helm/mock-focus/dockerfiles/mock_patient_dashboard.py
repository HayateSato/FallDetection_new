"""
Minimal mock Patient Dashboard.

Serves a single HTML page that:
  - Polls the cross-namespace fall_dashboard `/api/patients` to list patients
  - Connects to the cross-namespace fall_dashboard `/api/stream` for live alerts
  - Flags any patient with an active fall alert in red

Purpose: prove that cross-namespace DNS + HTTP + SSE all work end-to-end
in a real Kubernetes cluster, BEFORE handing the chart to FOCUS DevOps.

The FALL_DASHBOARD_URL env var is set by the Helm template to the FQDN
of the fall_dashboard service in the OTHER namespace.
"""

import os
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

FALL_DASHBOARD_URL = os.getenv("FALL_DASHBOARD_URL", "http://localhost:8002")

app = FastAPI(title="mock-patient-dashboard (FOCUS simulation)")
HTML = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "fall_dashboard_url": FALL_DASHBOARD_URL}


@app.get("/", response_class=HTMLResponse)
def index():
    # Inject the FQDN into the page so the browser can reach our fall_dashboard.
    # The browser is OUTSIDE the cluster, so it sees the URL via NodePort/Ingress
    # rather than the internal service DNS. For dry-run we proxy instead.
    return HTML


# ---------------------------------------------------------------------------
# Browser-side calls go through this server (server-side proxy) so the browser
# does not need DNS access to the in-cluster service. This proves cross-namespace
# pod→pod traffic works.
# ---------------------------------------------------------------------------

@app.get("/proxy/patients")
async def proxy_patients():
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{FALL_DASHBOARD_URL}/api/patients")
        return JSONResponse(content=r.json(), status_code=r.status_code)


@app.get("/proxy/falls")
async def proxy_falls():
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{FALL_DASHBOARD_URL}/api/falls?limit=2000")
        return JSONResponse(content=r.json(), status_code=r.status_code)


@app.get("/proxy/stream")
async def proxy_stream():
    """
    Stream the fall_dashboard SSE feed through this server so the browser can
    consume it without needing DNS access to the cross-namespace service.
    Validates that SSE works across namespaces.
    """
    from fastapi.responses import StreamingResponse

    async def gen():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", f"{FALL_DASHBOARD_URL}/api/stream") as r:
                async for chunk in r.aiter_text():
                    yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream")
