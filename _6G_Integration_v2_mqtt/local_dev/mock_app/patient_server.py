"""
Patient confirmation server — mock_app component.

Serves a simple web UI to simulate the patient's phone popup.
When mock_app detects a fall, the poller calls notify_fall() which
makes the UI show the alert. The patient clicks Yes/No in the browser.
The poller waits via wait_for_response() and then publishes the MQTT
alert with the actual patient_confirmed value.

In the real system this is replaced by a native popup in the SmarKo
mobile app — this server only exists in local_dev.

Port: MOCK_PATIENT_SERVER_PORT in .env (default 8005)
UI:   http://localhost:8005/
"""

import logging
import threading
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PatientConfirmationServer:
    """
    Thread-safe bridge between the poller thread and the patient's browser.

    Poller usage:
        server.notify_fall(event)          # called when fall_detected=True
        response = server.wait_for_response(timeout=10)
        # response = {"patient_confirmed": "yes"|"no"|"not_answered",
        #              "needs_help": True|False|None}
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8005):
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._pending_fall: Optional[dict] = None
        self._response_ready = threading.Event()
        self._response: Optional[dict] = None
        self._app = self._build_app()

    # ------------------------------------------------------------------
    # Public interface — called from poller thread
    # ------------------------------------------------------------------

    def notify_fall(self, event: dict) -> None:
        """Signal to the browser that a fall needs confirmation."""
        with self._lock:
            self._pending_fall = event
            self._response_ready.clear()
            self._response = None
        logger.info(
            f"PatientServer: awaiting confirmation  patient={event.get('patient_id')}  "
            f"open http://localhost:{self.port}/ to respond"
        )

    def wait_for_response(self, timeout: int) -> dict:
        """
        Block until patient responds via browser OR timeout expires.
        Clears the pending fall state before returning.
        """
        responded = self._response_ready.wait(timeout=timeout)
        with self._lock:
            result = self._response if (responded and self._response) else {
                "patient_confirmed": "not_answered",
                "needs_help": None,
            }
            self._pending_fall = None
            self._response = None
        return result

    # ------------------------------------------------------------------
    # FastAPI app
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Patient Confirmation — mock_app", docs_url=None)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

        class ConfirmRequest(BaseModel):
            patient_confirmed: str          # "yes" or "no"
            needs_help: Optional[bool] = None

        @app.get("/api/fall")
        def get_pending_fall():
            """Browser polls this to know if a fall confirmation is pending."""
            with self._lock:
                if self._pending_fall:
                    return {"pending": True, "event": self._pending_fall}
            return {"pending": False, "event": None}

        @app.post("/api/confirm")
        def confirm(req: ConfirmRequest):
            """Receives patient response from the browser."""
            if req.patient_confirmed not in ("yes", "no"):
                return JSONResponse(
                    status_code=400,
                    content={"error": "patient_confirmed must be 'yes' or 'no'"},
                )
            with self._lock:
                self._response = {
                    "patient_confirmed": req.patient_confirmed,
                    "needs_help": req.needs_help,
                }
                self._response_ready.set()
            logger.info(
                f"PatientServer: response received — "
                f"confirmed={req.patient_confirmed}  needs_help={req.needs_help}"
            )
            return {"ok": True}

        @app.get("/", response_class=HTMLResponse)
        def patient_ui():
            return _PATIENT_HTML

        return app

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start uvicorn in a background daemon thread."""
        config = uvicorn.Config(
            app=self._app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)

        def _run():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(server.serve())

        t = threading.Thread(target=_run, daemon=True, name="PatientServer")
        t.start()
        logger.info(
            f"PatientConfirmationServer started — "
            f"open http://localhost:{self.port}/ to see patient popup"
        )


# ---------------------------------------------------------------------------
# Self-contained patient UI (no separate static files needed)
# ---------------------------------------------------------------------------

_PATIENT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fall Detection — Patient Confirmation</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f172a;
    color: #f1f5f9;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  #app { width: 100%; max-width: 480px; padding: 24px; }

  .idle {
    text-align: center;
    color: #64748b;
    font-size: 18px;
    padding: 48px 0;
  }
  .idle .dot { animation: blink 1.4s infinite; }
  .idle .dot:nth-child(2) { animation-delay: 0.2s; }
  .idle .dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes blink { 0%,80%,100%{opacity:0} 40%{opacity:1} }

  .alert-card {
    background: #1e293b;
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 32px;
    box-shadow: 0 0 40px rgba(239,68,68,0.25);
  }
  .alert-icon { font-size: 48px; text-align: center; margin-bottom: 16px; }
  .alert-title {
    font-size: 28px; font-weight: 700;
    color: #ef4444;
    text-align: center;
    margin-bottom: 8px;
  }
  .alert-patient {
    font-size: 16px; color: #94a3b8;
    text-align: center;
    margin-bottom: 8px;
  }
  .alert-confidence {
    font-size: 14px; color: #64748b;
    text-align: center;
    margin-bottom: 24px;
  }

  .countdown {
    font-size: 48px; font-weight: 700;
    text-align: center;
    color: #f59e0b;
    margin-bottom: 24px;
    font-variant-numeric: tabular-nums;
  }
  .countdown.urgent { color: #ef4444; }

  .question {
    font-size: 20px; font-weight: 600;
    text-align: center;
    margin-bottom: 24px;
    line-height: 1.4;
  }

  .btn-row {
    display: flex;
    gap: 16px;
  }
  .btn {
    flex: 1;
    padding: 18px;
    font-size: 18px; font-weight: 700;
    border: none; border-radius: 12px;
    cursor: pointer;
    transition: transform 0.1s, opacity 0.1s;
  }
  .btn:active { transform: scale(0.97); }
  .btn-yes  { background: #22c55e; color: #fff; }
  .btn-no   { background: #475569; color: #f1f5f9; }
  .btn-help { background: #3b82f6; color: #fff; }

  .status-card {
    background: #1e293b;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
  }
  .status-icon { font-size: 48px; margin-bottom: 16px; }
  .status-title { font-size: 22px; font-weight: 700; margin-bottom: 8px; }
  .status-sub { font-size: 14px; color: #64748b; }
</style>
</head>
<body>
<div id="app">
  <div class="idle" id="idle-view">
    Waiting for fall event
    <span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
  </div>
  <div id="fall-view" style="display:none"></div>
</div>
<script>
  let _countdown = null;
  let _timeLeft  = 0;
  let _step      = 'confirm';   // 'confirm' | 'help' | 'done'
  let _currentEvent = null;

  async function poll() {
    try {
      const r = await fetch('/api/fall');
      const data = await r.json();
      if (data.pending && (!_currentEvent ||
          _currentEvent.observation_id !== data.event.observation_id)) {
        _currentEvent = data.event;
        startConfirmFlow(data.event);
      } else if (!data.pending && _step === 'done') {
        resetIdle();
      }
    } catch(e) {}
  }

  function startConfirmFlow(event) {
    _step = 'confirm';
    clearInterval(_countdown);
    _timeLeft = 10;
    showConfirmCard(event);
    _countdown = setInterval(() => {
      _timeLeft--;
      updateCountdown();
      if (_timeLeft <= 0) {
        clearInterval(_countdown);
        sendConfirm('not_answered', null);
      }
    }, 1000);
  }

  function showConfirmCard(event) {
    const conf = event.confidence ? Math.round(event.confidence * 100) : '—';
    document.getElementById('idle-view').style.display = 'none';
    document.getElementById('fall-view').style.display = 'block';
    document.getElementById('fall-view').innerHTML = `
      <div class="alert-card">
        <div class="alert-icon">⚠️</div>
        <div class="alert-title">Fall Detected</div>
        <div class="alert-patient">Patient: ${event.patient_id || '—'}</div>
        <div class="alert-confidence">Confidence: ${conf}%</div>
        <div class="countdown" id="countdown">${_timeLeft}</div>
        <div class="question">Did you fall?</div>
        <div class="btn-row">
          <button class="btn btn-yes"  onclick="onYes()">Yes, I fell</button>
          <button class="btn btn-no"   onclick="onNo()">No, I'm fine</button>
        </div>
      </div>`;
  }

  function updateCountdown() {
    const el = document.getElementById('countdown');
    if (!el) return;
    el.textContent = _timeLeft;
    el.className = 'countdown' + (_timeLeft <= 3 ? ' urgent' : '');
  }

  function onYes() {
    clearInterval(_countdown);
    _step = 'help';
    document.getElementById('fall-view').innerHTML = `
      <div class="alert-card">
        <div class="alert-icon">🆘</div>
        <div class="question" style="margin-top:8px">Do you need help?</div>
        <div class="btn-row" style="margin-top:24px">
          <button class="btn btn-help" onclick="sendConfirm('yes', true)">Yes, need help</button>
          <button class="btn btn-no"   onclick="sendConfirm('yes', false)">No, I'm okay</button>
        </div>
      </div>`;
  }

  function onNo() {
    clearInterval(_countdown);
    sendConfirm('no', null);
  }

  async function sendConfirm(patient_confirmed, needs_help) {
    _step = 'done';
    try {
      await fetch('/api/confirm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_confirmed, needs_help }),
      });
    } catch(e) {}

    const icons   = { yes: '✅', no: '👍', not_answered: '⏱️' };
    const titles  = { yes: 'Fall confirmed', no: 'No fall — thank you', not_answered: 'No response — alert sent' };
    const subs    = {
      yes: needs_help ? 'Help is on the way.' : 'Alert sent to caregiver.',
      no:  'No alert sent to caregiver.',
      not_answered: 'Caregiver has been notified.',
    };
    document.getElementById('fall-view').innerHTML = `
      <div class="status-card">
        <div class="status-icon">${icons[patient_confirmed]}</div>
        <div class="status-title">${titles[patient_confirmed]}</div>
        <div class="status-sub">${subs[patient_confirmed] || ''}</div>
      </div>`;

    setTimeout(resetIdle, 4000);
  }

  function resetIdle() {
    _step = 'confirm';
    _currentEvent = null;
    document.getElementById('fall-view').style.display = 'none';
    document.getElementById('idle-view').style.display = 'block';
  }

  // Poll every second for a new fall event
  setInterval(poll, 1000);
  poll();
</script>
</body>
</html>
"""
