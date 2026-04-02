"""
Fall Detection — Trigger Client (Integration Build)
====================================================
Stripped-down InfluxDB polling loop that sends sensor data to the
inference server and fires a callback on every detection cycle.

All UI, recording state, CSV export, and manual truth marker code
has been removed.  Only the core data-fetch → POST /predict loop remains.

Usage
-----
    from _EcoSystem_Integration.trigger_client.client import FallDetectionClient

    def on_result(result: dict) -> None:
        if result["fall_detected"]:
            print(f"FALL  confidence={result['confidence']:.2%}")
        # result also contains: model_version, timestamp, window_size, features

    client = FallDetectionClient(on_result=on_result)
    client.start()          # blocks — Ctrl+C to stop
    # or:
    client.start_async()    # starts background thread, returns immediately
    ...
    client.stop()

Run as script (from project root):
    python -m _EcoSystem_Integration.trigger_client.client
"""

import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Project-root imports
# ---------------------------------------------------------------------------
from app.data_input.data_loader.influx_data_fetcher import fetch_and_preprocess_sensor_data
from config.settings import (
    ACC_SAMPLE_RATE,
    HARDWARE_ACC_SAMPLE_RATE,
    MONITORING_INTERVAL_SECONDS,
    MONITORING_LOOKBACK_SECONDS,
    REMOTE_API_KEY,
    REMOTE_SERVER_URL,
    WINDOW_SIZE_SECONDS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FallDetectionClient:
    """
    Polls InfluxDB every `interval_seconds`, sends data to the inference
    server, and invokes `on_result` with the server response.

    Parameters
    ----------
    on_result       : callable receiving a dict — see _run_cycle() for keys
    server_url      : base URL of the inference server  (env: REMOTE_SERVER_URL)
    api_key         : X-API-Key header value            (env: REMOTE_API_KEY)
    interval_seconds: how often to query InfluxDB and run inference
    lookback_seconds: how far back to fetch from InfluxDB
    window_seconds  : detection window sent to server (overrides server default)
    hardware_rate_hz: hardware ACC sample rate in Hz
    participant     : optional subject/device ID forwarded to the server
    """

    def __init__(
        self,
        on_result: Optional[Callable[[dict], None]] = None,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        interval_seconds: int = MONITORING_INTERVAL_SECONDS,
        lookback_seconds: int = MONITORING_LOOKBACK_SECONDS,
        window_seconds: Optional[float] = None,
        hardware_rate_hz: int = HARDWARE_ACC_SAMPLE_RATE,
        participant: Optional[str] = None,
    ):
        self.on_result        = on_result or _default_result_handler
        self.server_url       = (server_url or REMOTE_SERVER_URL).rstrip("/")
        self.api_key          = api_key or REMOTE_API_KEY
        self.interval_seconds = interval_seconds
        self.lookback_seconds = lookback_seconds
        self.window_seconds   = window_seconds
        self.hardware_rate_hz = hardware_rate_hz
        self.participant      = participant

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Derive minimum required samples from the configured window
        _ws = window_seconds or WINDOW_SIZE_SECONDS
        self._required_acc_samples = int(_ws * ACC_SAMPLE_RATE)

        # Fetch model info once on startup to know whether barometer is needed
        self._uses_barometer = self._fetch_uses_barometer()

        if not self.server_url:
            raise ValueError("REMOTE_SERVER_URL must be set in .env or passed as server_url=")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start monitoring loop in the current thread (blocking)."""
        logger.info(f"Starting fall detection client  server={self.server_url}")
        self._run_loop()

    def start_async(self) -> None:
        """Start monitoring loop in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Client already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"Client started (background thread)  server={self.server_url}")

    def stop(self) -> None:
        """Signal the monitoring loop to stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval_seconds + 5)
        logger.info("Client stopped")

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._run_cycle()
            except Exception as exc:
                logger.error(f"Cycle error: {exc}", exc_info=True)
            self._stop_event.wait(timeout=self.interval_seconds)

    def _run_cycle(self) -> None:
        """One fetch → preprocess → POST /predict → callback iteration."""
        timestamp = datetime.now(timezone.utc)

        # 1. Fetch from InfluxDB
        acc_data, acc_time, pressure, pressure_time, _ = fetch_and_preprocess_sensor_data(
            uses_barometer=self._uses_barometer,
            lookback_seconds=self.lookback_seconds,
        )

        if acc_data is None or acc_data.shape[1] < self._required_acc_samples:
            got = acc_data.shape[1] if acc_data is not None else 0
            logger.warning(f"Insufficient data: {got} samples (need {self._required_acc_samples})")
            return

        logger.info(f"Fetched {acc_data.shape[1]} ACC samples  "
                    f"({(acc_time[-1]-acc_time[0])/1000:.1f}s span)")

        # 2. Build JSON payload
        payload: dict = {
            "acc_x":        acc_data[0].tolist(),
            "acc_y":        acc_data[1].tolist(),
            "acc_z":        acc_data[2].tolist(),
            "timestamps_ms": acc_time.tolist(),
            "acc_unit":     "lsb",
            "sample_rate":  float(self.hardware_rate_hz),
        }
        if self.participant:
            payload["participant"] = self.participant
        if pressure is not None and len(pressure) > 0:
            payload["pressure"]               = pressure.tolist()
            payload["pressure_timestamps_ms"] = pressure_time.tolist()
        if self.window_seconds is not None:
            # Tell the server which window size to use for this request
            # (the server's POST /config can be used to change it permanently)
            payload["window_seconds"] = self.window_seconds

        # 3. POST to inference server
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        resp = requests.post(
            f"{self.server_url}/predict",
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        server_result = resp.json()

        # 4. Augment with client-side metadata and fire callback
        result = {
            "fall_detected": server_result["fall_detected"],
            "confidence":    server_result["confidence"],
            "threshold":     server_result["threshold"],
            "result":        server_result["result"],
            "model_version": server_result["model_version"],
            "window_size":   server_result["window_size"],
            "features":      server_result.get("features", {}),
            "timestamp":     timestamp.isoformat(),
            "participant":   self.participant,
        }

        self.on_result(result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_uses_barometer(self) -> bool:
        """Ask the server whether the loaded model needs barometer data."""
        if not self.server_url:
            return True
        try:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            resp = requests.get(f"{self.server_url}/model/info",
                                headers=headers, timeout=10)
            if resp.ok:
                return bool(resp.json().get("uses_barometer", True))
        except Exception as exc:
            logger.warning(f"Could not fetch model info ({exc}) — assuming barometer enabled")
        return True


# ---------------------------------------------------------------------------
# Default result handler (used when no callback is supplied)
# ---------------------------------------------------------------------------

def _default_result_handler(result: dict) -> None:
    ts = result["timestamp"]
    conf = result["confidence"]
    if result["fall_detected"]:
        logger.warning(f"[{ts}] *** FALL DETECTED ***  confidence={conf:.2%}  ({result['result']})")
    else:
        logger.info(f"[{ts}] No fall  confidence={conf:.2%}  ({result['result']})")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

def _sigint_handler(sig, frame):
    logger.info("Interrupt received — stopping client")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _sigint_handler)

    participant = os.getenv("PARTICIPANT_ID", None)

    client = FallDetectionClient(participant=participant)

    logger.info("Press Ctrl+C to stop")
    client.start()   # blocking
