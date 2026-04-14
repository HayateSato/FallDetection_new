"""
Thin synchronous HTTP client that calls the inference server.

  POST {INFERENCE_SERVER_URL}/predict
        Header  X-API-Key: <key>
        Body    {patient_id, device_id, acc_x/y/z, timestamps_ms,
                 pressure?, pressure_timestamps_ms?}

Returns the parsed JSON response (which already includes a FHIR R4 Observation).
The caller is responsible for storing it / publishing to dashboards.
"""

import logging
import os
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


class InferenceServerClient:
    def __init__(
        self,
        server_url: Optional[str] = None,
        api_key:    Optional[str] = None,
        timeout:    int = 15,
    ):
        self.server_url = (server_url or os.getenv("INFERENCE_SERVER_URL", "http://localhost:8001")).rstrip("/")
        self.api_key    = api_key or os.getenv("INFERENCE_API_KEY", "") or os.getenv("API_KEYS", "").split(",")[0]
        self.timeout    = timeout
        self._uses_baro_cache: Optional[bool] = None

    # ------------------------------------------------------------------
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    # ------------------------------------------------------------------
    def model_info(self) -> dict:
        resp = requests.get(f"{self.server_url}/model/info", headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def uses_barometer(self) -> bool:
        """Cached: query /model/info once to learn whether the server's model needs baro."""
        if self._uses_baro_cache is None:
            try:
                self._uses_baro_cache = bool(self.model_info().get("uses_barometer", False))
            except Exception as exc:
                logger.warning(f"Could not query /model/info: {exc} — assuming uses_barometer=False")
                self._uses_baro_cache = False
        return self._uses_baro_cache

    # ------------------------------------------------------------------
    def predict(
        self,
        patient_id:             str,
        device_id:              Optional[str],
        acc_x:                  List[float],
        acc_y:                  List[float],
        acc_z:                  List[float],
        timestamps_ms:          List[float],
        pressure:               Optional[List[float]] = None,
        pressure_timestamps_ms: Optional[List[float]] = None,
    ) -> Optional[dict]:
        """
        POST /predict and return the JSON response.
        On HTTP error returns None and logs the failure (caller decides what to do).
        """
        body = {
            "patient_id":    patient_id,
            "device_id":     device_id,
            "acc_x":         list(map(float, acc_x)),
            "acc_y":         list(map(float, acc_y)),
            "acc_z":         list(map(float, acc_z)),
            "timestamps_ms": list(map(float, timestamps_ms)),
        }
        if pressure is not None and len(pressure) > 0:
            body["pressure"] = list(map(float, pressure))
            body["pressure_timestamps_ms"] = list(map(float, pressure_timestamps_ms or []))

        try:
            resp = requests.post(
                f"{self.server_url}/predict",
                json=body,
                headers=self._headers(),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            logger.error(f"Inference server unreachable: {exc}")
            return None

        if not resp.ok:
            logger.error(f"Inference server returned {resp.status_code}: {resp.text[:200]}")
            return None
        return resp.json()
