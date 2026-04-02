"""
Remote Inference Client

Sends raw sensor data to the remote FastAPI server for ML inference
instead of running the model locally. Used when INFERENCE_MODE=remote.
"""

import logging
import requests
import numpy as np
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class RemoteInferenceClient:
    """
    HTTP client that sends sensor data to the remote inference server.

    Replaces local PipelineSelector when INFERENCE_MODE=remote.
    The remote server handles all preprocessing (resampling, LSB->g,
    windowing, feature extraction) and XGBoost inference.
    """

    def __init__(self, server_url: str, api_key: str = "", timeout: int = 30):
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._model_info_cache = None

        logger.info(f"Remote inference client initialized: {self.server_url}")

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def health_check(self) -> Dict[str, Any]:
        """Check if the remote server is reachable and healthy."""
        resp = requests.get(
            f"{self.server_url}/health",
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata from the remote server (cached after first call)."""
        if self._model_info_cache is not None:
            return self._model_info_cache

        resp = requests.get(
            f"{self.server_url}/model/info",
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        self._model_info_cache = resp.json()
        return self._model_info_cache

    def predict(
        self,
        acc_data: np.ndarray,
        acc_time: np.ndarray,
        pressure: Optional[np.ndarray] = None,
        pressure_time: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        acc_unit: str = "lsb",
    ) -> Dict[str, Any]:
        """
        Send sensor data to the remote server for fall detection.

        Args:
            acc_data: shape (3, N) raw accelerometer data [x, y, z]
            acc_time: shape (N,) timestamps in milliseconds
            pressure: shape (M,) barometer pressure values (optional)
            pressure_time: shape (M,) barometer timestamps in ms (optional)
            sample_rate: hardware sampling rate in Hz (optional, server uses its .env default)
            acc_unit: 'lsb' or 'g'

        Returns:
            Server response dict with fall_detected, confidence, etc.
        """
        payload = {
            "acc_x": acc_data[0].tolist(),
            "acc_y": acc_data[1].tolist(),
            "acc_z": acc_data[2].tolist(),
            "timestamps_ms": acc_time.tolist(),
            "acc_unit": acc_unit,
        }

        if sample_rate is not None:
            payload["sample_rate"] = sample_rate

        if pressure is not None and len(pressure) > 0:
            payload["pressure"] = pressure.tolist()
        if pressure_time is not None and len(pressure_time) > 0:
            payload["pressure_timestamps_ms"] = pressure_time.tolist()

        logger.info(
            f"Sending {acc_data.shape[1]} ACC samples to {self.server_url}/predict"
        )

        resp = requests.post(
            f"{self.server_url}/predict",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()
