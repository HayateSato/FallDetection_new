"""
MockAppPoller — simulates the mobile app's sensor-to-inference loop.

What the real mobile app will do:
  1. Read a 9-second ACC window directly from the SmarKo wearable (BLE)
  2. POST the raw LSB arrays to the inference server's /predict endpoint
  3. The inference server returns the result (and publishes to MQTT if a fall)
  4. The app shows a local notification if a fall is detected

What this mock does instead of step 1:
  - Fetches the same data from InfluxDB (where the wearable writes it in real time)

Everything from step 2 onward is identical to the real app behaviour.
No database writes happen here — the caregiver_client handles that via MQTT.
"""

import logging
import threading
from typing import List, Optional

from mock_app.influx_fetcher import fetch_raw_window
from mock_app.api_caller import InferenceServerClient

logger = logging.getLogger(__name__)


class MockAppPoller(threading.Thread):
    """
    Background thread: loops over configured patients, fetches the latest
    sensor window from InfluxDB, and POSTs it to the inference server.

    The inference server publishes to MQTT on fall detection — this class
    does not handle that; it's the caregiver_client's responsibility.
    """

    def __init__(
        self,
        inference_client: InferenceServerClient,
        patient_ids:      List[str],
        mac_map:          Optional[dict] = None,
        poll_interval:    int = 10,
        lookback_seconds: int = 15,
    ):
        super().__init__(daemon=True, name="MockAppPoller")
        self.client          = inference_client
        self.patient_ids     = patient_ids
        self.mac_map         = mac_map or {}
        self.poll_interval   = poll_interval
        self.lookback_seconds = lookback_seconds
        self._stop           = threading.Event()
        self._uses_barometer: Optional[bool] = None

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def _ensure_baro_known(self) -> bool:
        if self._uses_barometer is None:
            self._uses_barometer = self.client.uses_barometer()
            logger.info(f"Server uses_barometer = {self._uses_barometer}")
        return self._uses_barometer

    # ------------------------------------------------------------------
    def _poll_one(self, patient_id: str) -> None:
        uses_baro   = self._ensure_baro_known()
        mac_address = self.mac_map.get(patient_id)

        acc_data, acc_time, pressure, pressure_time = fetch_raw_window(
            patient_id       = patient_id,
            lookback_seconds = self.lookback_seconds,
            uses_barometer   = uses_baro,
            mac_address      = mac_address,
        )
        if acc_data is None:
            logger.info(f"No InfluxDB data for {patient_id} (lookback={self.lookback_seconds}s)")
            return

        result = self.client.predict(
            patient_id             = patient_id,
            device_id              = mac_address,
            acc_x                  = acc_data[0].tolist(),
            acc_y                  = acc_data[1].tolist(),
            acc_z                  = acc_data[2].tolist(),
            timestamps_ms          = acc_time.tolist(),
            pressure               = pressure.tolist()      if pressure      is not None else None,
            pressure_timestamps_ms = pressure_time.tolist() if pressure_time is not None else None,
        )
        if result is None:
            return

        inference  = result.get("inference", {})
        is_fall    = bool(inference.get("fall_detected", False))
        confidence = inference.get("confidence", 0.0)

        if is_fall:
            logger.info(
                f"*** FALL — patient={patient_id}  confidence={confidence:.3f}  "
                f"(inference server will publish to MQTT)"
            )
        else:
            logger.info(f"no fall — patient={patient_id}  confidence={confidence:.3f}")

    # ------------------------------------------------------------------
    def run(self) -> None:
        logger.info(
            f"MockAppPoller started  patients={self.patient_ids}  "
            f"interval={self.poll_interval}s  lookback={self.lookback_seconds}s  "
            f"mac_map={self.mac_map or '(none)'}"
        )
        while not self._stop.is_set():
            for pid in self.patient_ids:
                if self._stop.is_set():
                    break
                try:
                    self._poll_one(pid)
                except Exception as exc:
                    logger.error(f"Poll error for {pid}: {exc}", exc_info=True)
            self._stop.wait(self.poll_interval)
        logger.info("MockAppPoller stopped")
