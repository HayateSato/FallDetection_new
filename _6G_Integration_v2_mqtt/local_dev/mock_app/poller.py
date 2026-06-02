"""
MockAppPoller — simulates the mobile app's sensor-to-inference loop.

What the real mobile app will do:
  1. Read a 9-second ACC window directly from the SmarKo wearable (BLE)
  2. POST the raw LSB arrays to the inference server's /predict endpoint
  3. If fall_detected=True → show "Did you fall? / Do you need help?" popup (10s timeout)
  4. If patient confirms OR no answer within timeout → publish fall/alert/<patient_id> to MQTT
  5. Caregiver dashboard subscribes to fall/alert/# and shows the alert

What this mock does:
  - Step 1: fetches from InfluxDB instead of BLE wearable
  - Step 3: routes through PatientConfirmationServer (browser popup at http://localhost:8005/)
             if patient_server is provided; otherwise times out immediately (conservative)
  - Step 4: publishes MQTT alert with the actual patient response (or not_answered on timeout)

MQTT topics:
  fall/alert/<patient_id>   — published by THIS mock after patient confirmation / timeout
                              (fall_dashboard subscribes to fall/alert/#)
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional

from local_dev.mock_app.influx_fetcher import fetch_raw_window
from local_dev.mock_app.api_caller import InferenceServerClient

logger = logging.getLogger(__name__)


class MockAppPoller(threading.Thread):
    """
    Background thread: loops over patients, fetches sensor data, calls inference.

    On fall detection: opens a patient confirmation window (MOCK_PATIENT_RESPONSE_TIMEOUT
    seconds). After the window closes, publishes a fall/alert to the MQTT broker so
    the caregiver dashboard can show the alert.

    The caregiver is only notified AFTER the patient confirmation step — not immediately
    on fall detection.
    """

    def __init__(
        self,
        inference_client:      InferenceServerClient,
        patient_ids:           List[str],
        mqtt_publisher,                           # paho mqtt.Client (connected)
        mac_map:               Optional[dict] = None,
        poll_interval:         int = 10,
        lookback_seconds:      int = 15,
        alert_topic:           str = "fall/alert",
        confirmation_timeout:  int = 10,
        patient_server=None,                      # PatientConfirmationServer | None
    ):
        super().__init__(daemon=True, name="MockAppPoller")
        self.client               = inference_client
        self.patient_ids          = patient_ids
        self._mqtt                = mqtt_publisher
        self.mac_map              = mac_map or {}
        self.poll_interval        = poll_interval
        self.lookback_seconds     = lookback_seconds
        self._alert_topic         = alert_topic
        self._confirmation_timeout = confirmation_timeout
        self._patient_server      = patient_server
        self._stop                = threading.Event()
        self._uses_barometer:     Optional[bool] = None

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
    def _simulate_patient_confirmation(self, patient_id: str, event: dict) -> None:
        """
        Runs in a daemon thread, simulating the patient seeing the confirmation popup.

        Real mobile app behaviour:
          - Screen wakes up, shows: "Did you fall? (Yes/No)" with 10s countdown
          - If Yes: second popup "Do you need help? (Yes/No)"
          - If Yes or timeout: publish fall/alert → caregiver is notified
          - If No at first popup: no alert sent to caregiver

        Mock behaviour (with patient_server):
          - Browser popup appears at http://localhost:<MOCK_PATIENT_SERVER_PORT>/
          - Waits up to confirmation_timeout seconds for the patient to respond in browser
          - Uses actual response (yes/no/not_answered) in the MQTT payload

        Mock behaviour (without patient_server):
          - Waits confirmation_timeout seconds, then publishes not_answered (conservative)
        """
        if self._patient_server is not None:
            self._patient_server.notify_fall(event)
            confirmation = self._patient_server.wait_for_response(timeout=self._confirmation_timeout)
            patient_confirmed = confirmation["patient_confirmed"]
            needs_help        = confirmation["needs_help"]
            logger.info(
                f"[Patient confirmation received]  patient={patient_id}  "
                f"confirmed={patient_confirmed}  needs_help={needs_help}"
            )
        else:
            logger.info(
                f"[Patient confirmation window open — {self._confirmation_timeout}s]  "
                f"patient={patient_id}  (no patient_server — will time out)"
            )
            time.sleep(self._confirmation_timeout)
            patient_confirmed = "not_answered"
            needs_help        = True  # conservative: assume help needed on timeout

        alert_payload = {
            **event,
            "alert_time":        datetime.now(timezone.utc).isoformat(),
            "patient_confirmed": patient_confirmed,
            "needs_help":        needs_help,
        }

        topic = f"{self._alert_topic}/{patient_id}"
        try:
            self._mqtt.publish(topic, json.dumps(alert_payload, default=str))
            logger.info(
                f"*** ALERT published  topic={topic}  patient={patient_id}  "
                f"confirmed={patient_confirmed}"
            )
        except Exception as exc:
            logger.warning(f"Alert publish failed for {patient_id}: {exc}")

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

        if not is_fall:
            logger.info(f"no fall — patient={patient_id}  confidence={confidence:.3f}")
            return

        logger.info(
            f"FALL DETECTED — patient={patient_id}  confidence={confidence:.3f}  "
            f"opening patient confirmation window ({self._confirmation_timeout}s)..."
        )

        # Build the event payload that will eventually reach the caregiver.
        # observation_id is the UUID returned by /predict — after the confirmation
        # popup the real mobile app should POST to /inference/{observation_id}/confirm
        # so the retraining pipeline has the ground-truth label.
        event = {
            "patient_id":       patient_id,
            "device_id":        mac_address,
            "timestamp":        result.get("timestamp"),
            "observation_id":   result.get("observation_id"),   # UUID cross-ref key
            "fall_detected":    True,
            "confidence":       round(float(confidence), 4),
            "model_version":    inference.get("model_version"),
            "fhir_observation": result.get("fhir_observation"),
        }

        # Open patient confirmation in background — poller continues polling other patients
        t = threading.Thread(
            target=self._simulate_patient_confirmation,
            args=(patient_id, event),
            daemon=True,
            name=f"PatientConfirm-{patient_id}",
        )
        t.start()

    # ------------------------------------------------------------------
    def run(self) -> None:
        logger.info(
            f"MockAppPoller started  patients={self.patient_ids}  "
            f"interval={self.poll_interval}s  lookback={self.lookback_seconds}s  "
            f"confirmation_timeout={self._confirmation_timeout}s  "
            f"alert_topic={self._alert_topic}/#"
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
