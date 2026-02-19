"""
Continuous Monitoring Module for Fall Detection System.

Periodically fetches sensor data from InfluxDB and runs fall detection
using the PipelineSelector engine.
"""
import threading
import queue
from datetime import datetime, timezone
from typing import Optional, Callable
import logging

from app.core.inference_engine import PipelineSelector
from app.core.recording_state import recording_state
from app.data_input.sensor_data_reader import fetch_and_preprocess_sensor_data
from app.data_output.data_exporter import convert_to_dataframe, extract_window

from config.settings import (
    MONITORING_INTERVAL_SECONDS,
    MONITORING_LOOKBACK_SECONDS,
    ACC_SAMPLE_RATE,
    WINDOW_SIZE_SECONDS,
    MODEL_VERSION,
)

logger = logging.getLogger(__name__)


class ContinuousMonitor:
    """
    Continuous monitoring class that periodically checks for falls.

    Runs in a background thread and uses PipelineSelector for fall detection.
    """

    def __init__(
        self,
        inference_engine: PipelineSelector,
        notification_queue: Optional[queue.Queue] = None,
        export_callback: Optional[Callable] = None,
        notification_callback: Optional[Callable] = None
    ):
        """
        Initialize the continuous monitor.

        Args:
            inference_engine: PipelineSelector instance for fall detection
            notification_queue: Queue for sending fall notifications (SSE)
            export_callback: Optional callback for exporting detection data
            notification_callback: Optional callback called with fall data dict (for polling)
        """
        self.inference_engine = inference_engine
        self.notification_queue = notification_queue
        self.export_callback = export_callback
        self.notification_callback = notification_callback

        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Monitoring parameters
        self.interval_seconds = MONITORING_INTERVAL_SECONDS
        self.lookback_seconds = MONITORING_LOOKBACK_SECONDS
        self.required_acc_samples = int(WINDOW_SIZE_SECONDS * ACC_SAMPLE_RATE)

        # Model info
        self.model_info = inference_engine.get_model_info()
        self.uses_barometer = inference_engine.uses_barometer()

    def start(self) -> bool:
        """Start continuous monitoring in background thread."""
        if self.is_running:
            logger.warning("Continuous monitoring is already running")
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        self.is_running = True

        logger.info("Continuous monitoring started")
        return True

    def stop(self) -> None:
        """Stop continuous monitoring."""
        if not self.is_running:
            return

        logger.info("Stopping continuous monitoring...")
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        self.is_running = False
        logger.info("Continuous monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        logger.info("Monitoring loop started")

        while not self._stop_event.is_set():
            try:
                state = recording_state.get_current_state()
                self._run_detection(state)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)

            self._stop_event.wait(timeout=self.interval_seconds)

        logger.info("Monitoring loop ended")

    def _run_detection(self, state: dict) -> None:
        """Run a single fall detection cycle."""
        timestamp = datetime.now(timezone.utc)

        try:
            # Fetch and preprocess sensor data
            acc_data, acc_time, pressure, pressure_time, flux_records = fetch_and_preprocess_sensor_data(
                uses_barometer=self.uses_barometer,
                lookback_seconds=self.lookback_seconds,
            )

            if acc_data is None or acc_data.shape[1] < self.required_acc_samples:
                logger.warning(
                    f"Insufficient data: {acc_data.shape[1] if acc_data is not None else 0} "
                    f"ACC samples (need {self.required_acc_samples})"
                )
                return

            # Log data summary
            acc_duration = (acc_time[-1] - acc_time[0]) / 1000.0 if len(acc_time) > 1 else 0
            logger.info("--- Data Fetch ---")
            logger.info(f"  ACC: {acc_data.shape[1]} samples, {acc_duration:.1f}s duration")
            logger.info(
                f"  ACC X: [{acc_data[0].min():.0f}, {acc_data[0].max():.0f}]"
                f"  Y: [{acc_data[1].min():.0f}, {acc_data[1].max():.0f}]"
                f"  Z: [{acc_data[2].min():.0f}, {acc_data[2].max():.0f}]"
            )

            # Convert to DataFrame and extract detection window
            full_df = convert_to_dataframe(acc_data, acc_time)
            window_df, window_pressure, window_pressure_time = extract_window(
                full_df, self.required_acc_samples, pressure, pressure_time
            )

            baro_samples = len(window_pressure) if window_pressure is not None else 0
            if self.uses_barometer and window_pressure is not None and len(window_pressure) > 0:
                logger.info(f"  BARO: {baro_samples} samples, range [{window_pressure.min():.0f}, {window_pressure.max():.0f}] Pa")
            else:
                logger.info(f"  BARO: {baro_samples} samples")

            # Run inference
            result = self.inference_engine.predict(
                window_df,
                pressure=window_pressure,
                pressure_timestamps=window_pressure_time,
            )

            is_fall = result['is_fall']
            confidence = result['confidence']

            if is_fall:
                logger.info(f"\n\n\n\n                     >>> [FALL DETECTED] Confidence: {confidence:.2%}\n\n")
            else:
                logger.info(f"  Result: No fall (confidence: {confidence:.2%})")

            # Send notification
            notification_data = {
                'is_fall': is_fall,
                'confidence': confidence,
                'timestamp': timestamp.isoformat(),
                'model_version': MODEL_VERSION,
            }
            if self.notification_queue is not None:
                self.notification_queue.put(notification_data)
            if self.notification_callback is not None:
                self.notification_callback(notification_data)

            # Export data only when recording is active
            if self.export_callback and flux_records and state.get('recording_active', False):
                self.export_callback(
                    flux_records=flux_records,
                    is_fall=is_fall,
                    confidence=confidence,
                    participant_name=state.get('participant_name', 'unknown'),
                    participant_gender=state.get('participant_gender', 'unknown'),
                    manual_truth_fall=state.get('manual_truth_fall', 0),
                    timestamp_utc=timestamp,
                )

        except Exception as e:
            logger.error(f"Error in detection cycle: {e}", exc_info=True)
