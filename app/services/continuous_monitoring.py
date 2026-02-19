"""
Continuous Monitoring Module for Fall Detection System.

Periodically fetches sensor data from InfluxDB and runs fall detection
using the PipelineSelector engine.
"""
import threading
import time
import queue
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable
import logging

from app.data_input.data_loader.data_fetcher import fetch_data
from app.data_input.influx_to_numpy_converter import convert_acc_influx_to_numpy, convert_barometer_influx_to_numpy
from app.core.inference_engine import PipelineSelector
from app.core.recording_state import recording_state

from config.settings import (
    INFLUXDB_BUCKET,
    BAROMETER_FIELD,
    MONITORING_INTERVAL_SECONDS,
    MONITORING_LOOKBACK_SECONDS,
    ACC_SAMPLE_RATE,
    WINDOW_SIZE_SECONDS,
    TIMEZONE_OFFSET_HOURS,
    COLLECT_ADDITIONAL_SENSORS,
    MODEL_VERSION,
    # Sensor and sampling rate settings
    ACC_FIELD_X,
    ACC_FIELD_Y,
    ACC_FIELD_Z,
    HARDWARE_ACC_SAMPLE_RATE,
    MODEL_ACC_SAMPLE_RATE,
    RESAMPLING_ENABLED,
    RESAMPLING_METHOD,
    BAROMETER_ENABLED,
    SENSOR_CALIBRATION_ENABLED,
)
from app.data_input.resampler import AccelerometerResampler
from app.data_input.accelerometer_processor.nonbosch_calibration import transform_acc_array as calibrate_non_bosch_to_bosch

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
        self.window_size_seconds = WINDOW_SIZE_SECONDS
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
                # Get current recording state (for participant info and export)
                state = recording_state.get_current_state()

                # Run detection (always fetch and analyze data)
                self._run_detection(state)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)

            # Wait for next interval
            self._stop_event.wait(timeout=self.interval_seconds)

        logger.info("Monitoring loop ended")

    def _run_detection(self, state: dict) -> None:
        """Run a single fall detection cycle."""
        timestamp = datetime.now(timezone.utc)

        try:
            # Fetch data from InfluxDB
            acc_data, acc_time, pressure, pressure_time, flux_records = self._fetch_sensor_data()

            if acc_data is None or acc_data.shape[1] < self.required_acc_samples:
                logger.warning(f"Insufficient data: {acc_data.shape[1] if acc_data is not None else 0} ACC samples (need {self.required_acc_samples})")
                return

            # Convert to DataFrame
            window_df = self._convert_to_dataframe(acc_data, acc_time)

            if len(window_df) < self.required_acc_samples:
                logger.warning(f"Insufficient window samples: {len(window_df)}")
                return

            # Extract window
            window_df = window_df.tail(self.required_acc_samples).reset_index(drop=True)

            # Align barometer to window
            window_pressure = None
            window_pressure_time = None
            baro_samples = 0

            if self.uses_barometer and pressure is not None and len(pressure) > 0:
                window_start_ms = window_df['Device_Timestamp_[ms]'].iloc[0]
                window_end_ms = window_df['Device_Timestamp_[ms]'].iloc[-1]
                mask = (pressure_time >= window_start_ms) & (pressure_time <= window_end_ms)
                window_pressure = pressure[mask]
                window_pressure_time = pressure_time[mask]
                baro_samples = len(window_pressure)

            # Log fetched data details
            acc_duration = (acc_time[-1] - acc_time[0]) / 1000.0 if len(acc_time) > 1 else 0
            acc_x_range = (acc_data[0].min(), acc_data[0].max())
            acc_y_range = (acc_data[1].min(), acc_data[1].max())
            acc_z_range = (acc_data[2].min(), acc_data[2].max())

            logger.info(f"--- Data Fetch ---")
            logger.info(f"  ACC: {len(window_df)} samples, {acc_duration:.1f}s duration")
            logger.info(f"  ACC X: [{acc_x_range[0]:.0f}, {acc_x_range[1]:.0f}]  Y: [{acc_y_range[0]:.0f}, {acc_y_range[1]:.0f}]  Z: [{acc_z_range[0]:.0f}, {acc_z_range[1]:.0f}]")

            if self.uses_barometer and window_pressure is not None and len(window_pressure) > 0:
                pressure_range = (window_pressure.min(), window_pressure.max())
                logger.info(f"  BARO: {baro_samples} samples, range [{pressure_range[0]:.0f}, {pressure_range[1]:.0f}] Pa")
            else:
                logger.info(f"  BARO: {baro_samples} samples")

            # Run inference
            result = self.inference_engine.predict(
                window_df,
                pressure=window_pressure,
                pressure_timestamps=window_pressure_time
            )

            is_fall = result['is_fall']
            confidence = result['confidence']

            # Log result
            if is_fall:
                logger.info(f"\n\n\n\n                     >>> [FALL DETECTED] Confidence: {confidence:.2%}\n\n")
            else:
                logger.info(f"  Result: No fall (confidence: {confidence:.2%})")

            # Send notification
            notification_data = {
                'is_fall': is_fall,
                'confidence': confidence,
                'timestamp': timestamp.isoformat(),
                'model_version': MODEL_VERSION
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
                    timestamp_utc=timestamp
                )

        except Exception as e:
            logger.error(f"Error in detection cycle: {e}", exc_info=True)

    def _fetch_sensor_data(self):
        """Fetch sensor data from InfluxDB."""
        # Build query using configurable field names from settings
        fields_filter = f'r["_field"] == "{ACC_FIELD_X}" or r["_field"] == "{ACC_FIELD_Y}" or r["_field"] == "{ACC_FIELD_Z}"'

        # Add barometer field if model uses it and barometer is enabled
        if self.uses_barometer and BAROMETER_ENABLED:
            fields_filter += f' or r["_field"] == "{BAROMETER_FIELD}"'

        # Always include manual_truth_marker and user_feedback fields so they appear in exported CSVs
        fields_filter += ' or r["_field"] == "manual_truth_marker" or r["_field"] == "user_feedback"'

        query = f'''from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: -{self.lookback_seconds}s)
          |> filter(fn: (r) => {fields_filter})
        '''

        tables = fetch_data(query)

        if tables is None:
            return None, None, None, None, None

        # Collect all records
        flux_records = []
        for table in tables:
            for record in table.records:
                flux_records.append(record)

        if len(flux_records) == 0:
            return None, None, None, None, None

        # Preprocess using configurable field names
        acc_data, acc_time = convert_acc_influx_to_numpy(
            flux_records,
            acc_field_x=ACC_FIELD_X,
            acc_field_y=ACC_FIELD_Y,
            acc_field_z=ACC_FIELD_Z
        )

        # Apply resampling if needed (25Hz->50Hz or 100Hz->50Hz)
        if RESAMPLING_ENABLED and acc_data is not None and acc_data.shape[1] > 0:
            resampler = AccelerometerResampler(
                source_rate=HARDWARE_ACC_SAMPLE_RATE,
                target_rate=MODEL_ACC_SAMPLE_RATE,
                method=RESAMPLING_METHOD
            )
            acc_data, acc_time = resampler.process(acc_data, acc_time)

        # Apply sensor calibration if using non_bosch sensor
        # This transforms non_bosch values to bosch-equivalent values
        if SENSOR_CALIBRATION_ENABLED and acc_data is not None and acc_data.shape[1] > 0:
            acc_data = calibrate_non_bosch_to_bosch(acc_data)

        pressure = np.array([])
        pressure_time = np.array([])

        if self.uses_barometer and BAROMETER_ENABLED:
            pressure, pressure_time = convert_barometer_influx_to_numpy(flux_records, BAROMETER_FIELD)

        return acc_data, acc_time, pressure, pressure_time, flux_records

    def _convert_to_dataframe(self, acc_data: np.ndarray, acc_time: np.ndarray) -> pd.DataFrame:
        """Convert accelerometer arrays to DataFrame."""
        acc_scale_factor = 1.0 / 16384.0

        return pd.DataFrame({
            'Device_Timestamp_[ms]': acc_time,
            'Acc_X[g]': acc_data[0] * acc_scale_factor,
            'Acc_Y[g]': acc_data[1] * acc_scale_factor,
            'Acc_Z[g]': acc_data[2] * acc_scale_factor
        })
