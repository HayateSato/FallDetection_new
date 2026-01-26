"""
Fall Detection System - Dynamic Model Selection

Supports multiple model versions with automatic preprocessing:
- v1: ACC (statistical) + BARO (dual-path EMA)
- v2: ACC (paper magnitude) + BARO (paper slope-limit)
- v3: ACC (statistical) + BARO (paper slope-limit) [RECOMMENDED]
- v4: ACC (paper magnitude) + BARO (dual-path EMA)
- v5: Raw features (minimal preprocessing)

Data Sources:
- influx: Real-time data from InfluxDB with continuous monitoring
- csv: Batch validation from local CSV files

Set MODEL_VERSION and DATA_SOURCE in .env file.
"""
from flask import Flask, jsonify, Response, send_from_directory, request
import sys
import os
import json
import queue
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import logging

# Import app modules
from app.data_fetcher import fetch_data
from app.data_processor import preprocess_acc, preprocess_barometer
from app.unified_inference import UnifiedInference
from app.model_registry import get_model_type, get_model_config
from app.utils.model_logger import model_logger
from app.continuous_monitoring import ContinuousMonitor

# Import settings
from config.settings import (
    MODEL_VERSION,
    MODEL_PATH,
    BAROMETER_FIELD,
    MONITORING_ENABLED,
    MONITORING_INTERVAL_SECONDS,
    FALL_DATA_EXPORT_ENABLED,
    TIMEZONE_OFFSET_HOURS,
    WINDOW_SIZE_SECONDS,
    ACC_SAMPLE_RATE,
    BARO_SAMPLE_RATE,
    TODAY,
    INFLUXDB_BUCKET,
    COLLECT_ADDITIONAL_SENSORS,
    print_config,
    # Sensor and sampling rate settings
    ACC_SENSOR_TYPE,
    HARDWARE_ACC_SAMPLE_RATE,
    MODEL_ACC_SAMPLE_RATE,
    BAROMETER_ENABLED,
    UPSAMPLING_ENABLED,
    DOWNSAMPLING_ENABLED,
    RESAMPLING_ENABLED,
    RESAMPLING_METHOD,
    ACC_FIELD_X,
    ACC_FIELD_Y,
    ACC_FIELD_Z,
    WINDOW_SAMPLES,
    HARDWARE_WINDOW_SAMPLES,
    SENSOR_CALIBRATION_ENABLED,
)

# Import resampler for sample rate conversion (25Hz->50Hz or 100Hz->50Hz)
from app.resampler import AccelerometerResampler

# Import sensor calibration for non_bosch to bosch-equivalent transformation
from app.sensor_calibration import transform_acc_array as calibrate_non_bosch_to_bosch

# Legacy alias
SAMPLING_RATE = ACC_SAMPLE_RATE

# Data source settings
DATA_SOURCE = os.getenv('DATA_SOURCE', 'influx').lower()
CSV_FILE_PATH = os.getenv('CSV_FILE_PATH', '')
CSV_WINDOW_INTERVAL_SECONDS = float(os.getenv('CSV_WINDOW_INTERVAL_SECONDS', '1.0'))


# ------------------------------------------------------------------------
# CSV DATA LOADER
# ------------------------------------------------------------------------

class CSVDataLoader:
    """
    Load and process CSV data for fall detection validation.

    Expected CSV format:
    - timestamp: epoch milliseconds
    - is_accelerometer_bosch: 1 if row has ACC data
    - bosch_acc_x, bosch_acc_y, bosch_acc_z: accelerometer values (raw)
    - is_pressure: 1 if row has barometer data
    - pressure_in_pa: pressure value in Pascals
    """

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df = None
        self.acc_df = None
        self.baro_df = None
        self.min_timestamp = None
        self.max_timestamp = None

    def load(self) -> bool:
        """Load CSV file and separate ACC and BARO data."""
        try:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

            self.df = pd.read_csv(self.csv_path)

            # Extract accelerometer data where is_accelerometer_bosch == 1
            self.acc_df = self.df[self.df['is_accelerometer_bosch'] == 1].copy()
            self.acc_df = self.acc_df[['timestamp', 'bosch_acc_x', 'bosch_acc_y', 'bosch_acc_z']].reset_index(drop=True)

            # Extract barometer data where is_pressure == 1
            self.baro_df = self.df[self.df['is_pressure'] == 1].copy()
            self.baro_df = self.baro_df[['timestamp', 'pressure_in_pa']].reset_index(drop=True)

            # Get timestamp bounds
            self.min_timestamp = self.df['timestamp'].min()
            self.max_timestamp = self.df['timestamp'].max()

            duration_seconds = (self.max_timestamp - self.min_timestamp) / 1000.0

            print(f"CSV loaded: {self.csv_path.name}")
            print(f"  Total rows: {len(self.df)}")
            print(f"  ACC samples: {len(self.acc_df)}")
            print(f"  BARO samples: {len(self.baro_df)}")
            print(f"  Duration: {duration_seconds:.1f} seconds")
            print(f"  Timestamp range: {self.min_timestamp} - {self.max_timestamp}")

            return True

        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False

    def get_window(self, start_ms: int, window_size_seconds: float) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract a time window of data."""
        end_ms = start_ms + int(window_size_seconds * 1000)

        # Extract ACC window
        acc_window = self.acc_df[
            (self.acc_df['timestamp'] >= start_ms) &
            (self.acc_df['timestamp'] < end_ms)
        ].copy()

        if len(acc_window) == 0:
            return None, None, None

        # Convert to expected format (with g conversion)
        acc_scale_factor = 1.0 / 16384.0

        window_df = pd.DataFrame({
            'Device_Timestamp_[ms]': acc_window['timestamp'].values,
            'Acc_X[g]': acc_window['bosch_acc_x'].values * acc_scale_factor,
            'Acc_Y[g]': acc_window['bosch_acc_y'].values * acc_scale_factor,
            'Acc_Z[g]': acc_window['bosch_acc_z'].values * acc_scale_factor,
        })

        # Extract BARO window
        baro_window = self.baro_df[
            (self.baro_df['timestamp'] >= start_ms) &
            (self.baro_df['timestamp'] < end_ms)
        ].copy()

        if len(baro_window) > 0:
            pressure = baro_window['pressure_in_pa'].values
            pressure_timestamps = baro_window['timestamp'].values
        else:
            pressure = np.array([])
            pressure_timestamps = np.array([])

        return window_df, pressure, pressure_timestamps

    def get_window_timestamps(self, interval_seconds: float, window_size_seconds: float) -> List[int]:
        """Get list of window start timestamps for sliding window processing."""
        if self.min_timestamp is None or self.max_timestamp is None:
            return []

        interval_ms = int(interval_seconds * 1000)
        window_ms = int(window_size_seconds * 1000)

        last_start = self.max_timestamp - window_ms

        timestamps = []
        current = self.min_timestamp

        while current <= last_start:
            timestamps.append(current)
            current += interval_ms

        return timestamps


def save_csv_validation_results(csv_path: str, results: List[Dict], model_version: str) -> str:
    """Save fall detection results to a text file."""
    csv_name = Path(csv_path).stem
    output_dir = Path("results/csvdata_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{csv_name}_{model_version}_fall_detections.txt"

    fall_detections = [r for r in results if r.get('is_fall', False)]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Fall Detection Results - {csv_name}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Source CSV: {csv_path}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Version: {model_version}\n")
        f.write(f"Window Size: {WINDOW_SIZE_SECONDS} seconds\n")
        f.write(f"Window Interval: {CSV_WINDOW_INTERVAL_SECONDS} seconds\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total windows analyzed: {len(results)}\n")
        f.write(f"Falls detected: {len(fall_detections)}\n\n")

        if len(fall_detections) > 0:
            f.write("FALL DETECTIONS:\n")
            f.write("-" * 70 + "\n")

            for i, detection in enumerate(fall_detections, 1):
                start_ms = detection['window_start_ms']
                end_ms = detection['window_end_ms']
                confidence = detection['confidence']

                start_dt = datetime.fromtimestamp(start_ms / 1000)
                end_dt = datetime.fromtimestamp(end_ms / 1000)

                f.write(f"\n[Fall #{i}]\n")
                f.write(f"  Window: {start_ms} - {end_ms} (ms)\n")
                f.write(f"  Time: {start_dt.strftime('%H:%M:%S.%f')[:-3]} - {end_dt.strftime('%H:%M:%S.%f')[:-3]}\n")
                f.write(f"  Confidence: {confidence:.4f} ({confidence*100:.1f}%)\n")
                f.write(f"  ACC samples: {detection.get('acc_samples', 'N/A')}\n")
                f.write(f"  BARO samples: {detection.get('baro_samples', 'N/A')}\n")
        else:
            f.write("No falls detected in this recording.\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("End of Report\n")

    print(f"Results saved to: {output_file}")
    return str(output_file)


def process_csv_file(csv_path: str, interval_seconds: float, inference_engine: UnifiedInference, model_version: str) -> Dict:
    """Process a CSV file with sliding windows and detect falls."""
    model_info = inference_engine.get_model_info()

    print("=" * 70)
    print(f"CSV VALIDATION MODE - {model_info['name']}")
    print("=" * 70)
    print(f"CSV file: {csv_path}")
    print(f"Model: {model_info['description']}")
    print(f"Features: {model_info['num_features']} ({model_info['acc_features']} ACC + {model_info['baro_features']} BARO)")
    print(f"Window interval: {interval_seconds} seconds")
    print("=" * 70)

    # Load CSV
    loader = CSVDataLoader(csv_path)
    if not loader.load():
        return {
            'success': False,
            'error': 'Failed to load CSV file',
            'results': []
        }

    # Get window timestamps
    window_starts = loader.get_window_timestamps(interval_seconds, WINDOW_SIZE_SECONDS)

    if len(window_starts) == 0:
        return {
            'success': False,
            'error': 'CSV duration too short for analysis window',
            'results': []
        }

    print(f"\nProcessing {len(window_starts)} windows...")
    print("-" * 70)

    results = []
    fall_count = 0

    for i, start_ms in enumerate(window_starts):
        window_df, pressure, pressure_timestamps = loader.get_window(start_ms, WINDOW_SIZE_SECONDS)

        if window_df is None or len(window_df) < 100:
            continue

        try:
            # Use inference engine - it handles preprocessing based on model version
            if inference_engine.uses_barometer():
                result = inference_engine.predict(
                    window_df,
                    pressure=pressure,
                    pressure_timestamps=pressure_timestamps
                )
            else:
                result = inference_engine.predict(window_df)

            is_fall = result['is_fall']
            confidence = result['confidence']

            end_ms = start_ms + int(WINDOW_SIZE_SECONDS * 1000)

            results.append({
                'window_index': i,
                'window_start_ms': start_ms,
                'window_end_ms': end_ms,
                'is_fall': is_fall,
                'confidence': confidence,
                'acc_samples': len(window_df),
                'baro_samples': len(pressure) if pressure is not None else 0
            })

            if is_fall:
                fall_count += 1
                start_dt = datetime.fromtimestamp(start_ms / 1000)
                print(f"  [FALL DETECTED] Window {i}: {start_dt.strftime('%H:%M:%S.%f')[:-3]} - confidence: {confidence:.4f}")

        except Exception as e:
            print(f"  Error processing window {i}: {e}")
            continue

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(window_starts)} windows...")

    print("-" * 70)
    print(f"Processing complete!")
    print(f"  Total windows: {len(results)}")
    print(f"  Falls detected: {fall_count}")

    output_file = save_csv_validation_results(csv_path, results, model_version)

    return {
        'success': True,
        'csv_path': csv_path,
        'model_version': model_version,
        'total_windows': len(results),
        'falls_detected': fall_count,
        'output_file': output_file,
        'results': results
    }


# ------------------------------------------------------------------------
# SETUP LOGGING
# ------------------------------------------------------------------------

log_dir = os.path.join("results", "logs", TODAY)
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f"fall_detection_{MODEL_VERSION}_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    ]
)

class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

logger = logging.getLogger(__name__)
logger.info(f"=== Logging started ===")
logger.info(f"Logs will be saved to: {log_filename}")


# ------------------------------------------------------------------------
# MODEL INITIALIZATION
# ------------------------------------------------------------------------

# Get model configuration
model_type = get_model_type(MODEL_VERSION)
model_config = get_model_config(model_type)

# Initialize unified inference engine
inference_engine = UnifiedInference(MODEL_VERSION, MODEL_PATH)

# Log model info
model_info = inference_engine.get_model_info()
# logger.info(f"Model: {model_info['name']} - {model_info['description']}")
# logger.info(f"Features: {model_info['num_features']} ({model_info['acc_features']} ACC + {model_info['baro_features']} BARO)")


# ------------------------------------------------------------------------
# FLASK APP
# ------------------------------------------------------------------------

app = Flask(__name__, static_folder='app/static')

# Global queue for fall notifications (for SSE)
fall_notification_queue = queue.Queue()

# Global monitor instance
continuous_monitor = None


def convert_to_dataframe(acc_data: np.ndarray, acc_time: np.ndarray,
                          acc_scale_factor: float = 1.0 / 16384.0) -> pd.DataFrame:
    """Convert accelerometer arrays to DataFrame format."""
    acc_x = acc_data[0] * acc_scale_factor
    acc_y = acc_data[1] * acc_scale_factor
    acc_z = acc_data[2] * acc_scale_factor

    return pd.DataFrame({
        'Device_Timestamp_[ms]': acc_time,
        'Acc_X[g]': acc_x,
        'Acc_Y[g]': acc_y,
        'Acc_Z[g]': acc_z
    })


def extract_window(df: pd.DataFrame, required_samples: int,
                   pressure: np.ndarray = None,
                   pressure_time: np.ndarray = None):
    """Extract detection window from data."""
    if len(df) < required_samples:
        raise ValueError(f"Insufficient ACC data: need {required_samples}, got {len(df)}")

    # Take most recent ACC samples
    window_df = df.tail(required_samples).copy().reset_index(drop=True)

    # Align barometer data to ACC window
    windowed_pressure = None
    windowed_pressure_time = None

    if pressure is not None and len(pressure) > 0:
        window_start_ms = window_df['Device_Timestamp_[ms]'].iloc[0]
        window_end_ms = window_df['Device_Timestamp_[ms]'].iloc[-1]
        mask = (pressure_time >= window_start_ms) & (pressure_time <= window_end_ms)
        windowed_pressure = pressure[mask]
        windowed_pressure_time = pressure_time[mask]

    return window_df, windowed_pressure, windowed_pressure_time


def export_detection_data(flux_records, is_fall: bool, confidence: float,
                          participant_name: str, participant_gender: str,
                          ground_truth_fall: int, timestamp_utc: datetime):
    """Export detection data to CSV."""
    try:
        if not FALL_DATA_EXPORT_ENABLED:
            return

        timestamp_local = timestamp_utc + timedelta(hours=TIMEZONE_OFFSET_HOURS)
        date_str = timestamp_local.strftime('%Y%m%d')
        sensor_mode = str(COLLECT_ADDITIONAL_SENSORS)

        base_dir = Path("results/fall_data_exports") / date_str / sensor_mode / participant_name
        base_dir.mkdir(parents=True, exist_ok=True)

        prefix = "fall" if is_fall else "no_fall"
        filename = f"{prefix}_{timestamp_local.strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = base_dir / filename

        records_list = []
        for record in flux_records:
            records_list.append({
                'time': record.get_time(),
                'field': record.get_field(),
                'value': record.get_value(),
                'measurement': record.get_measurement() if hasattr(record, 'get_measurement') else 'imu',
                'prediction': 1 if is_fall else 0
            })

        df = pd.DataFrame(records_list)

        metadata_df = pd.DataFrame([{
            'detection_timestamp_utc': timestamp_utc.isoformat(),
            'detection_timestamp_local': timestamp_local.isoformat(),
            'timezone_offset_hours': TIMEZONE_OFFSET_HOURS,
            'confidence': confidence,
            'fall_detected': is_fall,
            'ground_truth_fall': ground_truth_fall,
            'participant_name': participant_name,
            'participant_gender': participant_gender,
            'model_type': MODEL_VERSION
        }])

        with open(filepath, 'w', newline='') as f:
            f.write("# Detection Metadata\n")
            metadata_df.to_csv(f, index=False)
            f.write("\n# Sensor Data\n")
            df.to_csv(f, index=False)

    except Exception as e:
        logger.error(f"Error exporting detection data: {e}", exc_info=True)


@app.route('/')
def index() -> Response:
    """Serve the frontend HTML page."""
    return send_from_directory('app/static', 'index.html')


@app.route('/trigger', methods=['POST'])
def trigger() -> Tuple[Response, int]:
    """
    Main endpoint for fall detection - automatically uses configured model.

    Supports both InfluxDB (real-time) and CSV (batch validation) data sources.
    Set DATA_SOURCE in .env to switch between them.
    """
    # Check data source
    if DATA_SOURCE == 'csv':
        # CSV mode - run batch validation
        if not CSV_FILE_PATH:
            return jsonify({
                "message": "CSV mode selected but CSV_FILE_PATH not set in .env",
                "error": "Configuration error"
            }), 400

        result = process_csv_file(CSV_FILE_PATH, CSV_WINDOW_INTERVAL_SECONDS, inference_engine, MODEL_VERSION)

        if result['success']:
            return jsonify({
                "message": f"CSV validation completed ({model_info['name']})",
                "data_source": "csv",
                "model_version": MODEL_VERSION,
                "model_name": model_info['name'],
                "csv_path": result['csv_path'],
                "total_windows": result['total_windows'],
                "falls_detected": result['falls_detected'],
                "output_file": result['output_file']
            }), 200
        else:
            return jsonify({
                "message": "CSV validation failed",
                "error": result.get('error', 'Unknown error')
            }), 500

    # InfluxDB mode - real-time detection
    request_data = request.get_json() if request.is_json else {}
    participant_name = request_data.get('participant_name', 'unknown')
    participant_gender = request_data.get('participant_gender', 'unknown')
    ground_truth_fall = request_data.get('ground_truth_fall', 0)

    print("="*70)
    print(f"FALL DETECTION PIPELINE - {model_info['name']}")
    print("="*70)
    print(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {model_info['description']}")
    print(f"Features: {model_info['num_features']} ({model_info['acc_features']} ACC + {model_info['baro_features']} BARO)")
    print(f"Participant: {participant_name} ({participant_gender})")
    print(f"Ground Truth: {ground_truth_fall}")
    print("="*70)

    try:
        # ========================================================================
        # STEP 1: FETCH DATA FROM INFLUXDB
        # ========================================================================
        query_start_time = datetime.now(timezone.utc)
        model_logger.log_data_fetch_start(query_start_time)

        # Build query based on hardware mode and model requirements
        # Use configurable field names based on HARDWARE_MODE (50hz uses bosch_acc_*, 100hz uses acc_*)
        fields_filter = f'r["_field"] == "{ACC_FIELD_X}" or r["_field"] == "{ACC_FIELD_Y}" or r["_field"] == "{ACC_FIELD_Z}"'

        # In 100Hz mode, barometer is disabled - only add if available and model needs it
        if BAROMETER_ENABLED and inference_engine.uses_barometer():
            fields_filter += f' or r["_field"] == "{BAROMETER_FIELD}"'

        # Always include ground_truth field so it appears in exported CSVs
        fields_filter += ' or r["_field"] == "ground_truth"'

        query = f'''from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: -30s)
          |> filter(fn: (r) => {fields_filter})
        '''

        tables = fetch_data(query)

        flux_objects = []
        for table in tables:
            for record in table.records:
                flux_objects.append(record)

        model_logger.log_data_fetch_complete(len(tables), len(flux_objects))

        if len(flux_objects) == 0:
            error_msg = "No data found in InfluxDB for this time range"
            model_logger.log_error(error_msg)
            return jsonify({
                "message": error_msg,
                "error": "No sensor data available"
            }), 404

        # ========================================================================
        # STEP 2: PREPROCESS DATA
        # ========================================================================
        model_logger.log_preprocessing_start()

        try:
            # Extract accelerometer data using configurable field names
            acc_data, acc_time = preprocess_acc(
                flux_objects,
                acc_field_x=ACC_FIELD_X,
                acc_field_y=ACC_FIELD_Y,
                acc_field_z=ACC_FIELD_Z
            )

            original_acc_samples = acc_data.shape[1]
            acc_duration = (acc_time[-1] - acc_time[0]) / 1000.0 if len(acc_time) > 1 else 0

            # Resample if needed (25Hz->50Hz upsample or 100Hz->50Hz downsample)
            if RESAMPLING_ENABLED:
                resampler = AccelerometerResampler(
                    source_rate=HARDWARE_ACC_SAMPLE_RATE,
                    target_rate=MODEL_ACC_SAMPLE_RATE,
                    method=RESAMPLING_METHOD
                )
                acc_data, acc_time = resampler.process(acc_data, acc_time)
                resample_type = "Upsampled" if UPSAMPLING_ENABLED else "Downsampled"
                print(f"  {resample_type}: {original_acc_samples} -> {acc_data.shape[1]} samples ({HARDWARE_ACC_SAMPLE_RATE}Hz -> {MODEL_ACC_SAMPLE_RATE}Hz)")

            # Apply sensor calibration if using non_bosch sensor
            # This transforms non_bosch values to bosch-equivalent values
            if SENSOR_CALIBRATION_ENABLED:
                acc_data = calibrate_non_bosch_to_bosch(acc_data)
                print(f"  Calibration: Applied non_bosch -> bosch transformation")

            # Extract barometer data if model uses it and barometer is available
            pressure = np.array([])
            pressure_time = np.array([])

            if BAROMETER_ENABLED and inference_engine.uses_barometer():
                pressure, pressure_time = preprocess_barometer(flux_objects, BAROMETER_FIELD)

            print(f"  ACC samples: {acc_data.shape[1]}, duration: {acc_duration:.1f}s")
            print(f"  BARO samples: {len(pressure)}")

            model_logger.log_preprocessing_complete(
                acc_samples=acc_data.shape[1],
                duration=acc_duration
            )

        except Exception as e:
            model_logger.log_error("Error preprocessing data", e)
            return jsonify({
                "message": "Data preprocessing failed",
                "error": str(e)
            }), 500

        # ========================================================================
        # STEP 3: CONVERT TO DATAFRAME
        # ========================================================================
        model_logger.log_dataframe_conversion_start()

        try:
            acc_scale_factor = 1.0 / 16384.0
            df = convert_to_dataframe(acc_data, acc_time, acc_scale_factor)

            time_diffs = df['Device_Timestamp_[ms]'].diff().dropna()
            actual_acc_rate = 1000 / time_diffs.median()

            print(f"  ACC sampling rate: {actual_acc_rate:.1f} Hz")

            model_logger.log_dataframe_conversion_complete(df, actual_acc_rate)

        except Exception as e:
            model_logger.log_error("Error converting to DataFrame", e)
            return jsonify({
                "message": "DataFrame conversion failed",
                "error": str(e)
            }), 500

        # ========================================================================
        # STEP 4: EXTRACT WINDOW
        # ========================================================================
        model_logger.log_windowing_start()

        try:
            required_samples = int(WINDOW_SIZE_SECONDS * ACC_SAMPLE_RATE)

            window_df, window_pressure, window_pressure_time = extract_window(
                df, required_samples, pressure, pressure_time
            )

            baro_count = len(window_pressure) if window_pressure is not None else 0
            print(f"  Window: {len(window_df)} ACC samples, {baro_count} BARO samples")

            model_logger.log_windowing_complete(len(window_df))

        except Exception as e:
            model_logger.log_error("Error extracting window", e)
            return jsonify({
                "message": "Window extraction failed",
                "error": str(e)
            }), 500

        # ========================================================================
        # STEP 5: RUN INFERENCE (automatic preprocessing based on model)
        # ========================================================================
        model_logger.log_inference_start()

        try:
            result = inference_engine.predict(
                window_df,
                pressure=window_pressure,
                pressure_timestamps=window_pressure_time
            )

            is_fall = result['is_fall']
            confidence = result['confidence']
            features_dict = result['features']

            model_logger.log_feature_extraction_complete(
                num_features=len(features_dict),
                sample_features=features_dict
            )

            threshold = model_config.threshold

            model_logger.log_inference_complete(is_fall, confidence, threshold)

        except Exception as e:
            model_logger.log_error("Error during prediction", e)
            return jsonify({
                "message": "Model prediction failed",
                "error": str(e)
            }), 500

        # ========================================================================
        # SUMMARY
        # ========================================================================
        model_logger.log_summary(
            num_records=len(flux_objects),
            sampling_rate=actual_acc_rate,
            num_samples=len(window_df),
            is_fall=is_fall,
            confidence=confidence
        )

        # Export detection data
        export_detection_data(
            flux_records=flux_objects,
            is_fall=is_fall,
            confidence=confidence,
            participant_name=participant_name,
            participant_gender=participant_gender,
            ground_truth_fall=ground_truth_fall,
            timestamp_utc=query_start_time
        )

        # Prepare result message
        if is_fall:
            if confidence > 0.75:
                result_message = "High confidence fall detection"
            elif confidence > 0.60:
                result_message = "Moderate confidence fall detection"
            else:
                result_message = "Low confidence fall detection"
        else:
            if confidence > 0.40:
                result_message = "Close to threshold - consider manual review"
            elif confidence > 0.25:
                result_message = "Borderline case"
            else:
                result_message = "Clear negative"

        return jsonify({
            "message": f"Fall detection completed ({model_info['name']}).",
            "data_source": "influx",
            "fall_detected": is_fall,
            "model_version": MODEL_VERSION,
            "model_name": model_info['name'],
            "result": result_message,
            "confidence": float(confidence),
            "threshold": float(threshold),
            "sampling_rate": float(actual_acc_rate),
            "window_size": len(window_df),
            "window_duration_seconds": WINDOW_SIZE_SECONDS,
            "num_features": model_info['num_features'],
            "acc_features": model_info['acc_features'],
            "baro_features": model_info['baro_features'],
            "baro_samples": baro_count,
            "participant_name": participant_name,
            "participant_gender": participant_gender,
            "ground_truth_fall": ground_truth_fall
        }), 200

    except ConnectionError as e:
        logger.error(f"Database connection error: {str(e)}")
        return jsonify({
            "message": "Failed to connect to InfluxDB.",
            "error": str(e)
        }), 503
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            "message": "An error occurred during fall detection.",
            "error": str(e)
        }), 500

    finally:
        logger.info("=== Fall detection process completed ===")


@app.route('/validate-csv', methods=['POST'])
def validate_csv() -> Tuple[Response, int]:
    """
    Endpoint for validating a specific CSV file.

    Request JSON:
        {
            "csv_path": "path/to/file.csv",  # optional, uses env if not provided
            "interval_seconds": 1.0  # optional
        }
    """
    request_data = request.get_json() if request.is_json else {}

    csv_path = request_data.get('csv_path', CSV_FILE_PATH)
    interval = request_data.get('interval_seconds', CSV_WINDOW_INTERVAL_SECONDS)

    if not csv_path:
        return jsonify({
            "message": "No CSV path provided",
            "error": "csv_path required in request or CSV_FILE_PATH in .env"
        }), 400

    result = process_csv_file(csv_path, interval, inference_engine, MODEL_VERSION)

    if result['success']:
        return jsonify({
            "message": "CSV validation completed",
            "model_version": MODEL_VERSION,
            "model_name": model_info['name'],
            "csv_path": result['csv_path'],
            "total_windows": result['total_windows'],
            "falls_detected": result['falls_detected'],
            "output_file": result['output_file'],
            "fall_windows": [r for r in result['results'] if r['is_fall']]
        }), 200
    else:
        return jsonify({
            "message": "CSV validation failed",
            "error": result.get('error', 'Unknown error')
        }), 500


@app.route('/model/info', methods=['GET'])
def get_model_info_endpoint():
    """Get information about the currently loaded model."""
    return jsonify(inference_engine.get_model_info())


@app.route('/recording/state', methods=['POST'])
def update_recording_state():
    """Update recording state and participant information."""
    from app.recording_state import recording_state

    try:
        data = request.get_json()
        recording_active = data.get('recording_active', False)
        participant_name = data.get('participant_name', 'unknown')
        participant_gender = data.get('participant_gender', 'unknown')
        ground_truth_fall = data.get('ground_truth_fall', 0)

        recording_state.update_participant_name(participant_name)
        recording_state.update_participant_gender(participant_gender)
        recording_state.update_ground_truth(ground_truth_fall)
        recording_state.set_recording_active(recording_active)

        logger.info(f"Recording state updated: active={recording_active}, name={participant_name}")

        return jsonify({
            'message': 'Recording state updated',
            'state': recording_state.get_current_state()
        }), 200

    except Exception as e:
        logger.error(f"Error updating recording state: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/ground_truth/marker', methods=['POST'])
def write_ground_truth_marker():
    """
    Write ground truth marker directly to InfluxDB at current timestamp.
    This is called when user presses the ground truth button.
    """
    from app.ground_truth_writer import write_ground_truth_marker as write_marker

    try:
        data = request.get_json()
        value = data.get('value', 1)  # 1 = fall event, 0 = no fall

        # Write marker to InfluxDB at current timestamp
        success = write_marker(value)

        if success:
            logger.info(f"Ground truth marker written to InfluxDB: value={value}")
            return jsonify({
                'message': 'Ground truth marker written',
                'value': value,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 200
        else:
            return jsonify({
                'error': 'Failed to write ground truth marker'
            }), 500

    except Exception as e:
        logger.error(f"Error writing ground truth marker: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/events')
def sse_stream():
    """Server-Sent Events stream for real-time fall notifications."""
    def event_stream():
        while True:
            try:
                fall_data = fall_notification_queue.get(timeout=30)

                event_data = json.dumps({
                    'fall_detected': fall_data.get('is_fall', False),
                    'timestamp': fall_data['timestamp'],
                    'confidence': fall_data.get('confidence', 0),
                    'model_type': MODEL_VERSION,
                    'message': f"Fall detected with {fall_data.get('confidence', 0):.0%} confidence" if fall_data.get('is_fall', False) else "No fall detected"
                })

                yield f"data: {event_data}\n\n"

            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return Response(event_stream(), mimetype='text/event-stream')


@app.route('/monitoring/status', methods=['GET'])
def monitoring_status():
    """Get current monitoring status."""
    return jsonify({
        'monitoring_enabled': MONITORING_ENABLED,
        'is_running': continuous_monitor is not None and getattr(continuous_monitor, 'is_running', False),
        'data_source': DATA_SOURCE,
        'csv_file': CSV_FILE_PATH if DATA_SOURCE == 'csv' else None,
        'model_version': MODEL_VERSION,
        'model_name': model_info['name'],
        'uses_barometer': model_info['uses_barometer'] and BAROMETER_ENABLED,
        'acc_sample_rate': ACC_SAMPLE_RATE,
        'baro_sample_rate': BARO_SAMPLE_RATE,
        'features': model_info['num_features'],
        # Sensor and resampling info
        'acc_sensor_type': ACC_SENSOR_TYPE,
        'hardware_acc_rate': HARDWARE_ACC_SAMPLE_RATE,
        'model_acc_rate': MODEL_ACC_SAMPLE_RATE,
        'barometer_enabled': BAROMETER_ENABLED,
        'upsampling_enabled': UPSAMPLING_ENABLED,
        'downsampling_enabled': DOWNSAMPLING_ENABLED,
        'resampling_method': RESAMPLING_METHOD if RESAMPLING_ENABLED else None,
        'acc_fields': [ACC_FIELD_X, ACC_FIELD_Y, ACC_FIELD_Z],
        'window_samples': WINDOW_SAMPLES,
    })


@app.route('/monitoring/stop', methods=['POST'])
def stop_monitoring():
    """Stop continuous monitoring."""
    global continuous_monitor

    if not continuous_monitor or not getattr(continuous_monitor, 'is_running', False):
        return jsonify({'message': 'Monitoring not running'}), 200

    continuous_monitor.stop()
    return jsonify({'message': 'Continuous monitoring stopped'}), 200


if __name__ == '__main__':
    # Print configuration
    # print_config()
    logger.info(f"="*70)
    logger.info(f"Starting Fall Detection System")
    logger.info(f"  Data Source:     {DATA_SOURCE}")
    logger.info(f"  Monitoring Interval: {MONITORING_INTERVAL_SECONDS} seconds")
    if DATA_SOURCE == 'csv':
        logger.info(f"  CSV Path:        {CSV_FILE_PATH}")
        logger.info(f"  Window Interval: {CSV_WINDOW_INTERVAL_SECONDS} seconds")
    logger.info(f"  Model Version:   {MODEL_VERSION.upper()}")
    # logger.info(f"  Model Name:      {model_info['name']}")
    logger.info(f"  Model Path:      {MODEL_PATH}")
    logger.info(f"  ACC Sensor Type: {ACC_SENSOR_TYPE}")
    logger.info(f"  ACC Fields:      {ACC_FIELD_X}, {ACC_FIELD_Y}, {ACC_FIELD_Z}")
    logger.info(f"  Hardware Rate:   {HARDWARE_ACC_SAMPLE_RATE} Hz")
    logger.info(f"  Model Rate:      {MODEL_ACC_SAMPLE_RATE} Hz")
    if UPSAMPLING_ENABLED:
        logger.info(f"  Resampling:      Upsampling ({HARDWARE_ACC_SAMPLE_RATE}Hz -> {MODEL_ACC_SAMPLE_RATE}Hz, method={RESAMPLING_METHOD})")
    elif DOWNSAMPLING_ENABLED:
        logger.info(f"  Resampling:      Downsampling ({HARDWARE_ACC_SAMPLE_RATE}Hz -> {MODEL_ACC_SAMPLE_RATE}Hz, method={RESAMPLING_METHOD})")
    else:
        logger.info(f"  Resampling:      Disabled (no conversion needed)")
    logger.info(f"  Window Size:     {WINDOW_SIZE_SECONDS}s ({HARDWARE_WINDOW_SAMPLES} HW -> {WINDOW_SAMPLES} model samples)")
    logger.info(f"  Barometer:       {'Enabled' if BAROMETER_ENABLED else 'Disabled'}")
    logger.info(f"  Uses Barometer:  {model_info['uses_barometer'] and BAROMETER_ENABLED}")
    logger.info(f"  Calibration:     {'Enabled (non_bosch -> bosch)' if SENSOR_CALIBRATION_ENABLED else 'Disabled'}")
    logger.info(f"  ACC Preprocess:  {model_info['acc_preprocessing']}")
    logger.info(f"  BARO Preprocess: {model_info['baro_preprocessing'] if BAROMETER_ENABLED else 'disabled'}")
    logger.info(f"  Features:        {model_info['num_features']} ({model_info['acc_features']} ACC + {model_info['baro_features']} BARO)")
    logger.info(f"="*70)

    # If CSV mode, run validation immediately
    if DATA_SOURCE == 'csv' and CSV_FILE_PATH:
        logger.info(f"\nCSV mode detected - running validation...")
        result = process_csv_file(CSV_FILE_PATH, CSV_WINDOW_INTERVAL_SECONDS, inference_engine, MODEL_VERSION)
        if result['success']:
            logger.info(f"Validation complete. Results saved to: {result['output_file']}")
        else:
            logger.error(f"Validation failed: {result.get('error')}")
    else:
        # InfluxDB mode - start continuous monitoring if enabled
        if MONITORING_ENABLED:
            logger.info(f"\nStarting continuous monitoring...")
            # logger.info(f"  Monitoring Interval: {MONITORING_INTERVAL_SECONDS} seconds")
            

            # Create continuous monitor with notification queue and export callback
            continuous_monitor = ContinuousMonitor(
                inference_engine=inference_engine,
                notification_queue=fall_notification_queue,
                export_callback=export_detection_data
            )

            # Start monitoring
            continuous_monitor.start()
            logger.info("Continuous monitoring started (CSV export requires recording active)")
        else:
            logger.info("\nContinuous monitoring is disabled. Use /trigger endpoint for manual detection.")

        # Start Flask app for real-time mode
        app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False, threaded=True)
