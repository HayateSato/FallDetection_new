import numpy as np
import os
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Dict, Optional
from pathlib import Path

# Import app modules
from app.core.inference_engine import PipelineSelector


# Import settings
from config.settings import (
    WINDOW_SIZE_SECONDS,
    ACC_SENSOR_SENSITIVITY,
)
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
        acc_scale_factor = 1.0 / ACC_SENSOR_SENSITIVITY

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


def process_csv_file(csv_path: str, interval_seconds: float, inference_engine: PipelineSelector, model_version: str) -> Dict:
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
