"""
Data preprocessing module for sensor data transformation.
Converts raw InfluxDB sensor records into the format required by fall detection.

Supports both accelerometer and barometer data preprocessing with switchable
barometer preprocessing versions (V1: dual-path EMA, V2: slope-limit from paper).
"""

from typing import List, Tuple, Optional, Union
import numpy as np
from influxdb_client.client.flux_table import FluxRecord

import sys
from pathlib import Path
# Add analysis directory to path for barometer preprocessing imports
_analysis_dir = Path(__file__).parent.parent / 'analysis'
if str(_analysis_dir) not in sys.path:
    sys.path.insert(0, str(_analysis_dir))


def filter_imu_data(records: List[FluxRecord]) -> Tuple[List[FluxRecord], List[FluxRecord]]:
    """
    Filter records to separate IMU data from other sensor data.

    Args:
        records: List of all FluxRecord objects from InfluxDB query

    Returns:
        Tuple of (imu_records, all_records):
        - imu_records: List containing only acc_x/y/z and gyr_x/y/z records
        - all_records: Original list of all records (unchanged)
    """
    # IMU field names to keep for fall detection
    imu_fields = {
        'bosch_acc_x',
        'bosch_acc_y',
        'bosch_acc_z',
        'bosch_gyr_x',
        'bosch_gyr_y',
        'bosch_gyr_z'
    }

    # Filter records to only include IMU data
    imu_records = [
        record for record in records
        if record.get_field() in imu_fields
    ]

    return imu_records, records


def preprocess_acc(
    records: List[FluxRecord],
    acc_field_x: str = 'bosch_acc_x',
    acc_field_y: str = 'bosch_acc_y',
    acc_field_z: str = 'bosch_acc_z'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess InfluxDB records for accelerometer-only models.

    Supports configurable field names for different hardware modes:
    - 50Hz mode: bosch_acc_x/y/z (default)
    - 100Hz mode: acc_x/y/z

    Args:
        records: List of FluxRecord objects from InfluxDB query
        acc_field_x: Field name for X axis accelerometer
        acc_field_y: Field name for Y axis accelerometer
        acc_field_z: Field name for Z axis accelerometer

    Returns:
        Tuple containing (acc_data_array, acc_time_array):
        - acc_data_array: numpy array shape (3, N) - accelerometer x/y/z data
        - acc_time_array: numpy array shape (N,) - accelerometer timestamps in ms
    """
    # Initialize data structures for accelerometer
    acc_data = {'x': [], 'y': [], 'z': []}
    acc_times = {'x': [], 'y': [], 'z': []}

    # Route each record to appropriate axis
    for record in records:
        field = record.get_field()
        value = record.get_value()
        time = record.get_time().timestamp() * 1000  # Convert to milliseconds

        # Accelerometer data routing based on configurable field names
        if field == acc_field_x:
            acc_data['x'].append(value)
            acc_times['x'].append(time)
        elif field == acc_field_y:
            acc_data['y'].append(value)
            acc_times['y'].append(time)
        elif field == acc_field_z:
            acc_data['z'].append(value)
            acc_times['z'].append(time)

    # Convert to numpy arrays
    # Shape: (3, N) where 3 = [x, y, z] axes and N = number of samples
    acc_data_array = np.array([acc_data['x'], acc_data['y'], acc_data['z']])

    # Use x-axis timestamps as reference
    acc_time_array = np.array(acc_times['x'])

    return acc_data_array, acc_time_array


def preprocess_barometer(
    records: List[FluxRecord],
    barometer_field: str = 'bmp_pressure'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract barometer data from InfluxDB records.

    Args:
        records: List of FluxRecord objects from InfluxDB query
        barometer_field: Field name for barometer pressure data

    Returns:
        Tuple containing (pressure_array, time_array):
        - pressure_array: numpy array shape (N,) - pressure values in Pa
        - time_array: numpy array shape (N,) - timestamps in ms
    """
    pressure_data = []
    pressure_times = []

    for record in records:
        field = record.get_field()
        if field == barometer_field:
            pressure_data.append(record.get_value())
            pressure_times.append(record.get_time().timestamp() * 1000)

    return np.array(pressure_data), np.array(pressure_times)


def preprocess_acc_and_barometer(
    records: List[FluxRecord],
    barometer_version: str = 'v1_ema',
    barometer_field: str = 'bmp_pressure'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Preprocess both accelerometer and barometer data with version-switchable
    barometer preprocessing.

    This function extracts accelerometer data and processes barometer data
    using the specified preprocessing version.

    Args:
        records: List of FluxRecord objects from InfluxDB query
        barometer_version: Barometer preprocessing version to use:
            - 'v1_ema' or 'v1': Dual-path EMA approach (original)
            - 'v2_paper' or 'v2': Slope-limit approach (from paper)
        barometer_field: Field name for barometer pressure data

    Returns:
        Tuple containing:
        - acc_data_array: numpy array shape (3, N) - accelerometer x/y/z
        - acc_time_array: numpy array shape (N,) - accelerometer timestamps in ms
        - baro_features: numpy array - processed barometer features
        - baro_result: dict with full preprocessing result details

    Example:
        >>> acc_data, acc_time, baro_features, baro_result = preprocess_acc_and_barometer(
        ...     records,
        ...     barometer_version='v2_paper'
        ... )
        >>> print(f"Barometer version: {baro_result['version']}")
    """
    # Import here to avoid circular imports
    from app.preprocessor.barometer_preprocessing import BarometerPreprocessingOrchestrator

    # Extract accelerometer data
    acc_data, acc_time = preprocess_acc(records)

    # Extract raw barometer data
    pressure, pressure_time = preprocess_barometer(records, barometer_field)

    # Initialize result dict
    baro_result = {
        'version': barometer_version,
        'raw_pressure': pressure,
        'timestamps': pressure_time,
        'features': {},
        'filtered_signal': np.array([])
    }

    if len(pressure) == 0:
        return acc_data, acc_time, np.array([]), baro_result

    # Process barometer with specified version
    orchestrator = BarometerPreprocessingOrchestrator(version=barometer_version)
    result = orchestrator.process(pressure, pressure_time)

    baro_result['version'] = result.version.value
    baro_result['features'] = result.features
    baro_result['filtered_signal'] = result.filtered_signal
    baro_result['raw_outputs'] = result.raw_outputs

    # Return primary feature array
    baro_features = result.get_primary_feature()

    return acc_data, acc_time, baro_features, baro_result


class SensorDataProcessor:
    """
    Unified sensor data processor with configurable barometer preprocessing.

    This class provides a stateful interface for processing sensor data with
    the ability to switch barometer preprocessing versions at runtime.

    Example:
        >>> processor = SensorDataProcessor(barometer_version='v1_ema')
        >>> result = processor.process(records)

        >>> # Switch to paper's approach
        >>> processor.set_barometer_version('v2_paper')
        >>> result = processor.process(records)

        >>> # Compare both versions
        >>> comparison = processor.compare_barometer_versions(records)
    """

    def __init__(
        self,
        barometer_version: str = 'v1_ema',
        barometer_field: str = 'bmp_pressure'
    ):
        """
        Initialize sensor data processor.

        Args:
            barometer_version: Initial barometer preprocessing version
            barometer_field: InfluxDB field name for barometer data
        """
        self._barometer_version = barometer_version
        self._barometer_field = barometer_field
        self._orchestrator = None

    def _get_orchestrator(self):
        """Lazy initialization of barometer orchestrator."""
        if self._orchestrator is None:
            from app.preprocessor.barometer_preprocessing import BarometerPreprocessingOrchestrator
            self._orchestrator = BarometerPreprocessingOrchestrator(
                version=self._barometer_version
            )
        return self._orchestrator

    def set_barometer_version(self, version: str) -> None:
        """
        Switch barometer preprocessing version.

        Args:
            version: Version to switch to ('v1_ema', 'v2_paper', etc.)
        """
        self._barometer_version = version
        if self._orchestrator is not None:
            self._orchestrator.set_version(version)

    def get_barometer_version(self) -> str:
        """Get current barometer preprocessing version."""
        return self._barometer_version

    def process(
        self,
        records: List[FluxRecord]
    ) -> dict:
        """
        Process sensor records with current configuration.

        Args:
            records: List of FluxRecord objects from InfluxDB

        Returns:
            dict with:
            - 'acc_data': accelerometer data array (3, N)
            - 'acc_time': accelerometer timestamps
            - 'baro_features': processed barometer features
            - 'baro_result': full barometer preprocessing result
        """
        acc_data, acc_time, baro_features, baro_result = preprocess_acc_and_barometer(
            records,
            barometer_version=self._barometer_version,
            barometer_field=self._barometer_field
        )

        return {
            'acc_data': acc_data,
            'acc_time': acc_time,
            'baro_features': baro_features,
            'baro_result': baro_result
        }

    def compare_barometer_versions(
        self,
        records: List[FluxRecord]
    ) -> dict:
        """
        Process barometer data with both versions for comparison.

        Args:
            records: List of FluxRecord objects from InfluxDB

        Returns:
            dict with results from both versions:
            - 'v1_ema': Result from dual-path EMA approach
            - 'v2_paper': Result from slope-limit approach
            - 'raw_pressure': Original pressure data
            - 'timestamps': Pressure timestamps
        """
        from app.preprocessor.barometer_preprocessing import BarometerPreprocessingOrchestrator

        # Extract raw barometer data
        pressure, pressure_time = preprocess_barometer(
            records, self._barometer_field
        )

        if len(pressure) == 0:
            return {
                'v1_ema': None,
                'v2_paper': None,
                'raw_pressure': np.array([]),
                'timestamps': np.array([])
            }

        # Process with both versions
        orchestrator = BarometerPreprocessingOrchestrator()
        comparison = orchestrator.process_both(pressure, pressure_time)

        return {
            'v1_ema': comparison['v1_ema'],
            'v2_paper': comparison['v2_paper'],
            'raw_pressure': pressure,
            'timestamps': pressure_time
        }
