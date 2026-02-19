"""
Raw data extraction and format conversion. 
Takes FluxRecord objects from InfluxDB and converts them into numpy arrays (will be sent to detection.py and continuous_monitoring.py).
Data preprocessing module for sensor data transformation.
Converts raw InfluxDB sensor records into the format required by fall detection.

Supports both accelerometer and barometer data preprocessing with switchable
barometer preprocessing versions (V1: dual-path EMA, V2: slope-limit from paper).
"""

from typing import List, Tuple
import numpy as np
from influxdb_client.client.flux_table import FluxRecord
import sys
from pathlib import Path
# Add analysis directory to path for barometer preprocessing imports
_analysis_dir = Path(__file__).parent.parent / 'analysis'
if str(_analysis_dir) not in sys.path:
    sys.path.insert(0, str(_analysis_dir))

def convert_acc_influx_to_numpy(
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


def convert_barometer_influx_to_numpy(
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

