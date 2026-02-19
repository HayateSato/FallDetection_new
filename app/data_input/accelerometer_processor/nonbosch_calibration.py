"""
Sensor Calibration Module - Transform non_bosch accelerometer data to bosch-equivalent values.

This module provides transformation functions to convert accelerometer readings from
the non_bosch sensor to values that approximate what the bosch sensor would measure.
This allows models trained on bosch sensor data to be used with non_bosch sensor input.

Calibration Method:
    The transformation was computed using least squares regression on time-aligned
    sensor pairs from multiple recording sessions. The transformation is:

        bosch_values = TRANSFORM_MATRIX @ non_bosch_values + TRANSFORM_OFFSET

    Where:
        - TRANSFORM_MATRIX is a 3x3 matrix that handles rotation and scaling
        - TRANSFORM_OFFSET is a 3x1 vector that handles bias differences

Calibration Accuracy:
    Based on 9,991 time-aligned sample pairs from 4 recording sessions:

    Axis    RMSE        R²
    X       1683.9      0.6701
    Y       1392.9      0.3888
    Z       1484.0      0.5222

    Magnitude correlation: 0.4605

    Note: The transformation provides a moderate approximation. The R² values indicate
    that 39-67% of variance in bosch readings can be explained by non_bosch readings.
    This is useful for experimental evaluation but results may not match bosch sensor
    performance exactly.

Usage:
    from app.sensor_calibration import transform_non_bosch_to_bosch, SensorCalibrator

    # Single sample
    bosch_x, bosch_y, bosch_z = transform_non_bosch_to_bosch(acc_x, acc_y, acc_z)

    # Batch transform
    calibrator = SensorCalibrator()
    acc_data_calibrated = calibrator.transform(acc_data)  # Shape: (3, N)

Data Source:
    Calibration computed from:
    - AIDAPT_25hz_Bosch_isa_FALL.csv
    - AIDAPT_25hz_Bosch_isa_FALL_2.csv
    - AIDAPT_25hz_Bosch_Daria_FALL.csv
    - AIDAPT_25hz_Bosch_armSwingUp.csv

Date: 2026-01-20
"""

import numpy as np

# =============================================================================
# CALIBRATION PARAMETERS
# =============================================================================

# Transformation matrix: converts non_bosch (x,y,z) to bosch-equivalent (x,y,z)
# Computed via least squares regression on 9,991 time-aligned sample pairs
TRANSFORM_MATRIX = np.array([
    [-0.680553, -3.208964, -0.447696],
    [+2.166274, +0.479881, -0.110818],
    [+0.337926, -0.280969, -2.802281],
])

# Offset vector: bias correction after rotation/scaling
TRANSFORM_OFFSET = np.array([-768.84, -1065.12, 429.37])

# Calibration metadata
CALIBRATION_INFO = {
    'method': 'least_squares_regression',
    'num_samples': 9991,
    'num_sessions': 4,
    'r2_x': 0.6701,
    'r2_y': 0.3888,
    'r2_z': 0.5222,
    'magnitude_correlation': 0.4605,
    'rmse_x': 1683.9,
    'rmse_y': 1392.9,
    'rmse_z': 1484.0,
    'date': '2026-01-20',
}


# =============================================================================
# TRANSFORMATION FUNCTIONS
# =============================================================================
def transform_acc_array(acc_data: np.ndarray) -> np.ndarray:
    """
    Transform accelerometer data array from non_bosch to bosch-equivalent values.

    Args:
        acc_data: Accelerometer data with shape (3, N) where:
                  - acc_data[0] = X values
                  - acc_data[1] = Y values
                  - acc_data[2] = Z values

    Returns:
        Transformed array with same shape (3, N) containing bosch-equivalent values

    Example:
        acc_data = np.array([
            [x1, x2, x3, ...],  # X axis
            [y1, y2, y3, ...],  # Y axis
            [z1, z2, z3, ...],  # Z axis
        ])
        transformed = transform_acc_array(acc_data)
    """
    if acc_data.shape[0] != 3:
        raise ValueError(f"Expected shape (3, N), got {acc_data.shape}")

    # Apply transformation: bosch = A @ non_bosch + offset
    transformed = TRANSFORM_MATRIX @ acc_data + TRANSFORM_OFFSET.reshape(3, 1)

    return transformed


# def transform_non_bosch_to_bosch(
#     acc_x: Union[float, np.ndarray],
#     acc_y: Union[float, np.ndarray],
#     acc_z: Union[float, np.ndarray]
# ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
#     """
#     Transform non_bosch accelerometer values to bosch-equivalent values.

#     This applies a linear transformation (rotation + scaling + offset) that was
#     computed by aligning time-synced readings from both sensors.

#     Args:
#         acc_x: Non-bosch X-axis value(s) in raw sensor units
#         acc_y: Non-bosch Y-axis value(s) in raw sensor units
#         acc_z: Non-bosch Z-axis value(s) in raw sensor units

#     Returns:
#         Tuple of (bosch_x, bosch_y, bosch_z) in bosch-equivalent units

#     Example:
#         # Single value
#         bx, by, bz = transform_non_bosch_to_bosch(-500, 270, -180)

#         # Batch (numpy arrays)
#         bx, by, bz = transform_non_bosch_to_bosch(
#             np.array([-500, -510, -505]),
#             np.array([270, 275, 268]),
#             np.array([-180, -175, -182])
#         )
#     """
#     # Stack into matrix form
#     non_bosch = np.array([acc_x, acc_y, acc_z])

#     # Apply transformation: bosch = A @ non_bosch + offset
#     if non_bosch.ndim == 1:
#         # Single sample: (3,)
#         bosch = TRANSFORM_MATRIX @ non_bosch + TRANSFORM_OFFSET
#     else:
#         # Batch: (3, N)
#         bosch = TRANSFORM_MATRIX @ non_bosch + TRANSFORM_OFFSET.reshape(3, 1)

#     return bosch[0], bosch[1], bosch[2]
