"""
Resampling utilities for converting sensor data to model-compatible rates.

Supports both upsampling (25Hz -> 50Hz) and downsampling (100Hz -> 50Hz)
to ensure accelerometer data matches the model's expected 50Hz sample rate.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from scipy import interpolate

class AccelerometerResampler:
    """
    Unified resampler for converting accelerometer data to model's expected rate.

    Handles both upsampling (25Hz -> 50Hz) and downsampling (100Hz -> 50Hz).

    Example:
        >>> resampler = AccelerometerResampler(source_rate=25, target_rate=50)
        >>> resampled_data, resampled_time = resampler.process(acc_data, acc_time)

        >>> resampler = AccelerometerResampler(source_rate=100, target_rate=50)
        >>> resampled_data, resampled_time = resampler.process(acc_data, acc_time)
    """

    def __init__(
        self,
        source_rate: float = 50.0,
        target_rate: float = 50.0,
        method: str = 'linear'
    ):
        """
        Initialize resampler.

        Args:
            source_rate: Hardware sampling rate in Hz (25, 50, or 100)
            target_rate: Model's expected sampling rate in Hz (typically 50)
            method: Resampling method:
                - For upsampling (25->50): 'linear', 'cubic', 'nearest'
                - For downsampling (100->50): 'decimate', 'average'
        """
        self.source_rate = source_rate
        self.target_rate = target_rate
        self.method = method

        self.upsampling = source_rate < target_rate
        self.downsampling = source_rate > target_rate
        self.enabled = self.upsampling or self.downsampling

    @property
    def resampling_factor(self) -> float:
        """Get the resampling factor."""
        if not self.enabled:
            return 1.0
        return self.target_rate / self.source_rate

    @property
    def resampling_type(self) -> str:
        """Get the resampling type."""
        if self.upsampling:
            return 'upsample'
        elif self.downsampling:
            return 'downsample'
        return 'none'

    def process(
        self,
        acc_data: np.ndarray,
        acc_time: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process accelerometer data through resampler.

        Args:
            acc_data: shape (3, N) accelerometer data
            acc_time: shape (N,) timestamps in ms

        Returns:
            Tuple of (resampled_data, resampled_time)
        """
        if not self.enabled:
            return acc_data, acc_time

        return resample_accelerometer(
            acc_data, acc_time,
            source_rate=self.source_rate,
            target_rate=self.target_rate,
            method=self.method
        )

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process DataFrame through resampler.

        Args:
            df: DataFrame with accelerometer data

        Returns:
            Resampled DataFrame
        """
        if not self.enabled:
            return df

        if self.upsampling:
            return upsample_acc_dataframe(
                df,
                source_rate=self.source_rate,
                target_rate=self.target_rate,
                method=self.method
            )
        else:
            return downsample_acc_dataframe(
                df,
                source_rate=self.source_rate,
                target_rate=self.target_rate,
                method=self.method
            )

    def get_info(self) -> dict:
        """Get resampler configuration info."""
        return {
            'enabled': self.enabled,
            'source_rate': self.source_rate,
            'target_rate': self.target_rate,
            'resampling_type': self.resampling_type,
            'resampling_factor': self.resampling_factor,
            'method': self.method
        }


def upsample_accelerometer(
    acc_data: np.ndarray,
    acc_time: np.ndarray,
    source_rate: float = 25.0,
    target_rate: float = 50.0,
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Upsample accelerometer data from source rate to target rate.

    Args:
        acc_data: Accelerometer data array shape (3, N) for [x, y, z]
        acc_time: Timestamp array shape (N,) in milliseconds
        source_rate: Original sampling rate in Hz (e.g., 25)
        target_rate: Target sampling rate in Hz (e.g., 50)
        method: Upsampling method:
            - 'linear': Linear interpolation between samples
            - 'cubic': Cubic spline interpolation (smoother)
            - 'nearest': Nearest neighbor (duplicates samples)

    Returns:
        Tuple of (upsampled_data, upsampled_time):
        - upsampled_data: shape (3, M) where M = N * (target_rate / source_rate)
        - upsampled_time: shape (M,) timestamps for upsampled data
    """
    if source_rate >= target_rate:
        # No upsampling needed
        return acc_data, acc_time

    if len(acc_time) < 2:
        return acc_data, acc_time

    # Calculate upsampling factor
    factor = target_rate / source_rate  # e.g., 50/25 = 2

    # Create new timestamp array with higher resolution
    start_time = acc_time[0]
    end_time = acc_time[-1]
    duration_ms = end_time - start_time

    # Calculate number of output samples
    n_output = int(len(acc_time) * factor)

    # Generate evenly spaced timestamps
    upsampled_time = np.linspace(start_time, end_time, n_output)

    # Interpolate each axis
    upsampled_data = np.zeros((3, n_output))

    for axis in range(3):
        if method == 'linear':
            interp_func = interpolate.interp1d(
                acc_time, acc_data[axis],
                kind='linear',
                fill_value='extrapolate'
            )
        elif method == 'cubic':
            interp_func = interpolate.interp1d(
                acc_time, acc_data[axis],
                kind='cubic',
                fill_value='extrapolate'
            )
        elif method == 'nearest':
            interp_func = interpolate.interp1d(
                acc_time, acc_data[axis],
                kind='nearest',
                fill_value='extrapolate'
            )
        else:
            raise ValueError(f"Unknown upsampling method: {method}")

        upsampled_data[axis] = interp_func(upsampled_time)

    return upsampled_data, upsampled_time


def downsample_accelerometer(
    acc_data: np.ndarray,
    acc_time: np.ndarray,
    source_rate: float = 100.0,
    target_rate: float = 50.0,
    method: str = 'decimate'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample accelerometer data from source rate to target rate.

    Args:
        acc_data: Accelerometer data array shape (3, N) for [x, y, z]
        acc_time: Timestamp array shape (N,) in milliseconds
        source_rate: Original sampling rate in Hz (e.g., 100)
        target_rate: Target sampling rate in Hz (e.g., 50)
        method: Downsampling method:
            - 'decimate': Simple decimation (take every Nth sample)
            - 'average': Average consecutive samples

    Returns:
        Tuple of (downsampled_data, downsampled_time):
        - downsampled_data: shape (3, M) where M = N / (source_rate / target_rate)
        - downsampled_time: shape (M,) timestamps for downsampled data
    """
    if source_rate <= target_rate:
        # No downsampling needed
        return acc_data, acc_time

    # Calculate decimation factor
    factor = int(source_rate / target_rate)

    if method == 'decimate':
        # Simple decimation - take every Nth sample
        downsampled_data = acc_data[:, ::factor]
        downsampled_time = acc_time[::factor]

    elif method == 'average':
        # Average consecutive samples
        n_samples = acc_data.shape[1]
        n_output = n_samples // factor

        # Reshape and average
        trimmed_data = acc_data[:, :n_output * factor]
        downsampled_data = trimmed_data.reshape(3, n_output, factor).mean(axis=2)

        # Take first timestamp of each group
        trimmed_time = acc_time[:n_output * factor]
        downsampled_time = trimmed_time.reshape(n_output, factor)[:, 0]

    else:
        raise ValueError(f"Unknown downsampling method: {method}")

    return downsampled_data, downsampled_time


def resample_accelerometer(
    acc_data: np.ndarray,
    acc_time: np.ndarray,
    source_rate: float,
    target_rate: float = 50.0,
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample accelerometer data to target rate (upsample or downsample as needed).

    Args:
        acc_data: Accelerometer data array shape (3, N) for [x, y, z]
        acc_time: Timestamp array shape (N,) in milliseconds
        source_rate: Original sampling rate in Hz (25, 50, or 100)
        target_rate: Target sampling rate in Hz (default 50)
        method: Resampling method:
            - For upsampling: 'linear', 'cubic', 'nearest'
            - For downsampling: 'decimate', 'average'

    Returns:
        Tuple of (resampled_data, resampled_time)
    """
    if source_rate < target_rate:
        # Upsample
        return upsample_accelerometer(acc_data, acc_time, source_rate, target_rate, method)
    elif source_rate > target_rate:
        # Downsample
        return downsample_accelerometer(acc_data, acc_time, source_rate, target_rate, method)
    else:
        # No resampling needed
        return acc_data, acc_time


def upsample_acc_dataframe(
    df: pd.DataFrame,
    source_rate: float = 25.0,
    target_rate: float = 50.0,
    method: str = 'linear'
) -> pd.DataFrame:
    """
    Upsample a DataFrame with accelerometer data.

    Args:
        df: DataFrame with columns ['Device_Timestamp_[ms]', 'Acc_X[g]', 'Acc_Y[g]', 'Acc_Z[g]']
        source_rate: Original sampling rate in Hz
        target_rate: Target sampling rate in Hz
        method: Upsampling method ('linear', 'cubic', 'nearest')

    Returns:
        Upsampled DataFrame with same columns
    """
    if source_rate >= target_rate:
        return df

    if len(df) < 2:
        return df

    # Extract data
    times = df['Device_Timestamp_[ms]'].values
    acc_x = df['Acc_X[g]'].values
    acc_y = df['Acc_Y[g]'].values
    acc_z = df['Acc_Z[g]'].values

    # Stack as (3, N)
    acc_data = np.array([acc_x, acc_y, acc_z])

    # Upsample
    upsampled_data, upsampled_time = upsample_accelerometer(
        acc_data, times, source_rate, target_rate, method
    )

    return pd.DataFrame({
        'Device_Timestamp_[ms]': upsampled_time,
        'Acc_X[g]': upsampled_data[0],
        'Acc_Y[g]': upsampled_data[1],
        'Acc_Z[g]': upsampled_data[2],
    })


def downsample_acc_dataframe(
    df: pd.DataFrame,
    source_rate: float = 100.0,
    target_rate: float = 50.0,
    method: str = 'decimate'
) -> pd.DataFrame:
    """
    Downsample a DataFrame with accelerometer data.

    Args:
        df: DataFrame with columns ['Device_Timestamp_[ms]', 'Acc_X[g]', 'Acc_Y[g]', 'Acc_Z[g]']
        source_rate: Original sampling rate in Hz
        target_rate: Target sampling rate in Hz
        method: Downsampling method ('decimate' or 'average')

    Returns:
        Downsampled DataFrame with same columns
    """
    if source_rate <= target_rate:
        return df

    factor = int(source_rate / target_rate)

    if method == 'decimate':
        # Simple decimation
        return df.iloc[::factor].reset_index(drop=True)

    elif method == 'average':
        # Average consecutive samples
        n_output = len(df) // factor

        result_rows = []
        for i in range(n_output):
            start_idx = i * factor
            end_idx = start_idx + factor
            group = df.iloc[start_idx:end_idx]

            result_rows.append({
                'Device_Timestamp_[ms]': group['Device_Timestamp_[ms]'].iloc[0],
                'Acc_X[g]': group['Acc_X[g]'].mean(),
                'Acc_Y[g]': group['Acc_Y[g]'].mean(),
                'Acc_Z[g]': group['Acc_Z[g]'].mean(),
            })

        return pd.DataFrame(result_rows)

    else:
        raise ValueError(f"Unknown downsampling method: {method}")
