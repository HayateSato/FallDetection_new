"""
Downsampling utilities for converting high-frequency sensor data to model-compatible rates.

When hardware is configured at 100Hz but the model was trained at 50Hz, this module
provides downsampling to maintain compatibility.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


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


def downsample_dataframe(
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


class AccelerometerDownsampler:
    """
    Stateful downsampler for converting 100Hz accelerometer data to 50Hz.

    This class handles the conversion when hardware is configured at 100Hz
    but the fall detection model was trained on 50Hz data.

    Example:
        >>> downsampler = AccelerometerDownsampler(source_rate=100, target_rate=50)
        >>> downsampled_data, downsampled_time = downsampler.process(acc_data, acc_time)
    """

    def __init__(
        self,
        source_rate: float = 100.0,
        target_rate: float = 50.0,
        method: str = 'decimate'
    ):
        """
        Initialize downsampler.

        Args:
            source_rate: Hardware sampling rate in Hz
            target_rate: Model's expected sampling rate in Hz
            method: Downsampling method ('decimate' or 'average')
        """
        self.source_rate = source_rate
        self.target_rate = target_rate
        self.method = method
        self.enabled = source_rate > target_rate

    @property
    def decimation_factor(self) -> int:
        """Get the decimation factor."""
        if not self.enabled:
            return 1
        return int(self.source_rate / self.target_rate)

    def process(
        self,
        acc_data: np.ndarray,
        acc_time: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process accelerometer data through downsampler.

        Args:
            acc_data: shape (3, N) accelerometer data
            acc_time: shape (N,) timestamps in ms

        Returns:
            Tuple of (downsampled_data, downsampled_time)
        """
        if not self.enabled:
            return acc_data, acc_time

        return downsample_accelerometer(
            acc_data, acc_time,
            source_rate=self.source_rate,
            target_rate=self.target_rate,
            method=self.method
        )

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process DataFrame through downsampler.

        Args:
            df: DataFrame with accelerometer data

        Returns:
            Downsampled DataFrame
        """
        if not self.enabled:
            return df

        return downsample_dataframe(
            df,
            source_rate=self.source_rate,
            target_rate=self.target_rate,
            method=self.method
        )

    def get_info(self) -> dict:
        """Get downsampler configuration info."""
        return {
            'enabled': self.enabled,
            'source_rate': self.source_rate,
            'target_rate': self.target_rate,
            'decimation_factor': self.decimation_factor,
            'method': self.method
        }
