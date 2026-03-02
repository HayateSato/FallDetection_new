"""
Barometer signal processing implementation based on Jatesiktat & Ang (2017).

This module implements the slope-limit filter approach from the paper:
"An Elderly Fall Detection using a Wrist-worn Accelerometer and Barometer"

Processing pipeline:
1. Slope-limit filter (removes sudden spikes while preserving altitude transitions)
2. Moving average filter (1-second window for smoothing)
3. Feature extraction (pressure_shift, middle_slope, post_fall_slope)

Reference:
    Jatesiktat, P., & Ang, W. T. (2017). An elderly fall detection using a
    wrist-worn accelerometer and barometer. IEEE EMBS.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class PaperBarometerConfig:
    """
    Configuration for paper-based barometer processing.

    Based on parameters from Jatesiktat & Ang (2017).

    Attributes
    ----------
    slope_limit : float
        Maximum allowed change per sample (raw units).
        Default: 25 units/sample at 25Hz (from paper)

    moving_avg_window : float
        Window size for moving average filter in seconds.
        Default: 1.0s (from paper)

    sample_rate : float
        Barometer sampling rate in Hz.
        Default: 25.0 Hz (from paper)

    patch_duration : float
        Duration of pressure patch for feature extraction in seconds.
        Default: 6.0s (3s before + 3s after impact reference point)

    pressure_to_altitude_factor : float
        Conversion factor: 1 raw unit ≈ 2.03mm at sea level (from paper)
        Based on LPS25HB sensor: 1/4096 hPa per raw unit
    """
    slope_limit: float = 25.0  # units/sample
    moving_avg_window: float = 1.0  # seconds
    sample_rate: float = 25.0  # Hz
    patch_duration: float = 6.0  # seconds
    pressure_to_altitude_factor: float = 2.03e-3  # meters per raw unit

    def __post_init__(self):
        if self.slope_limit <= 0:
            raise ValueError(f"slope_limit must be positive, got {self.slope_limit}")
        if self.moving_avg_window <= 0:
            raise ValueError(f"moving_avg_window must be positive, got {self.moving_avg_window}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")


class PaperBarometerProcessor:
    """
    Batch processor implementing the paper's barometer preprocessing approach.

    This processor implements the slope-limit filter followed by moving average
    as described in Jatesiktat & Ang (2017). The approach is specifically designed
    to handle noisy MEMS barometer signals while preserving altitude transition
    signatures during falls.

    Key differences from dual-path EMA approach:
    - Uses slope-limiting instead of median filtering for spike removal
    - Single-path moving average instead of dual-path EMA
    - Designed for event-triggered feature extraction (6-second patches)

    Examples
    --------
    Basic usage:
    >>> config = PaperBarometerConfig(sample_rate=25.0)
    >>> processor = PaperBarometerProcessor(config)
    >>> filtered = processor.process(raw_pressure)

    Feature extraction from 6-second patch:
    >>> features = processor.extract_features(filtered_patch, reference_idx=75)
    """

    def __init__(self, config: PaperBarometerConfig):
        """
        Initialize the paper-based barometer processor.

        Parameters
        ----------
        config : PaperBarometerConfig
            Configuration object with processing parameters
        """
        self.config = config
        self.slope_limit = config.slope_limit
        self.sample_rate = config.sample_rate
        self.ma_window_samples = int(config.moving_avg_window * config.sample_rate)

        # Ensure odd window size for centered moving average
        if self.ma_window_samples % 2 == 0:
            self.ma_window_samples += 1

    def slope_limit_filter(self, data: np.ndarray, slope_limit: Optional[float] = None) -> np.ndarray:
        """
        Apply slope-limit filter to remove sudden spikes.

        This filter limits the rate of change between consecutive samples,
        effectively removing high-frequency spikes while preserving the
        slower altitude transitions that occur during falls (~1.5s transition time).

        Algorithm (from paper):
            if newSample > latestOutput + slopeLimit:
                newFilteredOutput = latestOutput + slopeLimit
            elif newSample < latestOutput - slopeLimit:
                newFilteredOutput = latestOutput - slopeLimit
            else:
                newFilteredOutput = newSample

        Parameters
        ----------
        data : np.ndarray
            Raw pressure data (in raw sensor units or Pascals)
        slope_limit : float, optional
            Maximum change per sample. If None, uses config value.

        Returns
        -------
        np.ndarray
            Slope-limited pressure data

        Notes
        -----
        - With slope_limit=25 at 25Hz, signals changing faster than
          625 units/second are attenuated
        - This preserves fall transitions (~330 units over 1.5s = 220 units/s)
        - Reduces standard deviation of stationary signal from ~184 to ~85 units
        """
        if slope_limit is None:
            slope_limit = self.slope_limit

        if len(data) == 0:
            return data.copy()

        filtered = np.zeros_like(data, dtype=float)
        filtered[0] = data[0]

        for i in range(1, len(data)):
            diff = data[i] - filtered[i-1]
            if diff > slope_limit:
                filtered[i] = filtered[i-1] + slope_limit
            elif diff < -slope_limit:
                filtered[i] = filtered[i-1] - slope_limit
            else:
                filtered[i] = data[i]

        return filtered

    def moving_average_filter(self, data: np.ndarray, window_samples: Optional[int] = None) -> np.ndarray:
        """
        Apply moving average filter for noise reduction.

        Parameters
        ----------
        data : np.ndarray
            Input data (typically slope-limited pressure)
        window_samples : int, optional
            Window size in samples. If None, uses config value (1 second).

        Returns
        -------
        np.ndarray
            Smoothed data

        Notes
        -----
        - Uses 'same' mode to maintain output length
        - Edge effects are present at boundaries
        """
        if window_samples is None:
            window_samples = self.ma_window_samples

        if len(data) < window_samples:
            # Fall back to simple mean for short signals
            return np.full_like(data, np.mean(data), dtype=float)

        # Use uniform filter (moving average)
        kernel = np.ones(window_samples) / window_samples
        filtered = np.convolve(data, kernel, mode='same')

        return filtered

    def process(
        self,
        pressure: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Process raw pressure data through the paper's preprocessing pipeline.

        Pipeline:
        1. Slope-limit filter (spike removal)
        2. Moving average filter (smoothing)

        Parameters
        ----------
        pressure : np.ndarray
            Raw pressure values (sensor units or Pascals)
        timestamps : np.ndarray, optional
            Timestamps for adaptive sample rate calculation.
            If provided, updates internal sample rate estimate.

        Returns
        -------
        np.ndarray
            Filtered pressure data ready for feature extraction

        Examples
        --------
        >>> raw_pressure = np.array([101325, 101320, ...])
        >>> filtered = processor.process(raw_pressure)
        """
        # Optionally update sample rate from timestamps
        if timestamps is not None and len(timestamps) > 1:
            if isinstance(timestamps[0], np.datetime64):
                dt_array = np.diff(timestamps.astype('datetime64[ns]')).astype(float) / 1e9
            else:
                # Assume milliseconds
                dt_array = np.diff(timestamps) / 1000.0
            dt = np.median(dt_array)
            if dt > 0:
                actual_rate = 1.0 / dt
                # Update moving average window for actual sample rate
                self.ma_window_samples = int(self.config.moving_avg_window * actual_rate)
                if self.ma_window_samples % 2 == 0:
                    self.ma_window_samples += 1

        # Step 1: Slope-limit filter
        slope_limited = self.slope_limit_filter(pressure)

        # Step 2: Moving average filter
        filtered = self.moving_average_filter(slope_limited)

        return filtered

    def extract_features(
        self,
        filtered_pressure: np.ndarray,
        reference_idx: int
    ) -> Dict[str, float]:
        """
        Extract the three features from a 6-second pressure patch.

        Features are extracted relative to the reference point (ground impact time),
        which should be at the center of the 6-second window.

        Parameters
        ----------
        filtered_pressure : np.ndarray
            Filtered pressure data (6 seconds = 150 samples at 25Hz)
        reference_idx : int
            Index of the reference point (ground impact) in the array.
            Should be at the center (e.g., index 75 for 150 samples).

        Returns
        -------
        dict
            Features:
            - pressure_shift: Average pressure increase after impact
            - middle_slope: Slope during transition period (linear regression)
            - post_fall_slope: Slope after fall settles (linear regression)

        Notes
        -----
        Feature definitions from paper:
        - pressure_shift: avg(last 2s) - avg(first 2s)
        - middle_slope: linear regression slope from 2.6-4.0s (35 samples)
        - post_fall_slope: linear regression slope from last 2s (50 samples)

        Time references (6s window, reference at 3s):
        - First 2s: samples 0-49 (indices 0:50)
        - Middle slope: samples 65-99 (indices 65:100, i.e., 2.6-4.0s)
        - Last 2s: samples 100-149 (indices 100:150)
        """
        n_samples = len(filtered_pressure)
        samples_per_second = n_samples / self.config.patch_duration

        # Calculate sample ranges based on paper's definitions
        # First 2 seconds
        first_2s_end = int(2.0 * samples_per_second)
        first_2s = filtered_pressure[:first_2s_end]

        # Last 2 seconds
        last_2s_start = int(4.0 * samples_per_second)
        last_2s = filtered_pressure[last_2s_start:]

        # Middle slope region (2.6s to 4.0s from start)
        middle_start = int(2.6 * samples_per_second)
        middle_end = int(4.0 * samples_per_second)
        middle_region = filtered_pressure[middle_start:middle_end]

        # Feature 1: Pressure Shift
        pressure_shift = np.mean(last_2s) - np.mean(first_2s)

        # Feature 2: Middle Slope (linear regression)
        if len(middle_region) > 1:
            middle_x = np.arange(len(middle_region))
            middle_slope, _ = np.polyfit(middle_x, middle_region, 1)
            # Convert to units/second
            middle_slope *= samples_per_second
        else:
            middle_slope = 0.0

        # Feature 3: Post-fall Slope (linear regression)
        if len(last_2s) > 1:
            post_x = np.arange(len(last_2s))
            post_fall_slope, _ = np.polyfit(post_x, last_2s, 1)
            # Convert to units/second
            post_fall_slope *= samples_per_second
        else:
            post_fall_slope = 0.0

        return {
            'pressure_shift': float(pressure_shift),
            'middle_slope': float(middle_slope),
            'post_fall_slope': float(post_fall_slope)
        }

    def pressure_to_altitude_change(self, pressure_shift: float) -> float:
        """
        Convert pressure shift to approximate altitude change.

        Based on paper: 333 raw units ≈ 80cm fall

        Parameters
        ----------
        pressure_shift : float
            Pressure shift in raw sensor units

        Returns
        -------
        float
            Approximate altitude change in meters
        """
        # From paper: 80cm = 333 units, so 1 unit ≈ 2.4mm
        return pressure_shift * (0.80 / 333.0)

    def get_config(self) -> PaperBarometerConfig:
        """Get current configuration."""
        return self.config
