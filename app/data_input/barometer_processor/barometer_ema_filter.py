"""
Barometer signal processing implementation.

This module implements the dual-path EMA filtering approach for extracting
altitude change features from barometer pressure data.

Processing pipeline:
1. Pressure to altitude conversion (barometric formula)
2. Median filtering for spike removal
3. Dual-path EMA filtering (fast + baseline)
4. Height change feature extraction (Δh = h_fast - h_base)
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional, Union
from barometer_config import BarometerConfig


class BarometerProcessor:
    """
    Batch processor for barometer pressure data.

    This class processes arrays of pressure samples to extract altitude features
    that can be used as inputs to machine learning models. It implements the
    dual-path filtering approach where:
    - Fast path (τ_fast): Tracks rapid height changes
    - Baseline path (τ_base): Tracks slow environmental drift
    - Delta_h feature: Isolates short-term movements by subtracting baseline from fast

    The processor outputs processed features (delta_h, h_fast, h_base) without
    making any classification decisions. These features can be combined with
    other sensor data (e.g., IMU) as inputs to an external ML model.

    Examples
    --------
    Basic usage with default configuration:
    >>> from app.barometer_processing import BarometerProcessor, BarometerConfig
    >>> config = BarometerConfig(median_window=5)
    >>> processor = BarometerProcessor(config)
    >>> h_filtered, h_fast, h_base, delta_h = processor.process(pressure_array, timestamps)

    Using with InfluxDB records:
    >>> from app.barometer_processing import BarometerProcessor, BarometerConfig
    >>> from app.data_processor import convert_barometer_influx_to_numpy  # Not yet implemented
    >>> config = BarometerConfig()
    >>> processor = BarometerProcessor(config)
    >>> pressure, timestamps = convert_barometer_influx_to_numpy(flux_records)
    >>> _, _, _, delta_h = processor.process(pressure, timestamps)
    """

    def __init__(self, config: BarometerConfig):
        """
        Initialize the barometer processor.

        Parameters
        ----------
        config : BarometerConfig
            Configuration object with all processing parameters
        """
        self.config = config
        self.p_ref = config.p_ref
        self.median_window = config.median_window
        self.tau_fast = config.tau_fast
        self.tau_base = config.tau_base
        self.dt = config.dt

    def pressure_to_altitude(self, pressure: np.ndarray) -> np.ndarray:
        """
        Convert pressure to relative altitude using barometric formula.

        Formula: h = 44330 * (1 - (P / P_ref)^0.1903)

        This is based on the international barometric formula assuming
        standard atmosphere conditions.

        Parameters
        ----------
        pressure : np.ndarray
            Pressure values in Pascals

        Returns
        -------
        np.ndarray
            Altitude in meters (relative to reference pressure)

        Notes
        -----
        - Altitude is relative, not absolute
        - Assumes standard atmosphere (temperature, humidity)
        - Accuracy: ±0.5m for small altitude changes
        """
        return 44330.0 * (1.0 - (pressure / self.p_ref) ** 0.1903)

    def median_filter(self, data: np.ndarray, window_size: Optional[int] = None) -> np.ndarray:
        """
        Apply median filter to remove sensor spikes.

        Median filtering is effective at removing outliers and single-sample
        glitches common in barometer sensors without introducing phase lag.

        Parameters
        ----------
        data : np.ndarray
            Input altitude data
        window_size : int, optional
            Window size for median filter. If None, uses config value.
            Must be odd (scipy requirement).

        Returns
        -------
        np.ndarray
            Filtered data with spikes removed

        Notes
        -----
        - Window size must be odd (enforced by scipy.signal.medfilt)
        - Larger windows remove more noise but introduce lag
        - Default window=5 (0.2s at 25Hz) is good for most applications
        """
        if window_size is None:
            window_size = self.median_window

        # Ensure odd window size (scipy requirement)
        if window_size % 2 == 0:
            window_size += 1

        return signal.medfilt(data, kernel_size=window_size)

    def ema_filter(self, data: np.ndarray, tau: float, dt: float) -> np.ndarray:
        """
        Apply exponential moving average (EMA) filter.

        EMA is a first-order low-pass filter that smooths data while
        maintaining causality (only uses past data).

        Formula:
            h[k] = h[k-1] + α * (h_raw[k] - h[k-1])
        where:
            α = dt / (τ + dt)

        Parameters
        ----------
        data : np.ndarray
            Input data to filter
        tau : float
            Time constant in seconds (controls filter responsiveness)
        dt : float
            Sampling interval in seconds

        Returns
        -------
        np.ndarray
            Filtered data

        Notes
        -----
        - Larger τ = slower response, smoother output
        - After time τ, filter reaches 63% of step change
        - After time 3τ, filter reaches 95% of step change
        """
        alpha = dt / (tau + dt)
        filtered = np.zeros_like(data)
        filtered[0] = data[0]

        for i in range(1, len(data)):
            filtered[i] = filtered[i-1] + alpha * (data[i] - filtered[i-1])

        return filtered

    def process(
        self,
        pressure: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        update_baseline: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process pressure data to extract altitude features.

        This is the main processing method that implements the complete pipeline:
        1. Convert pressure to altitude (barometric formula)
        2. Apply median filter (spike removal)
        3. Apply fast EMA (track quick changes)
        4. Apply baseline EMA (track slow drift)
        5. Compute Δh = h_fast - h_base (height change feature)

        Parameters
        ----------
        pressure : np.ndarray
            Pressure values in Pascals
            Shape: (N,) where N is number of samples
        timestamps : np.ndarray, optional
            Timestamps for each sample (datetime64 or float in ms)
            If provided, dt will be computed automatically.
            If None, uses config.dt
        update_baseline : bool, default=True
            Whether to auto-calibrate reference pressure from initial samples.
            Recommended: True for first call, False for subsequent batches.

        Returns
        -------
        h_filtered : np.ndarray
            Median-filtered altitude (meters)
        h_fast : np.ndarray
            Fast-path EMA filtered altitude (meters)
        h_base : np.ndarray
            Baseline-path EMA filtered altitude (meters)
        delta_h : np.ndarray
            Height change feature (meters)
            Positive = sensor moving up
            Negative = sensor moving down (fall)

        Notes
        -----
        - delta_h is the primary height change feature for ML models
        - delta_h ≈ 0 when stationary (fast and baseline converged)
        - Negative delta_h indicates downward movement
        - After ~90s stationary, delta_h returns to near zero
        - This method does NOT perform fall detection - it only extracts features

        Examples
        --------
        >>> pressure = np.array([101325, 101320, 101315, ...])  # Pa
        >>> timestamps = np.array([0, 40, 80, ...])  # ms
        >>> h_f, h_fast, h_base, delta_h = processor.process(pressure, timestamps)
        >>> # Use delta_h as input to your ML model
        >>> features = {'delta_h': delta_h, 'h_fast': h_fast}
        """
        # Compute sampling interval
        if timestamps is not None:
            if len(timestamps) > 1:
                # Handle different timestamp formats
                if isinstance(timestamps[0], np.datetime64):
                    dt_array = np.diff(timestamps.astype('datetime64[ns]')).astype(float) / 1e9
                else:
                    # Assume milliseconds
                    dt_array = np.diff(timestamps) / 1000.0
                dt = np.median(dt_array)
            else:
                dt = self.dt if self.dt is not None else 0.04
        else:
            dt = self.dt if self.dt is not None else 0.04

        # Auto-calibrate reference pressure if requested
        if update_baseline and len(pressure) > 0:
            # Use median of first samples as reference
            n_samples = min(10, len(pressure))
            self.p_ref = np.median(pressure[:n_samples])

        # Step 1: Convert pressure to altitude
        altitude_raw = self.pressure_to_altitude(pressure)

        # Step 2: Apply median filter to remove spikes
        h_filtered = self.median_filter(altitude_raw)

        # Step 3: Apply fast EMA (tracks quick changes)
        h_fast = self.ema_filter(h_filtered, self.tau_fast, dt)

        # Step 4: Apply baseline EMA (tracks slow drift)
        h_base = self.ema_filter(h_filtered, self.tau_base, dt)

        # Step 5: Compute height change feature
        delta_h = h_fast - h_base

        return h_filtered, h_fast, h_base, delta_h

    def get_config(self) -> BarometerConfig:
        """
        Get current configuration.

        Returns
        -------
        BarometerConfig
            Current processor configuration
        """
        return self.config


class StreamingBarometerProcessor:
    """
    Real-time streaming processor for barometer data.

    This class processes pressure samples one at a time, maintaining internal
    state for real-time applications. It's designed for embedded systems or
    live streaming scenarios where batch processing isn't feasible.

    Unlike BarometerProcessor which processes arrays, this class processes
    individual samples and updates internal filter states incrementally.

    Examples
    --------
    Real-time processing:
    >>> from app.barometer_processing import StreamingBarometerProcessor, BarometerConfig
    >>> config = BarometerConfig(median_window=5, dt=0.04)  # 25Hz
    >>> processor = StreamingBarometerProcessor(config)
    >>>
    >>> # Process samples as they arrive
    >>> for pressure_sample in pressure_stream:
    ...     h_fast, h_base, delta_h = processor.process_sample(pressure_sample)
    ...     # Send delta_h to your ML model for classification
    ...     features = {'delta_h': delta_h, 'h_fast': h_fast}

    With initialization period:
    >>> processor = StreamingBarometerProcessor(config)
    >>> # First 10 samples for initialization
    >>> for i in range(10):
    ...     processor.process_sample(pressure_samples[i])
    >>> # Now ready for feature extraction
    >>> h_fast, h_base, delta_h = processor.process_sample(new_sample)
    """

    def __init__(self, config: BarometerConfig):
        """
        Initialize streaming processor.

        Parameters
        ----------
        config : BarometerConfig
            Configuration object with all processing parameters.
            Note: config.dt must be specified (cannot be None for streaming)
        """
        self.config = config
        self.p_ref = config.p_ref
        self.median_window = config.median_window
        self.tau_fast = config.tau_fast
        self.tau_base = config.tau_base

        # Validate dt is provided
        if config.dt is None:
            raise ValueError(
                "config.dt must be specified for StreamingBarometerProcessor. "
                "Cannot auto-compute from timestamps in streaming mode."
            )
        self.dt = config.dt

        # Compute EMA alpha values
        self.alpha_fast = self.dt / (self.tau_fast + self.dt)
        self.alpha_base = self.dt / (self.tau_base + self.dt)

        # Internal state
        self.median_buffer = []
        self.h_fast = None
        self.h_base = None
        self.is_initialized = False
        self.init_counter = 0
        self.init_samples = 10  # Number of samples for initialization

    def pressure_to_altitude(self, pressure: float) -> float:
        """
        Convert single pressure value to altitude.

        Parameters
        ----------
        pressure : float
            Pressure in Pascals

        Returns
        -------
        float
            Altitude in meters
        """
        return 44330.0 * (1.0 - (pressure / self.p_ref) ** 0.1903)

    def process_sample(self, pressure: float) -> Tuple[float, float, float]:
        """
        Process a single pressure sample.

        This method updates internal filter states and returns the current
        altitude estimates.

        Parameters
        ----------
        pressure : float
            Pressure value in Pascals

        Returns
        -------
        h_fast : float
            Fast-filtered altitude (meters)
        h_base : float
            Baseline altitude (meters)
        delta_h : float
            Height change feature (meters)

        Notes
        -----
        - First 10 samples are used for initialization (returns 0, 0, 0)
        - After initialization, returns meaningful values
        - Maintains median buffer of size median_window
        - Updates EMA filter states incrementally

        Examples
        --------
        >>> h_fast, h_base, delta_h = processor.process_sample(101325.0)
        >>> print(f"Height change: {delta_h:.3f}m")
        """
        # Initialization phase: collect samples to calibrate reference pressure
        if not self.is_initialized:
            self.median_buffer.append(pressure)
            if len(self.median_buffer) > self.median_window:
                self.median_buffer.pop(0)

            self.init_counter += 1
            if self.init_counter >= self.init_samples:
                # Initialize reference pressure and filter states
                self.p_ref = np.median(self.median_buffer)
                h_init = self.pressure_to_altitude(pressure)
                self.h_fast = h_init
                self.h_base = h_init
                self.is_initialized = True

            # Return zeros during initialization
            return 0.0, 0.0, 0.0

        # Maintain median buffer (rolling window)
        self.median_buffer.append(pressure)
        if len(self.median_buffer) > self.median_window:
            self.median_buffer.pop(0)

        # Apply median filter
        p_filtered = np.median(self.median_buffer)

        # Convert to altitude
        h_raw = self.pressure_to_altitude(p_filtered)

        # Update EMA filters incrementally
        self.h_fast = self.h_fast + self.alpha_fast * (h_raw - self.h_fast)
        self.h_base = self.h_base + self.alpha_base * (h_raw - self.h_base)

        # Compute height change feature
        delta_h = self.h_fast - self.h_base

        return self.h_fast, self.h_base, delta_h

    def reset(self):
        """
        Reset processor to initial state.

        Clears all internal buffers and filter states.
        Call this to start fresh processing with new data.
        """
        self.median_buffer = []
        self.h_fast = None
        self.h_base = None
        self.is_initialized = False
        self.init_counter = 0
        self.p_ref = self.config.p_ref

    def is_ready(self) -> bool:
        """
        Check if processor has completed initialization.

        Returns
        -------
        bool
            True if ready for meaningful output, False if still initializing
        """
        return self.is_initialized

    def get_config(self) -> BarometerConfig:
        """
        Get current configuration.

        Returns
        -------
        BarometerConfig
            Current processor configuration
        """
        return self.config
