"""
Configuration dataclass for barometer processing parameters.

This module provides a clean, type-safe way to configure the barometer processor
with validation and documentation of all parameters.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class BarometerConfig:
    """
    Configuration for barometer signal processing.

    This dataclass encapsulates all parameters needed for the dual-path EMA
    filtering approach used for extracting height change features from barometer data.

    These features (delta_h, h_fast, h_base) can be used as inputs to external
    machine learning models for fall detection, activity recognition, etc.

    Attributes
    ----------
    median_window : int
        Window size for median filter (must be odd).
        Larger values = more noise reduction but slower response.
        Default: 5 samples (0.2s at 25Hz)
        Recommended range: 3-11 for 25Hz sampling

    tau_fast : float
        Time constant for fast EMA path in seconds.
        Controls how quickly the fast path responds to height changes.
        Default: 0.5s (responds in ~1.5s)
        Recommended range: 0.3-0.8s

    tau_base : float
        Time constant for baseline EMA path in seconds.
        Controls how slowly the baseline tracks environmental drift.
        Default: 30.0s (responds in ~90s)
        Recommended range: 20-60s

    p_ref : float
        Reference pressure in Pascals.
        Used for barometric altitude calculation.
        Default: 101325.0 Pa (standard sea level pressure)
        Note: Auto-calibrated from data if update_baseline=True in process()

    dt : Optional[float]
        Sampling interval in seconds.
        If None, will be auto-computed from timestamps.
        Default: None (auto-compute)
        Example: 0.04s for 25Hz sampling

    Examples
    --------
    Default configuration (recommended for 25Hz sampling):
    >>> config = BarometerConfig()

    Custom configuration for noisy sensor:
    >>> config = BarometerConfig(
    ...     median_window=9,   # More aggressive filtering
    ...     tau_fast=0.7,      # Slower response, more stable
    ...     tau_base=45.0      # Longer baseline tracking
    ... )

    Configuration for fast response (less noise rejection):
    >>> config = BarometerConfig(
    ...     median_window=3,   # Minimal filtering
    ...     tau_fast=0.3,      # Quick response
    ...     tau_base=20.0      # Faster baseline adaptation
    ... )
    """

    median_window: int = 5
    tau_fast: float = 0.5
    tau_base: float = 30.0
    p_ref: float = 101325.0
    dt: Optional[float] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate median_window is odd and positive
        if self.median_window <= 0:
            raise ValueError(f"median_window must be positive, got {self.median_window}")

        if self.median_window % 2 == 0:
            # Auto-convert to odd
            old_value = self.median_window
            self.median_window = self.median_window + 1
            import warnings
            warnings.warn(
                f"median_window must be odd (scipy requirement). "
                f"Converting {old_value} to {self.median_window}",
                UserWarning
            )

        # Validate time constants are positive
        if self.tau_fast <= 0:
            raise ValueError(f"tau_fast must be positive, got {self.tau_fast}")
        if self.tau_base <= 0:
            raise ValueError(f"tau_base must be positive, got {self.tau_base}")

        # Validate tau_base > tau_fast (baseline should be slower)
        if self.tau_base <= self.tau_fast:
            raise ValueError(
                f"tau_base ({self.tau_base}) must be greater than tau_fast ({self.tau_fast}). "
                f"Baseline should track slower than fast path."
            )

        # Validate reference pressure is reasonable
        if not (50000 <= self.p_ref <= 110000):
            import warnings
            warnings.warn(
                f"p_ref={self.p_ref} Pa is outside typical range (50-110 kPa). "
                f"This may indicate an error.",
                UserWarning
            )

        # Validate dt if provided
        if self.dt is not None:
            if self.dt <= 0:
                raise ValueError(f"dt must be positive, got {self.dt}")
            if self.dt > 1.0:
                import warnings
                warnings.warn(
                    f"dt={self.dt}s is very large. Typical values are 0.01-0.1s",
                    UserWarning
                )

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns
        -------
        dict
            Dictionary representation of configuration
        """
        return {
            'median_window': self.median_window,
            'tau_fast': self.tau_fast,
            'tau_base': self.tau_base,
            'p_ref': self.p_ref,
            'dt': self.dt,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'BarometerConfig':
        """
        Create configuration from dictionary.

        Parameters
        ----------
        config_dict : dict
            Dictionary with configuration parameters

        Returns
        -------
        BarometerConfig
            New configuration instance
        """
        return cls(**config_dict)

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"BarometerConfig("
            f"median_window={self.median_window}, "
            f"tau_fast={self.tau_fast}s, "
            f"tau_base={self.tau_base}s, "
            f"p_ref={self.p_ref:.0f}Pa, "
            f"dt={self.dt}s)"
        )


# Preset configurations for common use cases
class BarometerPresets:
    """
    Preset configurations for common barometer processing scenarios.

    Note: Presets are designed for typical sampling rates (1-25Hz for barometer).
    The dt parameter will be auto-computed from timestamps if not specified.
    """

    @staticmethod
    def default() -> BarometerConfig:
        """
        Default balanced configuration for barometer feature extraction.

        - Balanced noise reduction and response time
        - Good for extracting height change features for ML models
        - Recommended starting point for most applications
        """
        return BarometerConfig(
            median_window=5,
            tau_fast=0.5,
            tau_base=30.0
        )

    @staticmethod
    def fall_detection() -> BarometerConfig:
        """
        Alias for default() - kept for backward compatibility.

        Note: This module does NOT perform fall detection. It extracts
        barometer features (delta_h, h_fast, h_base) that can be used
        as inputs to an external ML model for fall detection.
        """
        return BarometerPresets.default()

    @staticmethod
    def fast_response() -> BarometerConfig:
        """
        Configuration for fastest response time.

        - Minimal filtering for quick detection
        - Higher noise, may have false positives
        - Use when speed is critical
        """
        return BarometerConfig(
            median_window=3,
            tau_fast=0.3,
            tau_base=20.0
        )

    @staticmethod
    def high_precision() -> BarometerConfig:
        """
        Configuration for maximum noise reduction.

        - Heavy filtering for clean signal
        - Slower response time
        - Use with very noisy sensors
        """
        return BarometerConfig(
            median_window=9,
            tau_fast=0.8,
            tau_base=45.0
        )

    @staticmethod
    def stair_detection() -> BarometerConfig:
        """
        Configuration optimized for detecting stair climbing.

        - Moderate filtering
        - Balanced response for step-wise changes
        """
        return BarometerConfig(
            median_window=7,
            tau_fast=0.6,
            tau_base=40.0
        )
