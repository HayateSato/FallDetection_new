"""
Unified barometer preprocessing interface with version switching.

This module provides a clean interface to switch between two preprocessing approaches:

Version 1 - Dual-path EMA (Original):
    - Median filter + dual-path EMA filtering
    - Outputs: h_filtered, h_fast, h_base, delta_h
    - Good for continuous monitoring with slow drift compensation

Version 2 - Slope-limit (Paper):
    - Slope-limit filter + moving average
    - Outputs: filtered_pressure, features (pressure_shift, middle_slope, post_fall_slope)
    - Designed for event-triggered detection with robust spike removal

Usage:
    >>> orchestrator = BarometerPreprocessingOrchestrator(version='v1_ema')
    >>> result = orchestrator.process(pressure, timestamps)

    # Switch versions
    >>> orchestrator.set_version('v2_paper')
    >>> result = orchestrator.process(pressure, timestamps)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

from barometer_config import BarometerConfig
from app.data_input.barometer_processor.barometer_ema_filter import BarometerProcessor, StreamingBarometerProcessor
from app.data_input.barometer_processor.barometer_slope_limit_paper import (
    PaperBarometerConfig,
    PaperBarometerProcessor,
    StreamingPaperBarometerProcessor
)


class PreprocessingVersion(Enum):
    """Available preprocessing versions."""
    V1_EMA = "v1_ema"          # Original dual-path EMA approach
    V2_PAPER = "v2_paper"      # Paper's slope-limit approach


@dataclass
class PreprocessingResult:
    """
    Unified result container for both preprocessing versions.

    Attributes
    ----------
    version : PreprocessingVersion
        Which preprocessing version produced this result

    filtered_signal : np.ndarray
        Primary filtered output signal

    features : Dict[str, Any]
        Version-specific features:
        - V1_EMA: {'h_fast', 'h_base', 'delta_h'}
        - V2_PAPER: {'pressure_shift', 'middle_slope', 'post_fall_slope'} (if extracted)

    raw_outputs : Dict[str, np.ndarray]
        All raw output arrays for detailed analysis
    """
    version: PreprocessingVersion
    filtered_signal: np.ndarray
    features: Dict[str, Any]
    raw_outputs: Dict[str, np.ndarray]

    def get_primary_feature(self) -> np.ndarray:
        """
        Get the primary feature array for fall detection.

        Returns
        -------
        np.ndarray
            - V1_EMA: delta_h (height change relative to baseline)
            - V2_PAPER: filtered_signal (filtered pressure)
        """
        if self.version == PreprocessingVersion.V1_EMA:
            return self.raw_outputs.get('delta_h', self.filtered_signal)
        else:
            return self.filtered_signal


class BaseBarometerPreprocessor(ABC):
    """Abstract base class for barometer preprocessors."""

    @abstractmethod
    def process(
        self,
        pressure: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> PreprocessingResult:
        """Process pressure data and return unified result."""
        pass

    @abstractmethod
    def get_version(self) -> PreprocessingVersion:
        """Return the preprocessing version."""
        pass

    @abstractmethod
    def get_config(self) -> Any:
        """Return the current configuration."""
        pass


class V1EMAPreprocessor(BaseBarometerPreprocessor):
    """
    Wrapper for the original dual-path EMA preprocessing approach.

    This approach uses:
    1. Pressure to altitude conversion
    2. Median filtering for spike removal
    3. Dual-path EMA (fast + baseline)
    4. Delta_h = h_fast - h_base for height change detection
    """

    def __init__(self, config: Optional[BarometerConfig] = None):
        """
        Initialize V1 EMA preprocessor.

        Parameters
        ----------
        config : BarometerConfig, optional
            Configuration for dual-path EMA. If None, uses defaults.
        """
        self.config = config or BarometerConfig()
        self.processor = BarometerProcessor(self.config)

    def process(
        self,
        pressure: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> PreprocessingResult:
        """
        Process pressure data using dual-path EMA.

        Parameters
        ----------
        pressure : np.ndarray
            Pressure values in Pascals
        timestamps : np.ndarray, optional
            Timestamps for automatic dt calculation

        Returns
        -------
        PreprocessingResult
            Unified result with EMA-specific outputs
        """
        h_filtered, h_fast, h_base, delta_h = self.processor.process(
            pressure, timestamps
        )

        return PreprocessingResult(
            version=PreprocessingVersion.V1_EMA,
            filtered_signal=h_filtered,
            features={
                'delta_h_mean': float(np.mean(delta_h)),
                'delta_h_min': float(np.min(delta_h)),
                'delta_h_max': float(np.max(delta_h)),
                'delta_h_std': float(np.std(delta_h)),
            },
            raw_outputs={
                'h_filtered': h_filtered,
                'h_fast': h_fast,
                'h_base': h_base,
                'delta_h': delta_h
            }
        )

    def get_version(self) -> PreprocessingVersion:
        return PreprocessingVersion.V1_EMA

    def get_config(self) -> BarometerConfig:
        return self.config


class V2PaperPreprocessor(BaseBarometerPreprocessor):
    """
    Wrapper for the paper's slope-limit preprocessing approach.

    This approach uses:
    1. Slope-limit filter (removes sudden spikes)
    2. Moving average filter (1-second window)
    3. Feature extraction (pressure_shift, middle_slope, post_fall_slope)
    """

    def __init__(self, config: Optional[PaperBarometerConfig] = None):
        """
        Initialize V2 paper preprocessor.

        Parameters
        ----------
        config : PaperBarometerConfig, optional
            Configuration for slope-limit approach. If None, uses paper defaults.
        """
        self.config = config or PaperBarometerConfig()
        self.processor = PaperBarometerProcessor(self.config)

    def process(
        self,
        pressure: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        extract_features: bool = False,
        reference_idx: Optional[int] = None
    ) -> PreprocessingResult:
        """
        Process pressure data using slope-limit filter.

        Parameters
        ----------
        pressure : np.ndarray
            Pressure values (raw units or Pascals)
        timestamps : np.ndarray, optional
            Timestamps for sample rate estimation
        extract_features : bool
            Whether to extract paper's 3 features (requires 6-second window)
        reference_idx : int, optional
            Reference point for feature extraction (impact time)

        Returns
        -------
        PreprocessingResult
            Unified result with paper-specific outputs
        """
        filtered = self.processor.process(pressure, timestamps)

        features = {}
        if extract_features:
            if reference_idx is None:
                reference_idx = len(filtered) // 2
            features = self.processor.extract_features(filtered, reference_idx)

        return PreprocessingResult(
            version=PreprocessingVersion.V2_PAPER,
            filtered_signal=filtered,
            features=features,
            raw_outputs={
                'filtered_pressure': filtered,
                'slope_limited': self.processor.slope_limit_filter(pressure)
            }
        )

    def get_version(self) -> PreprocessingVersion:
        return PreprocessingVersion.V2_PAPER

    def get_config(self) -> PaperBarometerConfig:
        return self.config


class BarometerPreprocessingOrchestrator:
    """
    Orchestrator for switching between barometer preprocessing versions.

    This class provides a unified interface to easily switch between different
    preprocessing approaches for comparison and evaluation.

    Examples
    --------
    Basic usage with version switching:

    >>> # Start with V1 (EMA approach)
    >>> orchestrator = BarometerPreprocessingOrchestrator(version='v1_ema')
    >>> result_v1 = orchestrator.process(pressure, timestamps)
    >>> print(f"V1 delta_h mean: {result_v1.features['delta_h_mean']}")

    >>> # Switch to V2 (paper approach)
    >>> orchestrator.set_version('v2_paper')
    >>> result_v2 = orchestrator.process(pressure, timestamps)
    >>> print(f"V2 filtered signal shape: {result_v2.filtered_signal.shape}")

    Processing both versions for comparison:

    >>> orchestrator = BarometerPreprocessingOrchestrator()
    >>> comparison = orchestrator.process_both(pressure, timestamps)
    >>> print(f"V1 output: {comparison['v1_ema'].features}")
    >>> print(f"V2 output: {comparison['v2_paper'].features}")
    """

    VERSION_MAP = {
        'v1_ema': PreprocessingVersion.V1_EMA,
        'v1': PreprocessingVersion.V1_EMA,
        'ema': PreprocessingVersion.V1_EMA,
        'v2_paper': PreprocessingVersion.V2_PAPER,
        'v2': PreprocessingVersion.V2_PAPER,
        'paper': PreprocessingVersion.V2_PAPER,
        'slope_limit': PreprocessingVersion.V2_PAPER,
    }

    def __init__(
        self,
        version: Union[str, PreprocessingVersion] = 'v1_ema',
        v1_config: Optional[BarometerConfig] = None,
        v2_config: Optional[PaperBarometerConfig] = None
    ):
        """
        Initialize the orchestrator.

        Parameters
        ----------
        version : str or PreprocessingVersion
            Initial preprocessing version to use.
            Accepts: 'v1_ema', 'v1', 'ema', 'v2_paper', 'v2', 'paper', 'slope_limit'

        v1_config : BarometerConfig, optional
            Configuration for V1 EMA preprocessor

        v2_config : PaperBarometerConfig, optional
            Configuration for V2 paper preprocessor
        """
        # Initialize both preprocessors
        self._v1_preprocessor = V1EMAPreprocessor(v1_config)
        self._v2_preprocessor = V2PaperPreprocessor(v2_config)

        # Set initial version
        self._current_version = self._resolve_version(version)
        self._current_preprocessor = self._get_preprocessor(self._current_version)

    def _resolve_version(self, version: Union[str, PreprocessingVersion]) -> PreprocessingVersion:
        """Convert string version to enum."""
        if isinstance(version, PreprocessingVersion):
            return version

        version_lower = version.lower().strip()
        if version_lower not in self.VERSION_MAP:
            valid = list(self.VERSION_MAP.keys())
            raise ValueError(f"Unknown version '{version}'. Valid options: {valid}")

        return self.VERSION_MAP[version_lower]

    def _get_preprocessor(self, version: PreprocessingVersion) -> BaseBarometerPreprocessor:
        """Get preprocessor for given version."""
        if version == PreprocessingVersion.V1_EMA:
            return self._v1_preprocessor
        else:
            return self._v2_preprocessor

    def set_version(self, version: Union[str, PreprocessingVersion]) -> None:
        """
        Switch to a different preprocessing version.

        Parameters
        ----------
        version : str or PreprocessingVersion
            Version to switch to
        """
        self._current_version = self._resolve_version(version)
        self._current_preprocessor = self._get_preprocessor(self._current_version)

    def get_version(self) -> PreprocessingVersion:
        """Get current preprocessing version."""
        return self._current_version

    def get_version_name(self) -> str:
        """Get current version as string."""
        return self._current_version.value

    def process(
        self,
        pressure: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        **kwargs
    ) -> PreprocessingResult:
        """
        Process pressure data using the current version.

        Parameters
        ----------
        pressure : np.ndarray
            Pressure values
        timestamps : np.ndarray, optional
            Timestamps for the pressure samples
        **kwargs
            Version-specific parameters:
            - V2: extract_features (bool), reference_idx (int)

        Returns
        -------
        PreprocessingResult
            Unified result from current preprocessor
        """
        if self._current_version == PreprocessingVersion.V2_PAPER:
            return self._v2_preprocessor.process(pressure, timestamps, **kwargs)
        else:
            return self._v1_preprocessor.process(pressure, timestamps)

    def process_both(
        self,
        pressure: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        v2_kwargs: Optional[Dict] = None
    ) -> Dict[str, PreprocessingResult]:
        """
        Process pressure data using both versions for comparison.

        Parameters
        ----------
        pressure : np.ndarray
            Pressure values
        timestamps : np.ndarray, optional
            Timestamps for the pressure samples
        v2_kwargs : dict, optional
            Additional kwargs for V2 processor (extract_features, reference_idx)

        Returns
        -------
        dict
            Results from both versions:
            - 'v1_ema': PreprocessingResult from V1
            - 'v2_paper': PreprocessingResult from V2
        """
        v2_kwargs = v2_kwargs or {}

        result_v1 = self._v1_preprocessor.process(pressure, timestamps)
        result_v2 = self._v2_preprocessor.process(pressure, timestamps, **v2_kwargs)

        return {
            'v1_ema': result_v1,
            'v2_paper': result_v2
        }

    def get_configs(self) -> Dict[str, Any]:
        """
        Get configurations for both versions.

        Returns
        -------
        dict
            Configurations for both preprocessors
        """
        return {
            'v1_ema': self._v1_preprocessor.get_config(),
            'v2_paper': self._v2_preprocessor.get_config()
        }

    def update_config(
        self,
        version: Union[str, PreprocessingVersion],
        config: Union[BarometerConfig, PaperBarometerConfig]
    ) -> None:
        """
        Update configuration for a specific version.

        Parameters
        ----------
        version : str or PreprocessingVersion
            Version to update
        config : BarometerConfig or PaperBarometerConfig
            New configuration
        """
        resolved = self._resolve_version(version)

        if resolved == PreprocessingVersion.V1_EMA:
            if not isinstance(config, BarometerConfig):
                raise TypeError("V1 requires BarometerConfig")
            self._v1_preprocessor = V1EMAPreprocessor(config)
        else:
            if not isinstance(config, PaperBarometerConfig):
                raise TypeError("V2 requires PaperBarometerConfig")
            self._v2_preprocessor = V2PaperPreprocessor(config)

        # Update current preprocessor reference if needed
        self._current_preprocessor = self._get_preprocessor(self._current_version)


class StreamingBarometerOrchestrator:
    """
    Streaming version of the orchestrator for real-time processing.

    Processes samples one at a time while maintaining internal state.
    """

    def __init__(
        self,
        version: Union[str, PreprocessingVersion] = 'v1_ema',
        v1_config: Optional[BarometerConfig] = None,
        v2_config: Optional[PaperBarometerConfig] = None
    ):
        """Initialize streaming orchestrator."""
        # Ensure dt is set for streaming mode
        if v1_config is None:
            v1_config = BarometerConfig(dt=0.04)  # 25Hz default
        elif v1_config.dt is None:
            v1_config = BarometerConfig(
                median_window=v1_config.median_window,
                tau_fast=v1_config.tau_fast,
                tau_base=v1_config.tau_base,
                p_ref=v1_config.p_ref,
                dt=0.04
            )

        self._v1_processor = StreamingBarometerProcessor(v1_config)
        self._v2_processor = StreamingPaperBarometerProcessor(
            v2_config or PaperBarometerConfig()
        )

        self._current_version = BarometerPreprocessingOrchestrator._resolve_version(
            BarometerPreprocessingOrchestrator, version
        )

    def set_version(self, version: Union[str, PreprocessingVersion]) -> None:
        """Switch preprocessing version."""
        self._current_version = BarometerPreprocessingOrchestrator._resolve_version(
            BarometerPreprocessingOrchestrator, version
        )

    def process_sample(self, pressure: float) -> Tuple[float, Dict[str, float]]:
        """
        Process a single pressure sample.

        Parameters
        ----------
        pressure : float
            Single pressure value

        Returns
        -------
        Tuple[float, Dict[str, float]]
            - Primary output value
            - Additional outputs as dict
        """
        if self._current_version == PreprocessingVersion.V1_EMA:
            h_fast, h_base, delta_h = self._v1_processor.process_sample(pressure)
            return delta_h, {'h_fast': h_fast, 'h_base': h_base, 'delta_h': delta_h}
        else:
            filtered = self._v2_processor.process_sample(pressure)
            return filtered, {'filtered_pressure': filtered}

    def reset(self) -> None:
        """Reset both processors."""
        self._v1_processor.reset()
        self._v2_processor.reset()

    def is_ready(self) -> bool:
        """Check if current processor is ready."""
        if self._current_version == PreprocessingVersion.V1_EMA:
            return self._v1_processor.is_ready()
        else:
            return self._v2_processor.is_ready()
