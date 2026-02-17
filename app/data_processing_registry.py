"""
Unified Inference Engine - Handles all model versions with automatic preprocessing.

This module provides a single interface for running fall detection inference
regardless of which model version is selected. It automatically applies the
correct preprocessing pipeline based on the model configuration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
import sys
from pathlib import Path

# Add analysis directory to path
_analysis_dir = Path(__file__).parent.parent / 'analysis'
if str(_analysis_dir) not in sys.path:
    sys.path.insert(0, str(_analysis_dir))

from .core.model_registry import (
    ModelType, 
    # ModelConfig, 
    get_model_type, get_model_config,
    # load_inference_class, 
    get_model_path
)


class PipelineSelector:
    """
    Unified inference engine that automatically selects the correct
    preprocessing pipeline based on model version.

    Usage:
        >>> engine = PipelineSelector('v3')
        >>> result = engine.predict(acc_df, pressure, pressure_time)
    """

    def __init__(self, model_version: str, model_path: Optional[str] = None):
        """
        Initialize the unified inference engine.

        Args:
            model_version: Model version string (v1, v2, v3, v4, v5)
            model_path: Optional custom model file path
        """
        self.model_type = get_model_type(model_version)
        self.config = get_model_config(self.model_type)
        self.model_path = model_path or get_model_path(self.model_type)

        # Initialize the correct inference class
        self._init_inference()

        # Initialize preprocessors based on model config
        self._init_preprocessors()

    def _init_inference(self):
        """Initialize the model-specific inference class."""
        # All models use the generic XGBoost loader
        # PipelineSelector handles all preprocessing and feature extraction internally
        self._init_generic_inference()

    def _init_generic_inference(self):
        """Initialize generic XGBoost inference for V1, V2, V4, V5."""
        import xgboost as xgb
        import joblib

        model_path_obj = Path(self.model_path)

        if model_path_obj.suffix == '.json':
            self.model = xgb.Booster()
            self.model.load_model(str(self.model_path))
            self.is_native_booster = True
        elif model_path_obj.suffix == '.pkl':
            self.model = joblib.load(self.model_path)
            self.is_native_booster = not hasattr(self.model, 'get_booster')
        else:
            raise ValueError(f"Unsupported model format: {model_path_obj.suffix}")

        self.inference = None  # Will use _predict_generic

    def _init_preprocessors(self):
        """Initialize preprocessing components based on model config."""
        self.acc_preprocessor = None
        self.baro_preprocessor = None

        # Initialize ACC preprocessor
        if self.config.acc_preprocessing == 'v2_paper':
            from app.data_input.accelerometer_processor.magnitude_based_acc_processor_paper import (
                PaperAccelerometerConfig, PaperAccelerometerProcessor
            )
            acc_config = PaperAccelerometerConfig(
                impact_threshold_g=4.0,
                crossing_threshold_g=1.0,
                sample_rate=50.0,
            )
            self.acc_preprocessor = PaperAccelerometerProcessor(acc_config)

        # Initialize BARO preprocessor
        if self.config.baro_preprocessing == 'v1_ema':
            from app.data_input.barometer_processor.barometer_config import BarometerConfig
            from app.data_input.barometer_processor.barometer_ema_filter import BarometerProcessor
            baro_config = BarometerConfig(
                median_window=5,
                tau_fast=0.5,
                tau_base=30.0,
            )
            self.baro_preprocessor = BarometerProcessor(baro_config)

        elif self.config.baro_preprocessing == 'v2_paper':
            from app.data_input.barometer_processor.barometer_slope_limit_paper import (
                PaperBarometerConfig, PaperBarometerProcessor
            )
            baro_config = PaperBarometerConfig(
                slope_limit=25.0,
                moving_avg_window=1.0,
                sample_rate=25.0,
            )
            self.baro_preprocessor = PaperBarometerProcessor(baro_config)

    def _compute_basic_stats(self, data: np.ndarray, prefix: str) -> Dict[str, float]:
        """Compute basic statistical features."""
        if len(data) == 0:
            return {
                f'{prefix}_min': 0.0,
                f'{prefix}_max': 0.0,
                f'{prefix}_mean': 0.0,
                f'{prefix}_var': 0.0,
            }
        return {
            f'{prefix}_min': float(np.min(data)),
            f'{prefix}_max': float(np.max(data)),
            f'{prefix}_mean': float(np.mean(data)),
            f'{prefix}_var': float(np.var(data)),
        }

    def _compute_slope(self, data: np.ndarray, sample_rate: float) -> float:
        """Compute linear slope of data."""
        if len(data) < 2:
            return 0.0
        x = np.arange(len(data)) / sample_rate
        slope, _ = np.polyfit(x, data, 1)
        return float(slope)

    def _extract_v1_acc_features(self, acc_x, acc_y, acc_z, acc_mag) -> Dict[str, float]:
        """Extract V1 style ACC features (statistical)."""
        features = {}
        features.update(self._compute_basic_stats(acc_x, 'acc_x'))
        features.update(self._compute_basic_stats(acc_y, 'acc_y'))
        features.update(self._compute_basic_stats(acc_z, 'acc_z'))
        features.update(self._compute_basic_stats(acc_mag, 'acc_mag'))
        return features

    def _extract_v2_acc_features(self, acc_data, acc_timestamps) -> Dict[str, float]:
        """Extract V2 style ACC features (paper magnitude + events)."""
        magnitude, events = self.acc_preprocessor.process(acc_data, acc_timestamps)
        return {
            'acc_mag_max': float(np.max(magnitude)),
            'acc_mag_mean': float(np.mean(magnitude)),
            'acc_mag_var': float(np.var(magnitude)),
            'impact_count': len(events),
            'max_impact_g': max((e.peak_magnitude for e in events), default=0.0),
            'has_high_impact': 1.0 if len(events) > 0 else 0.0,
        }

    def _extract_v1_baro_features(self, pressure, pressure_timestamps) -> Dict[str, float]:
        """Extract V1 style BARO features (dual-path EMA)."""
        if len(pressure) == 0:
            return {
                'delta_h_min': 0.0, 'delta_h_max': 0.0,
                'delta_h_mean': 0.0, 'delta_h_var': 0.0,
                'delta_h_range': 0.0, 'delta_h_slope': 0.0,
            }

        h_filtered, h_fast, h_base, delta_h = self.baro_preprocessor.process(
            pressure, pressure_timestamps
        )
        features = self._compute_basic_stats(delta_h, 'delta_h')
        features['delta_h_range'] = float(np.max(delta_h) - np.min(delta_h))
        features['delta_h_slope'] = self._compute_slope(delta_h, 25.0)
        return features

    def _extract_v2_baro_features(self, pressure, pressure_timestamps) -> Dict[str, float]:
        """Extract V2 style BARO features (paper slope-limit)."""
        if len(pressure) == 0:
            return {
                'pressure_shift': 0.0,
                'middle_slope': 0.0,
                'post_fall_slope': 0.0,
                'filtered_pressure_var': 0.0,
            }

        filtered = self.baro_preprocessor.process(pressure, pressure_timestamps)
        ref_idx = len(filtered) // 2
        paper_features = self.baro_preprocessor.extract_features(filtered, ref_idx)

        return {
            'pressure_shift': paper_features['pressure_shift'],
            'middle_slope': paper_features['middle_slope'],
            'post_fall_slope': paper_features['post_fall_slope'],
            'filtered_pressure_var': float(np.var(filtered)),
        }

    def _extract_raw_features(self, acc_x, acc_y, acc_z, acc_mag,
                               pressure, pressure_timestamps) -> Dict[str, float]:
        """Extract raw features (V5 style)."""
        features = {}
        features.update(self._compute_basic_stats(acc_x, 'raw_acc_x'))
        features.update(self._compute_basic_stats(acc_y, 'raw_acc_y'))
        features.update(self._compute_basic_stats(acc_z, 'raw_acc_z'))
        features.update(self._compute_basic_stats(acc_mag, 'raw_acc_mag'))

        if len(pressure) > 0:
            features.update(self._compute_basic_stats(pressure, 'raw_pressure'))
            features['raw_pressure_range'] = float(np.max(pressure) - np.min(pressure))
            features['raw_pressure_slope'] = self._compute_slope(pressure, 25.0)
        else:
            features.update(self._compute_basic_stats(np.array([]), 'raw_pressure'))
            features['raw_pressure_range'] = 0.0
            features['raw_pressure_slope'] = 0.0

        return features

    def extract_features(
        self,
        acc_df: pd.DataFrame,
        pressure: Optional[np.ndarray] = None,
        pressure_timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Extract features based on model configuration.

        Args:
            acc_df: DataFrame with Acc_X[g], Acc_Y[g], Acc_Z[g] columns
            pressure: Barometer pressure values
            pressure_timestamps: Barometer timestamps in ms

        Returns:
            Dict of feature name -> value
        """
        # Extract ACC arrays
        acc_x = acc_df['Acc_X[g]'].values
        acc_y = acc_df['Acc_Y[g]'].values
        acc_z = acc_df['Acc_Z[g]'].values
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

        features = {}

        # Extract ACC features based on config
        if self.config.acc_preprocessing == 'v1_features':
            features.update(self._extract_v1_acc_features(acc_x, acc_y, acc_z, acc_mag))
        elif self.config.acc_preprocessing == 'v2_paper':
            acc_data = np.array([acc_x, acc_y, acc_z])
            acc_timestamps = acc_df['Device_Timestamp_[ms]'].values
            features.update(self._extract_v2_acc_features(acc_data, acc_timestamps))
        elif self.config.acc_preprocessing == 'raw':
            features.update(self._compute_basic_stats(acc_x, 'raw_acc_x'))
            features.update(self._compute_basic_stats(acc_y, 'raw_acc_y'))
            features.update(self._compute_basic_stats(acc_z, 'raw_acc_z'))
            features.update(self._compute_basic_stats(acc_mag, 'raw_acc_mag'))

        # Extract BARO features if model uses barometer
        if pressure is None:
            pressure = np.array([])
        if pressure_timestamps is None:
            pressure_timestamps = np.array([])

        if self.config.baro_preprocessing == 'v1_ema':
            features.update(self._extract_v1_baro_features(pressure, pressure_timestamps))
        elif self.config.baro_preprocessing == 'v2_paper':
            features.update(self._extract_v2_baro_features(pressure, pressure_timestamps))
        elif self.config.baro_preprocessing == 'raw':
            if len(pressure) > 0:
                features.update(self._compute_basic_stats(pressure, 'raw_pressure'))
                features['raw_pressure_range'] = float(np.max(pressure) - np.min(pressure))
                features['raw_pressure_slope'] = self._compute_slope(pressure, 25.0)
            else:
                features.update(self._compute_basic_stats(np.array([]), 'raw_pressure'))
                features['raw_pressure_range'] = 0.0
                features['raw_pressure_slope'] = 0.0

        return features

    def _get_feature_names(self) -> list:
        """Get ordered feature names for current model."""
        # Define feature names directly here to avoid import conflicts
        feature_definitions = {
            ModelType.V0: {
                "acc_features": [
                    "acc_x_min", "acc_x_max", "acc_x_mean", "acc_x_var",
                    "acc_y_min", "acc_y_max", "acc_y_mean", "acc_y_var",
                    "acc_z_min", "acc_z_max", "acc_z_mean", "acc_z_var",
                    "acc_mag_min", "acc_mag_max", "acc_mag_mean", "acc_mag_var",
                ],
                "baro_features": [],  # V0 has no barometer features
            },
            ModelType.V1: {
                "acc_features": [
                    "acc_x_min", "acc_x_max", "acc_x_mean", "acc_x_var",
                    "acc_y_min", "acc_y_max", "acc_y_mean", "acc_y_var",
                    "acc_z_min", "acc_z_max", "acc_z_mean", "acc_z_var",
                    "acc_mag_min", "acc_mag_max", "acc_mag_mean", "acc_mag_var",
                ],
                "baro_features": [
                    "delta_h_min", "delta_h_max", "delta_h_mean", "delta_h_var",
                    "delta_h_range", "delta_h_slope",
                ],
            },
            ModelType.V2: {
                "acc_features": [
                    "acc_mag_max", "acc_mag_mean", "acc_mag_var",
                    "impact_count", "max_impact_g", "has_high_impact",
                ],
                "baro_features": [
                    "pressure_shift", "middle_slope", "post_fall_slope",
                    "filtered_pressure_var",
                ],
            },
            ModelType.V3: {
                "acc_features": [
                    "acc_x_min", "acc_x_max", "acc_x_mean", "acc_x_var",
                    "acc_y_min", "acc_y_max", "acc_y_mean", "acc_y_var",
                    "acc_z_min", "acc_z_max", "acc_z_mean", "acc_z_var",
                    "acc_mag_min", "acc_mag_max", "acc_mag_mean", "acc_mag_var",
                ],
                "baro_features": [
                    "pressure_shift", "middle_slope", "post_fall_slope",
                    "filtered_pressure_var",
                ],
            },
            ModelType.V4: {
                "acc_features": [
                    "acc_mag_max", "acc_mag_mean", "acc_mag_var",
                    "impact_count", "max_impact_g", "has_high_impact",
                ],
                "baro_features": [
                    "delta_h_min", "delta_h_max", "delta_h_mean", "delta_h_var",
                    "delta_h_range", "delta_h_slope",
                ],
            },
            ModelType.V5: {
                "acc_features": [
                    "raw_acc_x_min", "raw_acc_x_max", "raw_acc_x_mean", "raw_acc_x_var",
                    "raw_acc_y_min", "raw_acc_y_max", "raw_acc_y_mean", "raw_acc_y_var",
                    "raw_acc_z_min", "raw_acc_z_max", "raw_acc_z_mean", "raw_acc_z_var",
                    "raw_acc_mag_min", "raw_acc_mag_max", "raw_acc_mag_mean", "raw_acc_mag_var",
                ],
                "baro_features": [
                    "raw_pressure_min", "raw_pressure_max", "raw_pressure_mean", "raw_pressure_var",
                    "raw_pressure_range", "raw_pressure_slope",
                ],
            },
            ModelType.V1_TUNED: {
                "acc_features": [
                    "acc_x_min", "acc_x_max", "acc_x_mean", "acc_x_var",
                    "acc_y_min", "acc_y_max", "acc_y_mean", "acc_y_var",
                    "acc_z_min", "acc_z_max", "acc_z_mean", "acc_z_var",
                    "acc_mag_min", "acc_mag_max", "acc_mag_mean", "acc_mag_var",
                ],
                "baro_features": [
                    "delta_h_min", "delta_h_max", "delta_h_mean", "delta_h_var",
                    "delta_h_range", "delta_h_slope",
                ],
            },
            ModelType.V3_TUNED: {
                "acc_features": [
                    "acc_x_min", "acc_x_max", "acc_x_mean", "acc_x_var",
                    "acc_y_min", "acc_y_max", "acc_y_mean", "acc_y_var",
                    "acc_z_min", "acc_z_max", "acc_z_mean", "acc_z_var",
                    "acc_mag_min", "acc_mag_max", "acc_mag_mean", "acc_mag_var",
                ],
                "baro_features": [
                    "pressure_shift", "middle_slope", "post_fall_slope",
                    "filtered_pressure_var",
                ],
            },
        }

        if self.model_type in feature_definitions:
            fd = feature_definitions[self.model_type]
            return fd['acc_features'] + fd['baro_features']

        raise ValueError(f"Unknown model type: {self.model_type}")

    def predict(
        self,
        acc_df: pd.DataFrame,
        pressure: Optional[np.ndarray] = None,
        pressure_timestamps: Optional[np.ndarray] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run fall detection prediction.

        Args:
            acc_df: DataFrame with accelerometer data
            pressure: Barometer pressure values
            pressure_timestamps: Barometer timestamps

        Returns:
            dict with is_fall, confidence, features
        """
        # All models use generic inference - unified preprocessing and feature extraction
        return self._predict_generic(acc_df, pressure, pressure_timestamps, threshold)

    def _predict_generic(
        self,
        acc_df: pd.DataFrame,
        pressure: Optional[np.ndarray],
        pressure_timestamps: Optional[np.ndarray],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generic prediction for models without dedicated inference class."""
        import xgboost as xgb

        # Extract features
        features_dict = self.extract_features(acc_df, pressure, pressure_timestamps)

        # Get feature names and create input array
        feature_names = self._get_feature_names()
        X = np.array([[features_dict.get(name, 0.0) for name in feature_names]])

        # Get threshold
        if threshold is None:
            threshold = self.config.threshold

        # Run prediction
        if self.is_native_booster:
            dmatrix = xgb.DMatrix(X, feature_names=feature_names)
            proba = self.model.predict(dmatrix)[0]
            confidence = float(proba)
            prediction = 1 if confidence > threshold else 0
        else:
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            confidence = float(probabilities[1])

        return {
            'is_fall': bool(prediction),
            'confidence': confidence,
            'features': features_dict
        }

    def uses_barometer(self) -> bool:
        """Check if current model uses barometer data."""
        return self.config.uses_barometer

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model."""
        return {
            'version': self.model_type.value,
            'name': self.config.name,
            'description': self.config.description,
            'uses_barometer': self.config.uses_barometer,
            'acc_preprocessing': self.config.acc_preprocessing,
            'baro_preprocessing': self.config.baro_preprocessing,
            'num_features': self.config.num_features,
            'acc_features': self.config.acc_features,
            'baro_features': self.config.baro_features,
        }
