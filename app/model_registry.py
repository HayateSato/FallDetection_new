"""
Model Registry - Maps model versions to their inference classes and configurations.

This module provides a unified interface for loading different model versions
and their corresponding preprocessing pipelines.

Supported Models:
- v0: ACC only (statistical) - 16 features (no barometer, baseline for comparison)
- v1: ACC (statistical) + BARO (dual-path EMA) - 22 features
- v2: ACC (paper magnitude) + BARO (paper slope-limit) - 10 features
- v3: ACC (statistical) + BARO (paper slope-limit) [BEST] - 20 features
- v4: ACC (paper magnitude) + BARO (dual-path EMA) - 12 features
- v5: Raw features (minimal preprocessing) - 22 features
- v1_tuned: V1 with tuned hyperparameters (optimized for recall) - 22 features
- v3_tuned: V3 with tuned hyperparameters (optimized for recall) - 20 features
"""

from dataclasses import dataclass
from typing import Dict, Optional, Type, Callable, Any
from enum import Enum
from pathlib import Path
import os


class ModelType(Enum):
    """Available model types."""
    V0 = "v0"
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    V4 = "v4"
    V5 = "v5"
    V1_TUNED = "v1_tuned"
    V3_TUNED = "v3_tuned"


@dataclass
class ModelConfig:
    """Configuration for a model version."""
    name: str
    description: str
    model_path: str
    inference_class: str  # Module path to inference class
    uses_barometer: bool
    acc_preprocessing: str  # v1_features, v2_paper, or raw
    baro_preprocessing: str  # v1_ema, v2_paper, raw, or none
    num_features: int
    acc_features: int
    baro_features: int
    threshold: float = 0.5


# Model configurations
MODEL_CONFIGS: Dict[ModelType, ModelConfig] = {
    ModelType.V0: ModelConfig(
        name="V0",
        description="ACC only: Statistical features (no barometer, baseline)",
        model_path="model/model_v0/model_v0_xgboost.pkl",
        inference_class="app.unified_inference.UnifiedInference",
        uses_barometer=False,
        acc_preprocessing="v1_features",
        baro_preprocessing="none",
        num_features=16,
        acc_features=16,
        baro_features=0,
    ),
    ModelType.V1: ModelConfig(
        name="V1",
        description="ACC: Statistical features, BARO: Dual-path EMA",
        model_path="model/model_v1/model_v1_xgboost.pkl",
        inference_class="app.models.v1_inference.ModelV1Inference",
        uses_barometer=True,
        acc_preprocessing="v1_features",
        baro_preprocessing="v1_ema",
        num_features=22,
        acc_features=16,
        baro_features=6,
    ),
    ModelType.V2: ModelConfig(
        name="V2",
        description="ACC: Paper magnitude + events, BARO: Paper slope-limit",
        model_path="model/model_v2/model_v2_xgboost.pkl",
        inference_class="app.models.v2_inference.ModelV2Inference",
        uses_barometer=True,
        acc_preprocessing="v2_paper",
        baro_preprocessing="v2_paper",
        num_features=10,
        acc_features=6,
        baro_features=4,
    ),
    ModelType.V3: ModelConfig(
        name="V3",
        description="ACC: Statistical features, BARO: Paper slope-limit (BEST)",
        model_path="model/model_v3/model_v3_xgboost.pkl",
        inference_class="app.model_v3.inference.ModelV3Inference",
        uses_barometer=True,
        acc_preprocessing="v1_features",
        baro_preprocessing="v2_paper",
        num_features=20,
        acc_features=16,
        baro_features=4,
    ),
    ModelType.V4: ModelConfig(
        name="V4",
        description="ACC: Paper magnitude, BARO: Dual-path EMA",
        model_path="model/model_v4/model_v4_xgboost.pkl",
        inference_class="app.models.v4_inference.ModelV4Inference",
        uses_barometer=True,
        acc_preprocessing="v2_paper",
        baro_preprocessing="v1_ema",
        num_features=12,
        acc_features=6,
        baro_features=6,
    ),
    ModelType.V5: ModelConfig(
        name="V5",
        description="Raw features with minimal preprocessing",
        model_path="model/model_v5/model_v5_xgboost.pkl",
        inference_class="app.unified_inference.UnifiedInference",
        uses_barometer=True,
        acc_preprocessing="raw",
        baro_preprocessing="raw",
        num_features=22,
        acc_features=16,
        baro_features=6,
    ),
    ModelType.V1_TUNED: ModelConfig(
        name="V1_TUNED",
        description="V1 with tuned hyperparameters (optimized for recall)",
        model_path="model/model_v1_tuned/model_v1_tuned.pkl",
        inference_class="app.unified_inference.UnifiedInference",
        uses_barometer=True,
        acc_preprocessing="v1_features",
        baro_preprocessing="v1_ema",
        num_features=22,
        acc_features=16,
        baro_features=6,
    ),
    ModelType.V3_TUNED: ModelConfig(
        name="V3_TUNED",
        description="V3 with tuned hyperparameters (optimized for recall)",
        model_path="model/model_v3_tuned/model_v3_tuned.pkl",
        inference_class="app.unified_inference.UnifiedInference",
        uses_barometer=True,
        acc_preprocessing="v1_features",
        baro_preprocessing="v2_paper",
        num_features=20,
        acc_features=16,
        baro_features=4,
    ),
}


def get_model_type(version_string: str) -> ModelType:
    """
    Convert version string to ModelType enum.

    Args:
        version_string: Version string (e.g., 'v3', 'V3', 'model_c')

    Returns:
        ModelType enum value

    Raises:
        ValueError: If version string is not recognized
    """
    version_lower = version_string.lower().strip()

    # Handle various naming conventions
    version_map = {
        'v0': ModelType.V0,
        'v1': ModelType.V1,
        'v2': ModelType.V2,
        'v3': ModelType.V3,
        'v4': ModelType.V4,
        'v5': ModelType.V5,
        'v1_tuned': ModelType.V1_TUNED,
        'v3_tuned': ModelType.V3_TUNED,
    }

    if version_lower in version_map:
        return version_map[version_lower]

    raise ValueError(
        f"Unknown model version: '{version_string}'. "
        f"Valid options: {list(version_map.keys())}"
    )


def get_model_config(model_type: ModelType) -> ModelConfig:
    """Get configuration for a model type."""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"No configuration found for {model_type}")
    return MODEL_CONFIGS[model_type]


def load_inference_class(model_type: ModelType):
    """
    Dynamically load the inference class for a model type.

    Args:
        model_type: The model type to load

    Returns:
        The inference class (not instantiated)
    """
    config = get_model_config(model_type)

    # Parse module path and class name
    module_path, class_name = config.inference_class.rsplit('.', 1)

    # Dynamic import
    import importlib
    module = importlib.import_module(module_path)
    inference_class = getattr(module, class_name)

    return inference_class


def get_model_path(model_type: ModelType, custom_path: Optional[str] = None) -> str:
    """
    Get the model file path.

    Args:
        model_type: The model type
        custom_path: Optional custom path override

    Returns:
        Path to the model file
    """
    if custom_path:
        return custom_path

    config = get_model_config(model_type)
    return config.model_path


def list_available_models() -> Dict[str, str]:
    """List all available models with their descriptions."""
    return {
        model_type.value: config.description
        for model_type, config in MODEL_CONFIGS.items()
    }


def get_influxdb_query_fields(model_type: ModelType, barometer_field: str = "bmp_pressure") -> list:
    """
    Get the InfluxDB fields needed for a model.

    Args:
        model_type: The model type
        barometer_field: Field name for barometer data

    Returns:
        List of field names to query
    """
    config = get_model_config(model_type)

    # Always need accelerometer
    fields = ["bosch_acc_x", "bosch_acc_y", "bosch_acc_z"]

    # Add barometer if model uses it
    if config.uses_barometer:
        fields.append(barometer_field)

    return fields


class ModelRegistry:
    """
    Registry for managing model loading and inference.

    This class provides a high-level interface for working with different
    model versions in the fall detection pipeline.
    """

    def __init__(self, default_model: str = "v3"):
        """
        Initialize the model registry.

        Args:
            default_model: Default model version to use
        """
        self._current_model_type = get_model_type(default_model)
        self._model_instance = None
        self._custom_model_path = None

    @property
    def current_model(self) -> ModelType:
        """Get the current model type."""
        return self._current_model_type

    @property
    def current_config(self) -> ModelConfig:
        """Get the current model configuration."""
        return get_model_config(self._current_model_type)

    def set_model(self, version: str, custom_path: Optional[str] = None):
        """
        Set the current model version.

        Args:
            version: Model version string
            custom_path: Optional custom model file path
        """
        self._current_model_type = get_model_type(version)
        self._custom_model_path = custom_path
        self._model_instance = None  # Clear cached instance

    def get_inference_instance(self):
        """
        Get or create the inference instance for the current model.

        Returns:
            Instantiated inference class
        """
        if self._model_instance is None:
            inference_class = load_inference_class(self._current_model_type)
            model_path = get_model_path(self._current_model_type, self._custom_model_path)
            self._model_instance = inference_class(model_path)

        return self._model_instance

    def get_query_fields(self, barometer_field: str = "bmp_pressure") -> list:
        """Get InfluxDB fields needed for current model."""
        return get_influxdb_query_fields(self._current_model_type, barometer_field)

    def uses_barometer(self) -> bool:
        """Check if current model uses barometer data."""
        return self.current_config.uses_barometer

    def get_preprocessing_config(self) -> dict:
        """Get preprocessing configuration for current model."""
        config = self.current_config
        return {
            'acc_preprocessing': config.acc_preprocessing,
            'baro_preprocessing': config.baro_preprocessing,
        }
