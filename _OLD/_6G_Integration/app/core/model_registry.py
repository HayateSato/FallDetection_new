from typing import Dict, Optional
from app.core.model_config import (
    ModelName,
    ModelConfig,
    MODEL_CONFIGS,
)

def get_model_name(version_string: str) -> ModelName:
    """
    Convert version string to ModelName enum.

    Args:
        version_string: Version string (e.g., 'v3', 'V3')

    Returns:
        ModelName enum value

    Raises:
        ValueError: If version string is not recognized
    """
    try:
        return ModelName(version_string.lower().strip())
    except ValueError:
        valid = [m.value for m in ModelName]
        raise ValueError(
            f"Unknown model version: '{version_string}'. "
            f"Valid options: {valid}"
        )

def get_model_config(model_type: ModelName) -> ModelConfig:
    """Get configuration for a model type."""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"No configuration found for {model_type}")
    return MODEL_CONFIGS[model_type]


def load_inference_class(model_type: ModelName):
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


def get_model_path(model_type: ModelName, custom_path: Optional[str] = None) -> str:
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


def get_influxdb_query_fields(model_type: ModelName, barometer_field: str = "bmp_pressure") -> list:
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
        self._current_model_type = get_model_name(default_model)
        self._model_instance = None
        self._custom_model_path = None

    @property
    def current_model(self) -> ModelName:
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
        self._current_model_type = get_model_name(version)
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
