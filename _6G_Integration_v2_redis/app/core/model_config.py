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
from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum

from app.core.feature_config import _STAT_ACC, _PAPER_ACC, _RAW_ACC, _EMA_BARO, _PAPER_BARO, _RAW_BARO

class ModelName(Enum):
    """Available model types."""
    V0 = "v0"
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    V4 = "v4"
    V5 = "v5"
    V0_LSB_INT = "v0_lsb_int"
    V1_TUNED = "v1_tuned"
    V3_TUNED = "v3_tuned"
    V5_LSB = "v5_lsb"


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
    acc_feature_names: List[str] = field(default_factory=list)
    baro_feature_names: List[str] = field(default_factory=list)
    threshold: float = 0.5
    acc_in_lsb: bool = False  # If True, model expects raw LSB integers (no g conversion)

# Model configurations
MODEL_CONFIGS: Dict[ModelName, ModelConfig] = {
    ModelName.V0: ModelConfig(
        name="V0",
        description="ACC only: Statistical features (no barometer, baseline)",
        model_path="model/model_v0/model_v0_xgboost.pkl",
        inference_class="app.data_processing_registry.PipelineSelector",
        uses_barometer=False,
        acc_preprocessing="v1_features",
        baro_preprocessing="none",
        num_features=16,
        acc_features=16,
        baro_features=0,
        acc_feature_names=_STAT_ACC,
        baro_feature_names=[],
    ),
    ModelName.V0_LSB_INT: ModelConfig(
        name="V0_LSB_INT",
        description="ACC only: Statistical features (no barometer, baseline) - raw LSB integers",
        model_path="model/model_v0_lsb_int/model_v0_lsb_int_xgboost.pkl",
        inference_class="app.data_processing_registry.PipelineSelector",
        uses_barometer=False,
        acc_preprocessing="v1_features",
        baro_preprocessing="none",
        num_features=16,
        acc_features=16,
        baro_features=0,
        acc_feature_names=_STAT_ACC,
        baro_feature_names=[],
        acc_in_lsb=True,
    ),
    ModelName.V1: ModelConfig(
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
        acc_feature_names=_STAT_ACC,
        baro_feature_names=_EMA_BARO,
    ),
    ModelName.V2: ModelConfig(
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
        acc_feature_names=_PAPER_ACC,
        baro_feature_names=_PAPER_BARO,
    ),
    ModelName.V3: ModelConfig(
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
        acc_feature_names=_STAT_ACC,
        baro_feature_names=_PAPER_BARO,
    ),
    ModelName.V4: ModelConfig(
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
        acc_feature_names=_PAPER_ACC,
        baro_feature_names=_EMA_BARO,
    ),
    ModelName.V5: ModelConfig(
        name="V5",
        description="Raw features with minimal preprocessing",
        model_path="model/model_v5/model_v5_xgboost.pkl",
        inference_class="app.data_processing_registry.PipelineSelector",
        uses_barometer=True,
        acc_preprocessing="raw",
        baro_preprocessing="raw",
        num_features=22,
        acc_features=16,
        baro_features=6,
        acc_feature_names=_RAW_ACC,
        baro_feature_names=_RAW_BARO,
    ),
    ModelName.V1_TUNED: ModelConfig(
        name="V1_TUNED",
        description="V1 with tuned hyperparameters (optimized for recall)",
        model_path="model/model_v1_tuned/model_v1_tuned.pkl",
        inference_class="app.data_processing_registry.PipelineSelector",
        uses_barometer=True,
        acc_preprocessing="v1_features",
        baro_preprocessing="v1_ema",
        num_features=22,
        acc_features=16,
        baro_features=6,
        acc_feature_names=_STAT_ACC,
        baro_feature_names=_EMA_BARO,
    ),
    ModelName.V3_TUNED: ModelConfig(
        name="V3_TUNED",
        description="V3 with tuned hyperparameters (optimized for recall)",
        model_path="model/model_v3_tuned/model_v3_tuned.pkl",
        inference_class="app.data_processing_registry.PipelineSelector",
        uses_barometer=True,
        acc_preprocessing="v1_features",
        baro_preprocessing="v2_paper",
        num_features=20,
        acc_features=16,
        baro_features=4,
        acc_feature_names=_STAT_ACC,
        baro_feature_names=_PAPER_BARO,
    ),
    ModelName.V5_LSB: ModelConfig(
        name="V5_LSB",
        description="Raw features with minimal preprocessing (LSB version)",
        model_path="model/model_v5_lsb/model_v5_lsb_xgboost.pkl",
        inference_class="app.data_processing_registry.PipelineSelector",
        uses_barometer=True,
        acc_preprocessing="raw",
        baro_preprocessing="raw",
        num_features=22,
        acc_features=16,
        baro_features=6,
        acc_feature_names=_RAW_ACC,
        baro_feature_names=_RAW_BARO,
        acc_in_lsb=True,
    ),
}