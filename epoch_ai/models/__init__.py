"""Model layer: pluggable GBM backends, base interface and versioned registry.

LightGBM is the default backend; XGBoost is an optional, CUDA-GPU-capable backend
(lazy-imported via :func:`epoch_ai.models.factory.build_model`).
"""

from __future__ import annotations

from epoch_ai.models.base import BaseModel
from epoch_ai.models.factory import build_model, model_class, model_filename
from epoch_ai.models.lightgbm_model import LightGBMModel
from epoch_ai.models.registry import ModelRegistry

__all__ = [
    "BaseModel",
    "LightGBMModel",
    "ModelRegistry",
    "build_model",
    "model_class",
    "model_filename",
]
