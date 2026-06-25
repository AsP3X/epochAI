"""Backend-agnostic model construction.

The prediction engine, retrain job, promotion gate and live loop all build models
through :func:`build_model` so the concrete learner is chosen by configuration
(``model.backend``) rather than hard-coded. LightGBM is the default; XGBoost is an
optional backend that enables real CUDA-GPU training on NVIDIA cards.

The XGBoost implementation is imported lazily so the core pipeline keeps running when
the optional ``xgboost`` package is not installed.
"""

from __future__ import annotations

from epoch_ai.config.settings import ModelConfig
from epoch_ai.models.base import BaseModel
from epoch_ai.models.lightgbm_model import LightGBMModel

#: Supported backend identifiers (also stored in registry metadata).
BACKENDS = ("lightgbm", "xgboost")


def model_class(backend: str) -> type[BaseModel]:
    """Return the model class for ``backend`` (lazy-importing XGBoost on demand)."""
    if backend == "lightgbm":
        return LightGBMModel
    if backend == "xgboost":
        try:
            from epoch_ai.models.xgboost_model import XGBoostModel
        except ImportError as exc:  # pragma: no cover - exercised only without xgboost
            raise ImportError(
                "model.backend='xgboost' requires the optional 'xgboost' package. "
                "Install it with `pip install xgboost` (or "
                "`pip install -r requirements-optional.txt`)."
            ) from exc
        return XGBoostModel
    raise ValueError(f"Unknown model backend: {backend!r}. Expected one of {BACKENDS}.")


def model_filename(backend: str) -> str:
    """Return the registry filename a backend persists its booster under."""
    return model_class(backend).MODEL_FILENAME


def build_model(config: ModelConfig, task: str = "classification") -> BaseModel:
    """Construct the configured model backend for ``task``."""
    return model_class(config.backend)(config, task=task)
