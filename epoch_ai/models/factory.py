"""Backend-agnostic model construction.

The prediction engine, retrain job, promotion gate and live loop all build models
through :func:`build_model` so the concrete learner is chosen by configuration
(``model.backend``) rather than hard-coded. ``evolved_nn`` (default) uses an
evolutionary PyTorch MLP; LightGBM and XGBoost remain optional fallbacks.

Heavy backends are imported lazily so the core pipeline keeps running when optional
packages are not installed (use ``model.backend=lightgbm`` in that case).
"""

from __future__ import annotations

from epoch_ai.config.settings import ModelConfig
from epoch_ai.models.base import BaseModel
from epoch_ai.models.lightgbm_model import LightGBMModel

#: Supported backend identifiers (also stored in registry metadata).
BACKENDS = ("evolved_nn", "lightgbm", "xgboost")


def model_class(backend: str) -> type[BaseModel]:
    """Return the model class for ``backend`` (lazy-importing optional deps on demand)."""
    if backend == "lightgbm":
        return LightGBMModel
    if backend == "evolved_nn":
        try:
            from epoch_ai.models.evolved_nn_model import EvolvedNNModel
        except ImportError as exc:  # pragma: no cover - exercised only without torch
            raise ImportError(
                "model.backend='evolved_nn' requires PyTorch. "
                "Install with `pip install torch` (or "
                "`pip install -r requirements-optional.txt`), or set "
                "model.backend='lightgbm'."
            ) from exc
        return EvolvedNNModel
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
