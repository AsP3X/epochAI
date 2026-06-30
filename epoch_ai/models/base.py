"""Abstract model interface.

Defining a small interface keeps the prediction engine decoupled from the concrete
learner, making it straightforward to later add an incremental ``River`` model (or an
ensemble) alongside the LightGBM default.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Common interface for all prediction models."""

    task: str = "classification"

    #: Registry identifier for this backend (stored in model metadata).
    BACKEND: str = "lightgbm"
    #: Filename used to persist the open-weights booster inside a version dir.
    MODEL_FILENAME: str = "model.txt"

    @abstractmethod
    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> BaseModel:
        """Train the model on features ``x`` and target ``y``."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Return predictions.

        For classification this is P(up) in ``[0, 1]``; for regression it is the
        expected forward return.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model to ``path``."""
        raise NotImplementedError

    @abstractmethod
    def feature_importance(self) -> pd.Series:
        """Return feature importances indexed by feature name."""
        raise NotImplementedError


class MultiHeadModel(BaseModel):
    """Shared interface for multi-horizon torch backends (``evolved_nn``, ``tcn``).

    These models emit, per configured horizon, a direction logit plus return quantiles
    (see :class:`~epoch_ai.models.multi_head.MultiHeadSpec`). The progressive engine,
    runtime, promotion gate, acceptance scorer and live loop treat any subclass uniformly
    via :meth:`predict_structured`, so adding a new multi-head backend does not require
    touching those consumers.

    Attributes:
        multi_head_spec_: Output layout once trained multi-horizon (else ``None``).
        primary_horizon_: Primary head horizon (drives single-value ``predict``).
        sequence_lookback: For sequence models (TCN), the number of past bars each
            prediction depends on; callers should supply that many preceding feature
            rows as context. ``None`` for dense models (MLP) that predict per-row.
    """

    #: Set lazily by ``fit``/``load``; declared here so consumers can introspect.
    multi_head_spec_ = None
    primary_horizon_ = None
    #: Sequence models override with their window length; dense models leave ``None``.
    sequence_lookback: int | None = None

    def predict_logits(self, x: pd.DataFrame) -> np.ndarray:
        """Return raw multi-head logits (``n_rows x n_outputs``)."""
        raise NotImplementedError

    def predict_structured(
        self, x: pd.DataFrame
    ) -> dict[int, dict[str, np.ndarray | float]]:
        """Parse multi-head outputs into per-horizon quantile returns and P(up)."""
        raise NotImplementedError

    def seed_payload(self) -> dict:
        """Return warm-start kwargs for the next retrain's ``fit`` (empty by default)."""
        return {}
