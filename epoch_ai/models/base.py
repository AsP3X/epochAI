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
