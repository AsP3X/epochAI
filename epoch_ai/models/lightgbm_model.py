"""LightGBM model wrapper with training, prediction, persistence and importances."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from epoch_ai.config.settings import ModelConfig
from epoch_ai.models.base import BaseModel
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


class LightGBMModel(BaseModel):
    """A thin, walk-forward-friendly wrapper around LightGBM.

    Supports both binary classification (predicting P(up)) and regression
    (predicting forward return), optional time-ordered early stopping, per-sample
    weighting (for recency emphasis) and feature-importance extraction.
    """

    def __init__(self, config: ModelConfig, task: str = "classification") -> None:
        self.config = config
        self.task = task
        self.booster: lgb.Booster | None = None
        self.feature_names_: list[str] | None = None
        self.best_iteration_: int | None = None

    # -------------------------------------------------------------------- train
    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
        val_fraction: float = 0.15,
    ) -> LightGBMModel:
        """Fit the booster, using a time-ordered validation tail for early stopping.

        Args:
            x: Feature matrix (rows in chronological order).
            y: Target aligned to ``x``.
            sample_weight: Optional per-row weights (e.g. recency decay).
            val_fraction: Fraction of the *most recent* rows held out for early
                stopping. Set to 0 to disable early stopping.

        Returns:
            ``self`` (fitted).
        """
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        self.feature_names_ = list(x.columns)

        params = dict(self.config.params)
        params["objective"] = "binary" if self.task == "classification" else "regression"
        params["metric"] = "binary_logloss" if self.task == "classification" else "l2"

        use_es = (
            self.config.early_stopping_rounds is not None
            and 0.0 < val_fraction < 0.5
            and len(x) >= 200
        )
        callbacks = [lgb.log_evaluation(period=0)]
        valid_sets = None

        if use_es:
            split = int(len(x) * (1.0 - val_fraction))
            train_set = lgb.Dataset(
                x.iloc[:split],
                label=y.iloc[:split],
                weight=None if sample_weight is None else sample_weight[:split],
            )
            valid_set = lgb.Dataset(
                x.iloc[split:],
                label=y.iloc[split:],
                weight=None if sample_weight is None else sample_weight[split:],
                reference=train_set,
            )
            valid_sets = [valid_set]
            callbacks.append(
                lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)
            )
        else:
            train_set = lgb.Dataset(x, label=y, weight=sample_weight)

        self.booster = lgb.train(
            params,
            train_set,
            num_boost_round=self.config.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        self.best_iteration_ = self.booster.best_iteration or self.config.num_boost_round
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predict P(up) (classification) or forward return (regression)."""
        if self.booster is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        if self.feature_names_ is not None:
            x = x[self.feature_names_]
        return self.booster.predict(x, num_iteration=self.best_iteration_)

    # --------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        """Persist the booster (and feature order) to ``path``."""
        if self.booster is None:
            raise RuntimeError("Cannot save an untrained model.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(path)

    @classmethod
    def load(cls, path: str, config: ModelConfig, task: str = "classification") -> LightGBMModel:
        """Load a booster previously saved with :meth:`save`."""
        model = cls(config, task=task)
        model.booster = lgb.Booster(model_file=path)
        model.feature_names_ = model.booster.feature_name()
        model.best_iteration_ = model.booster.best_iteration or None
        return model

    # ----------------------------------------------------------------- insights
    def feature_importance(self) -> pd.Series:
        """Return gain-based feature importances, sorted descending."""
        if self.booster is None:
            raise RuntimeError("Model is not trained.")
        importance = self.booster.feature_importance(importance_type="gain")
        names = self.booster.feature_name()
        return pd.Series(importance, index=names, name="gain").sort_values(ascending=False)
