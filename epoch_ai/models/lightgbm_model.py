"""LightGBM model wrapper with training, prediction, persistence and importances."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from epoch_ai.config.settings import ModelConfig
from epoch_ai.models.base import BaseModel
from epoch_ai.models.calibration import ProbabilityCalibrator
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)

#: Suffix of the JSON sidecar storing the (optional) probability calibrator.
CALIBRATION_SUFFIX = ".calibration.json"


class LightGBMModel(BaseModel):
    """A thin, walk-forward-friendly wrapper around LightGBM.

    Supports both binary classification (predicting P(up)) and regression
    (predicting forward return), optional time-ordered early stopping, per-sample
    weighting (for recency emphasis), balanced class weighting, post-hoc probability
    calibration and feature-importance extraction.
    """

    BACKEND = "lightgbm"
    MODEL_FILENAME = "model.txt"

    def __init__(self, config: ModelConfig, task: str = "classification") -> None:
        self.config = config
        self.task = task
        self.booster: lgb.Booster | None = None
        self.feature_names_: list[str] | None = None
        self.best_iteration_: int | None = None
        #: Fitted probability calibrator (classification only); ``None`` = raw output.
        self.calibrator_: ProbabilityCalibrator | None = None

    # -------------------------------------------------------------------- device
    def _device_params(self) -> dict[str, object]:
        """LightGBM device params; empty for CPU so existing behaviour is unchanged."""
        device = self.config.device
        if device == "cpu":
            return {}
        params: dict[str, object] = {"device_type": device}
        # Platform/device ids only apply to the OpenCL ("gpu") backend; CUDA ignores
        # the platform id. ``-1`` means "let LightGBM auto-select".
        if device == "gpu" and self.config.gpu_platform_id >= 0:
            params["gpu_platform_id"] = self.config.gpu_platform_id
        if self.config.gpu_device_id >= 0:
            params["gpu_device_id"] = self.config.gpu_device_id
        return params

    def _train(
        self,
        params: dict[str, object],
        train_set: lgb.Dataset,
        *,
        num_boost_round: int,
        valid_sets: list[lgb.Dataset] | None,
        callbacks: list,
    ) -> lgb.Booster:
        """Train a booster, transparently downgrading GPU -> CPU on failure.

        GPU support is *optional*: if the installed LightGBM was not built with GPU
        support (or no compatible device is present), the first GPU attempt raises a
        ``LightGBMError``. Rather than fail the whole pipeline we log a warning, mutate
        ``params`` to CPU (so any subsequent refit in the same fit stays on CPU), and
        retry. This guarantees a model can always be trained on CPU.
        """
        try:
            return lgb.train(
                params,
                train_set,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                callbacks=callbacks,
            )
        except lgb.basic.LightGBMError as exc:
            if params.get("device_type", "cpu") == "cpu":
                raise
            logger.warning(
                "LightGBM GPU training failed (%s); falling back to CPU.", exc
            )
            params["device_type"] = "cpu"
            params.pop("gpu_platform_id", None)
            params.pop("gpu_device_id", None)
            return lgb.train(
                params,
                train_set,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                callbacks=callbacks,
            )

    # -------------------------------------------------------------------- train
    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
        val_fraction: float | None = None,
    ) -> LightGBMModel:
        """Fit the booster on a time-ordered split, then optionally calibrate.

        The most-recent ``val_fraction`` of rows is held out (chronologically) and
        reused for both early stopping and probability calibration, so neither peeks
        at future bars relative to the training rows.

        Args:
            x: Feature matrix (rows in chronological order).
            y: Target aligned to ``x``.
            sample_weight: Optional per-row weights (e.g. recency decay).
            val_fraction: Fraction of the *most recent* rows held out. ``None`` uses
                ``config.val_fraction``; ``0`` disables early stopping + calibration.

        Returns:
            ``self`` (fitted).
        """
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        self.feature_names_ = list(x.columns)
        self.calibrator_ = None

        is_classification = self.task == "classification"
        params = dict(self.config.params)
        params["objective"] = "binary" if is_classification else "regression"
        params["metric"] = "binary_logloss" if is_classification else "l2"
        # Optional GPU/CUDA acceleration (no-op for the default CPU device).
        params.update(self._device_params())

        # Balanced class weighting: scale the positive ("up") class so a skewed
        # up/down label balance does not bias the booster toward the majority class.
        if is_classification and self.config.class_weight == "balanced":
            params["scale_pos_weight"] = _scale_pos_weight(y)

        if val_fraction is None:
            val_fraction = self.config.val_fraction

        # A validation tail is worthwhile only with enough rows to be meaningful.
        has_val_tail = 0.0 < val_fraction < 0.5 and len(x) >= 200
        use_es = self.config.early_stopping_rounds is not None and has_val_tail
        callbacks = [lgb.log_evaluation(period=0)]
        valid_sets = None
        split = len(x)

        if has_val_tail:
            split = int(len(x) * (1.0 - val_fraction))

        if use_es:
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

        self.booster = self._train(
            params,
            train_set,
            num_boost_round=self.config.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        self.best_iteration_ = self.booster.best_iteration or self.config.num_boost_round

        # Fit probability calibration on the held-out validation tail (classification
        # only). Without a tail we cannot calibrate honestly, so we keep raw output.
        # Calibration is fit on the tail-naive booster (trained on [:split]); the map
        # it learns is reused below even after the optional full-data refit, which
        # targets the same objective and so produces a near-identical raw distribution.
        if is_classification and self.config.calibration != "none" and has_val_tail:
            raw_val = self.booster.predict(x.iloc[split:], num_iteration=self.best_iteration_)
            self.calibrator_ = ProbabilityCalibrator.fit(
                np.asarray(raw_val), y.iloc[split:].to_numpy(), self.config.calibration
            )

        # Refit on the full training window (including the validation tail) for the
        # early-stopping-selected number of rounds. The iteration count was chosen on
        # genuine out-of-sample data, but the deployed model should not throw away its
        # freshest rows — especially important for recency-sensitive crypto regimes.
        if use_es and self.config.refit_full_after_es and split < len(x):
            full_set = lgb.Dataset(x, label=y, weight=sample_weight)
            self.booster = self._train(
                params,
                full_set,
                num_boost_round=self.best_iteration_,
                valid_sets=None,
                callbacks=[lgb.log_evaluation(period=0)],
            )
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predict calibrated P(up) (classification) or forward return (regression)."""
        if self.booster is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        if self.feature_names_ is not None:
            x = x[self.feature_names_]
        raw = self.booster.predict(x, num_iteration=self.best_iteration_)
        if self.calibrator_ is not None:
            return self.calibrator_.transform(raw)
        return raw

    # --------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        """Persist the booster, plus a calibration sidecar when one was fitted."""
        if self.booster is None:
            raise RuntimeError("Cannot save an untrained model.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(path)
        # Calibrator (if any) lives next to the booster so the registry/export can
        # carry it alongside the open-weights model file.
        sidecar = Path(path).with_name(Path(path).name + CALIBRATION_SUFFIX)
        if self.calibrator_ is not None:
            sidecar.write_text(json.dumps(self.calibrator_.to_dict(), indent=2), encoding="utf-8")
        elif sidecar.exists():
            sidecar.unlink()

    @classmethod
    def load(cls, path: str, config: ModelConfig, task: str = "classification") -> LightGBMModel:
        """Load a booster (and its calibration sidecar, if present)."""
        model = cls(config, task=task)
        model.booster = lgb.Booster(model_file=path)
        model.feature_names_ = model.booster.feature_name()
        model.best_iteration_ = model.booster.best_iteration or None
        sidecar = Path(path).with_name(Path(path).name + CALIBRATION_SUFFIX)
        if sidecar.exists():
            model.calibrator_ = ProbabilityCalibrator.from_dict(
                json.loads(sidecar.read_text(encoding="utf-8"))
            )
        return model

    # ----------------------------------------------------------------- insights
    def feature_importance(self) -> pd.Series:
        """Return gain-based feature importances, sorted descending."""
        if self.booster is None:
            raise RuntimeError("Model is not trained.")
        importance = self.booster.feature_importance(importance_type="gain")
        names = self.booster.feature_name()
        return pd.Series(importance, index=names, name="gain").sort_values(ascending=False)


def _scale_pos_weight(y: pd.Series) -> float:
    """Return ``n_negative / n_positive`` for balanced binary class weighting.

    Falls back to ``1.0`` (no reweighting) when a class is absent, so a degenerate
    single-class training window never produces an infinite or zero weight.
    """
    labels = y.to_numpy()
    n_pos = float((labels > 0.5).sum())
    n_neg = float((labels <= 0.5).sum())
    if n_pos <= 0.0 or n_neg <= 0.0:
        return 1.0
    return n_neg / n_pos
