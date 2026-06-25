"""XGBoost model wrapper — an optional, CUDA-GPU-capable alternative backend.

This mirrors :class:`~epoch_ai.models.lightgbm_model.LightGBMModel` (same constructor,
``fit``/``predict``/``save``/``load``/``feature_importance`` surface, time-ordered
early stopping, recency sample weights, balanced class weighting, post-hoc probability
calibration) so it is a drop-in registry citizen. Its reason to exist is hardware:
XGBoost ships prebuilt CUDA wheels, so ``model.device="cuda"`` trains on an NVIDIA GPU,
whereas the stock LightGBM wheel has no CUDA build.

Weights are saved as plain XGBoost JSON (open and inspectable), keeping the project's
open-weights guarantee.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from epoch_ai.config.settings import ModelConfig
from epoch_ai.models.base import BaseModel
from epoch_ai.models.calibration import ProbabilityCalibrator
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)

#: Suffix of the JSON sidecar storing the (optional) probability calibrator.
CALIBRATION_SUFFIX = ".calibration.json"
#: Booster attribute key used to persist the early-stopping iteration across save/load.
_BEST_ITER_ATTR = "epoch_ai_best_iteration"


class XGBoostModel(BaseModel):
    """A walk-forward-friendly XGBoost wrapper with optional CUDA-GPU training."""

    BACKEND = "xgboost"
    MODEL_FILENAME = "model.json"

    def __init__(self, config: ModelConfig, task: str = "classification") -> None:
        self.config = config
        self.task = task
        self.booster: xgb.Booster | None = None
        self.feature_names_: list[str] | None = None
        self.best_iteration_: int | None = None
        #: Fitted probability calibrator (classification only); ``None`` = raw output.
        self.calibrator_: ProbabilityCalibrator | None = None

    # -------------------------------------------------------------------- params
    def _device(self) -> str:
        """Resolve the XGBoost ``device`` string from config.

        XGBoost only accelerates on CUDA GPUs, so both ``"gpu"`` and ``"cuda"`` map to
        ``"cuda"`` (optionally pinned to a device ordinal). ``"cpu"`` stays on CPU.
        """
        if self.config.device == "cpu":
            return "cpu"
        if self.config.gpu_device_id >= 0:
            return f"cuda:{self.config.gpu_device_id}"
        return "cuda"

    def _params(self) -> dict[str, object]:
        """Translate the shared LightGBM-style knobs into XGBoost params.

        The project's ``model.params`` are expressed in LightGBM vocabulary; we map the
        common ones so a single config drives either backend. Unmapped LightGBM-only
        keys (e.g. ``bagging_freq``) are ignored rather than passed through.
        """
        src = dict(self.config.params)
        is_classification = self.task == "classification"
        # ``lossguide`` + ``max_leaves`` reproduces LightGBM's leaf-wise growth so the
        # ``num_leaves`` knob keeps a comparable meaning across backends.
        params: dict[str, object] = {
            "objective": "binary:logistic" if is_classification else "reg:squarederror",
            "eval_metric": "logloss" if is_classification else "rmse",
            "tree_method": "hist",
            "device": self._device(),
            "grow_policy": "lossguide",
            "learning_rate": float(src.get("learning_rate", 0.03)),
            "max_leaves": int(src.get("num_leaves", 63)),
            # LightGBM uses -1 (= no limit); XGBoost uses 0 for "no limit".
            "max_depth": max(int(src.get("max_depth", -1)), 0),
            "subsample": float(src.get("bagging_fraction", 1.0)),
            "colsample_bytree": float(src.get("feature_fraction", 1.0)),
            "reg_alpha": float(src.get("lambda_l1", 0.0)),
            "reg_lambda": float(src.get("lambda_l2", 1.0)),
            "gamma": float(src.get("min_gain_to_split", 0.0)),
            "min_child_weight": float(src.get("min_data_in_leaf", 1.0)),
            "verbosity": 0,
        }
        return params

    def _train(
        self,
        params: dict[str, object],
        dtrain: xgb.DMatrix,
        *,
        num_boost_round: int,
        evals: list[tuple[xgb.DMatrix, str]] | None,
        early_stopping_rounds: int | None,
    ) -> xgb.Booster:
        """Train a booster, transparently downgrading CUDA -> CPU on failure.

        GPU training is optional: if no CUDA device/driver is available the first
        attempt raises ``XGBoostError``. We log a warning, mutate ``params`` to CPU (so a
        subsequent refit in the same fit stays on CPU) and retry, guaranteeing a model
        can always be trained.
        """
        try:
            return xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=evals or [],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )
        except xgb.core.XGBoostError as exc:
            if str(params.get("device", "cpu")).startswith("cpu"):
                raise
            logger.warning("XGBoost GPU training failed (%s); falling back to CPU.", exc)
            params["device"] = "cpu"
            return xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=evals or [],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )

    # -------------------------------------------------------------------- train
    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
        val_fraction: float | None = None,
    ) -> XGBoostModel:
        """Fit the booster on a time-ordered split, then optionally calibrate.

        The most-recent ``val_fraction`` of rows is held out (chronologically) and
        reused for both early stopping and probability calibration, mirroring the
        LightGBM backend so neither peeks at future bars relative to the training rows.
        """
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        self.feature_names_ = list(x.columns)
        self.calibrator_ = None

        is_classification = self.task == "classification"
        params = self._params()
        if is_classification and self.config.class_weight == "balanced":
            params["scale_pos_weight"] = _scale_pos_weight(y)

        if val_fraction is None:
            val_fraction = self.config.val_fraction

        has_val_tail = 0.0 < val_fraction < 0.5 and len(x) >= 200
        use_es = self.config.early_stopping_rounds is not None and has_val_tail
        split = int(len(x) * (1.0 - val_fraction)) if has_val_tail else len(x)

        if use_es:
            dtrain = self._qdmatrix(
                x.iloc[:split],
                y.iloc[:split],
                None if sample_weight is None else sample_weight[:split],
            )
            dvalid = self._qdmatrix(
                x.iloc[split:],
                y.iloc[split:],
                None if sample_weight is None else sample_weight[split:],
                ref=dtrain,
            )
            self.booster = self._train(
                params,
                dtrain,
                num_boost_round=self.config.num_boost_round,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=self.config.early_stopping_rounds,
            )
            self.best_iteration_ = int(self.booster.best_iteration) + 1
        else:
            dtrain = self._qdmatrix(x, y, sample_weight)
            self.booster = self._train(
                params,
                dtrain,
                num_boost_round=self.config.num_boost_round,
                evals=None,
                early_stopping_rounds=None,
            )
            self.best_iteration_ = self.config.num_boost_round

        # Calibrate on the held-out tail (classification only). Fit on the tail-naive
        # booster; the learned map is reused after the optional full-data refit, which
        # targets the same objective and so yields a near-identical raw distribution.
        if is_classification and self.config.calibration != "none" and has_val_tail:
            raw_val = self._raw_predict(x.iloc[split:])
            self.calibrator_ = ProbabilityCalibrator.fit(
                np.asarray(raw_val), y.iloc[split:].to_numpy(), self.config.calibration
            )

        # Refit on the full window (incl. validation tail) for the ES-selected rounds so
        # the deployed model does not discard its freshest bars (see LightGBM backend).
        if use_es and self.config.refit_full_after_es and split < len(x):
            full = self._qdmatrix(x, y, sample_weight)
            self.booster = self._train(
                params,
                full,
                num_boost_round=self.best_iteration_,
                evals=None,
                early_stopping_rounds=None,
            )
        self._store_best_iteration()
        return self

    def _ordered(self, x: pd.DataFrame) -> pd.DataFrame:
        return x[self.feature_names_] if self.feature_names_ else x

    def _qdmatrix(
        self,
        x: pd.DataFrame,
        y: pd.Series | None = None,
        weight: np.ndarray | None = None,
        *,
        ref: xgb.QuantileDMatrix | None = None,
    ) -> xgb.QuantileDMatrix:
        """Build a ``QuantileDMatrix`` for training.

        ``QuantileDMatrix`` is the memory-efficient, GPU-friendly container recommended
        for the ``hist`` tree method: it pre-bins on the target device, avoiding a dense
        host copy per round. Validation matrices must reference the training one so they
        share bin edges.
        """
        return xgb.QuantileDMatrix(
            self._ordered(x),
            label=None if y is None else y,
            weight=weight,
            feature_names=self.feature_names_,
            ref=ref,
        )

    def _dmatrix(self, x: pd.DataFrame) -> xgb.DMatrix:
        """Build a plain ``DMatrix`` for prediction with stable feature names/order."""
        return xgb.DMatrix(self._ordered(x), feature_names=self.feature_names_)

    # ------------------------------------------------------------------ predict
    def _raw_predict(self, x: pd.DataFrame) -> np.ndarray:
        """Raw booster output (P(up) or return) using the best iteration."""
        assert self.booster is not None
        iteration_range = (0, self.best_iteration_) if self.best_iteration_ else (0, 0)
        return self.booster.predict(self._dmatrix(x), iteration_range=iteration_range)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predict calibrated P(up) (classification) or forward return (regression)."""
        if self.booster is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        raw = self._raw_predict(x)
        if self.calibrator_ is not None:
            return self.calibrator_.transform(raw)
        return raw

    # --------------------------------------------------------------- persistence
    def _store_best_iteration(self) -> None:
        """Persist the best iteration as a booster attribute (survives save/load)."""
        if self.booster is not None and self.best_iteration_ is not None:
            self.booster.set_attr(**{_BEST_ITER_ATTR: str(self.best_iteration_)})

    def save(self, path: str) -> None:
        """Persist the booster (XGBoost JSON), plus a calibration sidecar if fitted."""
        if self.booster is None:
            raise RuntimeError("Cannot save an untrained model.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._store_best_iteration()
        self.booster.save_model(path)
        sidecar = Path(path).with_name(Path(path).name + CALIBRATION_SUFFIX)
        if self.calibrator_ is not None:
            sidecar.write_text(json.dumps(self.calibrator_.to_dict(), indent=2), encoding="utf-8")
        elif sidecar.exists():
            sidecar.unlink()

    @classmethod
    def load(cls, path: str, config: ModelConfig, task: str = "classification") -> XGBoostModel:
        """Load a booster (and its calibration sidecar, if present)."""
        model = cls(config, task=task)
        booster = xgb.Booster()
        booster.load_model(path)
        model.booster = booster
        model.feature_names_ = list(booster.feature_names) if booster.feature_names else None
        best_iter = booster.attr(_BEST_ITER_ATTR)
        model.best_iteration_ = int(best_iter) if best_iter is not None else None
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
        scores = self.booster.get_score(importance_type="gain")
        names = self.feature_names_ or list(scores.keys())
        values = [float(scores.get(name, 0.0)) for name in names]
        return pd.Series(values, index=names, name="gain").sort_values(ascending=False)


def _scale_pos_weight(y: pd.Series) -> float:
    """Return ``n_negative / n_positive`` for balanced binary class weighting.

    Falls back to ``1.0`` when a class is absent, so a degenerate single-class training
    window never produces an infinite or zero weight.
    """
    labels = y.to_numpy()
    n_pos = float((labels > 0.5).sum())
    n_neg = float((labels <= 0.5).sum())
    if n_pos <= 0.0 or n_neg <= 0.0:
        return 1.0
    return n_neg / n_pos
