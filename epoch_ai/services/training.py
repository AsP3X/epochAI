"""Training mode — progressive learning, backtests, sweeps, and retraining.

This module is the programmatic entry point for **training the AI**. Future interfaces
(Telegram bot, website API) should call :class:`TrainingService` rather than duplicating
CLI or engine logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from epoch_ai.backtesting.engine import Backtester, BacktestResult
from epoch_ai.config.overrides import apply_overrides
from epoch_ai.config.settings import AppConfig
from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.features.pipeline import FeaturePipeline
from epoch_ai.learning.progressive import ProgressiveLearningEngine
from epoch_ai.learning.promotion import AutoPromoteResult, auto_retrain_and_promote
from epoch_ai.learning.retrain_job import RetrainResult, run_retrain
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.models.registry import ModelRegistry
from epoch_ai.tracking.mlflow_tracker import MLflowTracker
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class TrainResult:
    """Outcome of an explicit training job."""

    model_version: str | None
    train_rows: int
    walk_forward_steps: int
    step_history: pd.DataFrame
    feature_importance: pd.Series = field(default_factory=pd.Series)
    resumed_from_step: int | None = None


class TrainingService:
    """Train models on historical data and evaluate learning quality."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def download(self, *, n_bars: int | None = None, force: bool = False) -> pd.DataFrame:
        """Fetch OHLCV history and cache as parquet.

        When ``model.backend`` is ``evolved_nn``, synthetic fallback is disabled so
        training always uses real exchange or cached real parquet data.
        """
        cfg = self._training_data_config()
        return HistoricalDownloader(cfg).load_or_download(
            cfg.primary_symbol,
            n_bars=n_bars,
            force=force,
        )

    def _training_data_config(self) -> AppConfig:
        """Return config with real-data guarantees for evolved_nn training."""
        cfg = self.config
        if cfg.model.backend != "evolved_nn":
            return cfg
        if cfg.data.use_synthetic_fallback:
            cfg = cfg.model_copy(deep=True)
            cfg.data.use_synthetic_fallback = False
        return cfg

    def train(
        self,
        *,
        n_bars: int | None = None,
        max_steps: int | None = None,
        log_predictions: bool = False,
        register: bool = True,
        resume: bool = True,
        fresh: bool = False,
    ) -> TrainResult:
        """Run progressive walk-forward training and register the final model.

        This is the primary **train-the-AI** operation: walk forward through history,
        learn from realised outcomes, and persist a versioned model to the registry.

        When ``walk_forward.checkpoint_enabled`` is true (default), progress is saved
        after each step. A later call with ``resume=True`` continues from the checkpoint.
        Pass ``fresh=True`` to discard saved progress and start from step 0.
        """
        cfg = self._training_data_config().model_copy(deep=True)
        if max_steps is not None:
            cfg.walk_forward.max_steps = max_steps

        market = self.download(n_bars=n_bars)
        features = FeaturePipeline(cfg).transform(market)
        store = PredictionStore(cfg.logging.db_path) if log_predictions else None

        try:
            engine = ProgressiveLearningEngine(cfg, register_models=register)
            learning = engine.run(
                market,
                features,
                store=store,
                resume=resume,
                fresh=fresh,
            )
        except KeyboardInterrupt:
            logger.info(
                "Walk-forward training interrupted; last completed step checkpoint is preserved."
            )
            raise
        finally:
            if store is not None:
                store.close()

        if learning.step_history.empty:
            if learning.resumed_from_step is not None and max_steps is not None:
                raise RuntimeError(
                    "Training produced no new walk-forward steps; checkpoint step "
                    f"{learning.resumed_from_step} may have reached --max-steps. "
                    "Increase --max-steps or omit it to continue."
                )
            raise RuntimeError("Training produced no walk-forward steps; increase --bars.")

        train_rows = int(learning.step_history["train_rows"].iloc[-1])
        return TrainResult(
            model_version=learning.final_model_version,
            train_rows=train_rows,
            walk_forward_steps=len(learning.step_history),
            step_history=learning.step_history,
            feature_importance=learning.feature_importance,
            resumed_from_step=learning.resumed_from_step,
        )

    def backtest(
        self,
        *,
        n_bars: int | None = None,
        max_steps: int | None = None,
        log_predictions: bool = False,
        register_models: bool = False,
        store: PredictionStore | None = None,
    ) -> BacktestResult:
        """Run a full progressive backtest with strategy metrics."""
        cfg = self._training_data_config().model_copy(deep=True)
        if max_steps is not None:
            cfg.walk_forward.max_steps = max_steps

        market = self.download(n_bars=n_bars)
        features = FeaturePipeline(cfg).transform(market)
        own_store = store is None and log_predictions
        if own_store:
            store = PredictionStore(cfg.logging.db_path)

        try:
            with MLflowTracker(cfg.tracking):
                return Backtester(cfg).run(
                    market,
                    features=features,
                    store=store,
                    register_models=register_models,
                )
        finally:
            if own_store and store is not None:
                store.close()

    def retrain(self, *, min_new_samples: int = 50, n_bars: int | None = None) -> RetrainResult:
        """Refresh the model from logged predictions or parquet fallback."""
        return run_retrain(self.config, min_new_samples=min_new_samples, n_bars=n_bars)

    def auto_retrain(self, *, n_bars: int | None = None) -> AutoPromoteResult:
        """Train a challenger and promote it only if it beats the champion on a holdout.

        This is the safe, self-updating entry point: the registry's promoted model is
        replaced only when out-of-sample quality improves (see
        :mod:`epoch_ai.learning.promotion`).
        """
        return auto_retrain_and_promote(self.config, n_bars=n_bars)

    def tune(
        self,
        sweep: dict[str, Any],
        *,
        n_bars: int | None = None,
        max_steps: int | None = None,
        base_raw: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run experiments defined in a sweep dict and return result summaries."""
        experiments = sweep.get("experiments", [])
        if not experiments:
            raise ValueError("Sweep has no experiments.")

        base = base_raw if base_raw is not None else self.config.model_dump()
        market = self.download(n_bars=n_bars)
        results: list[dict[str, Any]] = []

        for exp in experiments:
            name = exp.get("name", "unnamed")
            overrides = exp.get("overrides", {})
            merged = apply_overrides(base, overrides)
            exp_config = AppConfig.model_validate(merged)
            if max_steps is not None:
                exp_config.walk_forward.max_steps = max_steps

            features = FeaturePipeline(exp_config).transform(market)
            bt = Backtester(exp_config).run(market, features=features)
            results.append(
                {
                    "name": name,
                    "overrides": overrides,
                    "metrics": bt.metrics,
                    "learning_improvement": bt.learning_improvement,
                    "learning_curve": bt.learning_curve,
                }
            )
        return results

    def list_models(self) -> list[dict[str, Any]]:
        """Return metadata for all registered model versions."""
        return ModelRegistry(self.config.model.model_dir).list_versions()
