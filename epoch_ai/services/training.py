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


@dataclass(slots=True)
class TrainResult:
    """Outcome of an explicit training job."""

    model_version: str | None
    train_rows: int
    walk_forward_steps: int
    step_history: pd.DataFrame
    feature_importance: pd.Series = field(default_factory=pd.Series)


class TrainingService:
    """Train models on historical data and evaluate learning quality."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def download(self, *, n_bars: int | None = None, force: bool = False) -> pd.DataFrame:
        """Fetch or synthesize OHLCV history and cache as parquet."""
        return HistoricalDownloader(self.config).load_or_download(
            self.config.primary_symbol,
            n_bars=n_bars,
            force=force,
        )

    def train(
        self,
        *,
        n_bars: int | None = None,
        max_steps: int | None = None,
        log_predictions: bool = False,
        register: bool = True,
    ) -> TrainResult:
        """Run progressive walk-forward training and register the final model.

        This is the primary **train-the-AI** operation: walk forward through history,
        learn from realised outcomes, and persist a versioned model to the registry.
        """
        cfg = self.config.model_copy(deep=True)
        if max_steps is not None:
            cfg.walk_forward.max_steps = max_steps

        market = self.download(n_bars=n_bars)
        features = FeaturePipeline(cfg).transform(market)
        store = PredictionStore(cfg.logging.db_path) if log_predictions else None

        try:
            engine = ProgressiveLearningEngine(cfg, register_models=register)
            learning = engine.run(market, features, store=store)
        finally:
            if store is not None:
                store.close()

        if learning.step_history.empty:
            raise RuntimeError("Training produced no walk-forward steps; increase --bars.")

        train_rows = int(learning.step_history["train_rows"].iloc[-1])
        return TrainResult(
            model_version=learning.final_model_version,
            train_rows=train_rows,
            walk_forward_steps=len(learning.step_history),
            step_history=learning.step_history,
            feature_importance=learning.feature_importance,
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
        cfg = self.config.model_copy(deep=True)
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
