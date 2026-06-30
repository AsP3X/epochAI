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
from epoch_ai.data.training_policy import assert_training_cache_real, config_for_supervised_training
from epoch_ai.features.pipeline import FeaturePipeline
from epoch_ai.learning.progressive import ProgressiveLearningEngine
from epoch_ai.learning.promotion import AutoPromoteResult, auto_retrain_and_promote
from epoch_ai.learning.retrain_job import RetrainResult, run_retrain
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.models.registry import ModelRegistry
from epoch_ai.tracking.mlflow_tracker import MLflowTracker
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


def minimum_training_bars(config: AppConfig) -> int:
    """Conservative raw OHLCV bar count for default walk-forward + 1m feature stack.

    Uses an empirical resolved/raw ratio (~0.50 on 1m with full features + neutral_band
    labels). Warm-up and label drops are already reflected in that ratio — do not add
    ``min_buffer_bars`` again or the estimate overshoots (~95k vs ~86k).
    """
    wf = config.walk_forward
    pred = config.prediction
    max_h = max(pred.horizons) if pred.horizons else pred.horizon
    resolved_ratio = 0.50
    need_resolved = wf.initial_train_period + 1
    return int(need_resolved / resolved_ratio) + max_h + 500


def resolve_training_bars(
    config: AppConfig,
    n_bars: int | None,
    *,
    full_history: bool = False,
) -> int | None:
    """Choose download depth for ``train``: explicit cap, cached tail, or full backfill."""
    if n_bars is not None or full_history:
        return n_bars

    downloader = HistoricalDownloader(config)
    cache_path = downloader._cache_path(config.primary_symbol)
    if not cache_path.exists():
        return None

    cached = pd.read_parquet(cache_path)
    if cached.empty or not downloader._cache_is_live(cached):
        return None

    target = downloader._default_bar_count()
    if len(cached) >= target * 0.95 and downloader._cache_covers_start(cached):
        return None

    logger.info(
        "Using %d cached bars for training (%s). Pass --full-history to backfill "
        "~%d bars from exchange start, or --bars N to cap explicitly.",
        len(cached),
        cache_path,
        target,
    )
    return len(cached)


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

    def download(
        self,
        *,
        n_bars: int | None = None,
        force: bool = False,
        fetch_if_missing: bool = True,
    ) -> pd.DataFrame:
        """Fetch OHLCV history and cache as parquet.

        Supervised training always disables synthetic fallback and requires a
        provenanced exchange cache (see ``epoch_ai.data.training_policy``).
        """
        cfg = self._training_data_config()
        market = HistoricalDownloader(cfg).load_or_download(
            cfg.primary_symbol,
            n_bars=n_bars,
            force=force,
            fetch_if_missing=fetch_if_missing,
        )
        assert_training_cache_real(cfg, cfg.primary_symbol)
        return market

    def _training_data_config(self) -> AppConfig:
        """Return config with real-data guarantees for all training backends."""
        return config_for_supervised_training(self.config)

    def train(
        self,
        *,
        n_bars: int | None = None,
        max_steps: int | None = None,
        log_predictions: bool = False,
        register: bool = True,
        resume: bool = True,
        fresh: bool = False,
        full_history: bool = False,
        refresh_data: bool = False,
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

        min_bars = minimum_training_bars(cfg)
        n_bars = resolve_training_bars(cfg, n_bars, full_history=full_history)
        if n_bars is not None and n_bars < min_bars:
            raise ValueError(
                f"--bars {n_bars} is too small for training: need at least {min_bars} "
                f"(initial_train_period={cfg.walk_forward.initial_train_period}, "
                f"min_buffer_bars={cfg.execution.min_buffer_bars}). "
                f"Example: python -m epoch_ai train --bars {min_bars} --log-predictions"
            )
        if n_bars is None:
            logger.info(
                "Full-history train (~%d bars target). This can take a long time on first "
                "run. For a capped run: download with --bars N, then train (auto-uses "
                "cache) or pass --bars N explicitly.",
                HistoricalDownloader(cfg)._default_bar_count(),
            )

        market = self.download(
            n_bars=n_bars,
            force=refresh_data,
            fetch_if_missing=refresh_data or full_history,
        )
        features = FeaturePipeline(cfg).transform(market)
        from epoch_ai.learning.progressive import (
            count_resolved_walk_forward_rows,
            suggest_training_bars,
        )

        resolved = count_resolved_walk_forward_rows(market, features, cfg)
        if resolved <= cfg.walk_forward.initial_train_period:
            suggested = suggest_training_bars(len(market), resolved, cfg)
            raise ValueError(
                f"{len(market)} bars yield {resolved} resolved training rows; need > "
                f"{cfg.walk_forward.initial_train_period} for initial_train_period. "
                f"Try --bars {suggested}, or reduce walk_forward.initial_train_period "
                "for smoke tests."
            )
        if features.empty:
            raise ValueError(
                f"Feature pipeline produced no rows from {len(market)} bars "
                "(all dropped as warm-up/NaN). Use more history "
                f"(>= {min_bars} recommended) or reduce execution.min_buffer_bars / "
                "walk_forward.initial_train_period for smoke tests."
            )
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
