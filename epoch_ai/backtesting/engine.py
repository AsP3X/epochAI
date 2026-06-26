"""Backtesting engine.

Runs the progressive walk-forward simulation and turns the resulting per-bar
predictions into a realistic equity curve (with fees + slippage), then computes the
full trading-metrics suite. ``vectorbt`` is used for the portfolio simulation when
installed and enabled; otherwise an equivalent native simulation is used so the
backtest always runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from epoch_ai.backtesting.metrics import compute_metrics
from epoch_ai.backtesting.reporting import count_rebalances
from epoch_ai.config.settings import AppConfig
from epoch_ai.features.pipeline import FeaturePipeline
from epoch_ai.learning.curve_analysis import summarize_step_history
from epoch_ai.learning.degradation import summarize_learning_degradation
from epoch_ai.learning.progressive import ProgressiveLearningEngine, ProgressiveResult
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.utils.logging import get_logger
from epoch_ai.utils.timeframe import annualization_factor, timeframe_to_minutes

logger = get_logger(__name__)


@dataclass(slots=True)
class BacktestResult:
    """Aggregated results of a progressive backtest."""

    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    equity_curve: pd.Series
    strategy_returns: pd.Series
    learning: ProgressiveResult
    learning_improvement: dict[str, float]
    learning_curve: dict[str, float]


class Backtester:
    """Run a progressive walk-forward backtest end-to-end."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _ann_factor(self) -> tuple[float, float]:
        if self.config.backtest.annualization_factor is not None:
            ppy = float(self.config.backtest.annualization_factor)
            return float(np.sqrt(ppy)), ppy
        factor = annualization_factor(self.config.timeframe)
        ppy = (365 * 24 * 60) / timeframe_to_minutes(self.config.timeframe)
        return factor, ppy

    def _strategy_returns(
        self, market: pd.DataFrame, predictions: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """Build per-bar strategy and benchmark returns from target weights.

        When ``backtest.horizon_aware`` is set (default), each signal is held for the
        full ``prediction.horizon``: the effective per-bar position is the rolling
        mean of target weights over the horizon window, i.e. ``horizon`` overlapping
        positions each holding ``1/horizon`` of capital. This makes the equity curve
        measure the same horizon the model was trained to predict, and naturally
        reduces churn/turnover. The rolling mean is causal (current + past weights).
        """
        close = market["close"]
        weights = predictions["target_weight"].reindex(predictions.index)

        if self.config.backtest.horizon_aware:
            horizon = max(1, self.config.prediction.horizon)
            weights = weights.rolling(horizon, min_periods=1).mean()

        # Return realised over the bar *following* each decision.
        next_ret = close.pct_change().shift(-1).reindex(predictions.index)

        cost_rate = self.config.risk.fee_rate + self.config.risk.slippage
        turnover = weights.diff().abs().fillna(weights.abs())
        strat = weights * next_ret - turnover * cost_rate
        strat = strat.dropna()

        benchmark = next_ret.reindex(strat.index).fillna(0.0)
        return strat, benchmark

    def run(
        self,
        market: pd.DataFrame,
        features: pd.DataFrame | None = None,
        store: PredictionStore | None = None,
        register_models: bool = False,
    ) -> BacktestResult:
        """Run the progressive backtest.

        Args:
            market: Cleaned OHLCV(+context) frame.
            features: Optional precomputed feature matrix; computed if ``None``.
            store: Optional prediction/outcome store to populate.
            register_models: Persist each retrained model to the registry.

        Returns:
            A :class:`BacktestResult` with metrics, equity curve and learning stats.
        """
        if features is None:
            features = FeaturePipeline(self.config).transform(market)

        engine = ProgressiveLearningEngine(self.config, register_models=register_models)
        learning = engine.run(market, features, store=store)
        predictions = learning.predictions
        if predictions.empty:
            raise RuntimeError("Progressive engine produced no predictions.")

        strat, benchmark = self._strategy_returns(market, predictions)
        factor, ppy = self._ann_factor()

        if self.config.backtest.use_vectorbt:
            self._maybe_vectorbt(strat)

        metrics = compute_metrics(strat, annualization=factor, periods_per_year=ppy)
        benchmark_metrics = compute_metrics(benchmark, annualization=factor, periods_per_year=ppy)
        equity = (1.0 + strat).cumprod() * self.config.risk.initial_capital

        improvement = summarize_learning_degradation(learning.step_history)
        curve = summarize_step_history(learning.step_history)
        n_rebalances = count_rebalances(
            predictions["target_weight"],
            horizon_aware=self.config.backtest.horizon_aware,
            horizon=self.config.prediction.horizon,
        )
        metrics["n_rebalances"] = float(n_rebalances)

        logger.info(
            "Backtest complete: Sharpe=%.2f total_return=%.2f%% max_dd=%.2f%% "
            "rebalances=%d predictions=%d",
            metrics["sharpe"],
            metrics["total_return"] * 100,
            metrics["max_drawdown"] * 100,
            n_rebalances,
            len(predictions),
        )
        return BacktestResult(
            metrics=metrics,
            benchmark_metrics=benchmark_metrics,
            equity_curve=equity,
            strategy_returns=strat,
            learning=learning,
            learning_improvement=improvement,
            learning_curve=curve,
        )

    def _maybe_vectorbt(self, strat: pd.Series) -> None:
        """Cross-check via vectorbt if installed (optional, best-effort)."""
        try:
            import vectorbt as vbt  # noqa: PLC0415 - optional dependency
        except ImportError:
            logger.info("vectorbt not installed; using native metrics only.")
            return
        try:
            pf = vbt.Portfolio.from_returns(strat)
            logger.info("vectorbt cross-check Sharpe=%.3f", float(pf.sharpe_ratio()))
        except Exception as exc:  # noqa: BLE001 - cross-check must never break the run
            logger.info("vectorbt cross-check skipped: %s", exc)
