"""Near-real-time bar loop shared by paper-trade, live replay, and WebSocket modes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.paper_trader import PaperTrader
from epoch_ai.execution.portfolio_state import PortfolioState
from epoch_ai.execution.risk import RiskManager
from epoch_ai.features.pipeline import FeaturePipeline, build_target, forward_return
from epoch_ai.learning.retrain_job import run_retrain
from epoch_ai.models.lightgbm_model import LightGBMModel
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class LiveLoopResult:
    """Summary of a bar-driven live/replay session."""

    bars_processed: int
    fills: int
    final_equity: float
    retrain_count: int = 0


@dataclass(slots=True)
class _LiveContext:
    config: AppConfig
    feature_cols: list[str]
    risk_manager: RiskManager
    trader: PaperTrader
    portfolio: PortfolioState
    model: LightGBMModel
    pipeline: FeaturePipeline
    market: pd.DataFrame
    data: pd.DataFrame
    retrain_every: int
    bars_since_retrain: int = 0
    retrain_count: int = 0


def run_bar_loop(
    config: AppConfig,
    market: pd.DataFrame,
    *,
    start_pos: int,
    end_pos: int | None = None,
    retrain_every: int = 0,
    model: LightGBMModel | None = None,
) -> LiveLoopResult:
    """Step bar-by-bar through ``[start_pos, end_pos)`` with predict → risk → paper trade.

    Args:
        config: Application configuration.
        market: Full OHLCV frame (used for feature recomputation on each bar).
        start_pos: First positional index in the aligned supervised frame to trade.
        end_pos: Exclusive end index; ``None`` means through the last row.
        retrain_every: Retrain every N processed bars (0 = never).

    Returns:
        Session summary including fill count and final equity.
    """
    pipeline = FeaturePipeline(config)
    features = pipeline.transform(market)
    y = build_target(market, config.prediction)
    fwd = forward_return(market, config.prediction.horizon)
    data = features.join(y).join(fwd).dropna(subset=["target", "forward_return"])
    feature_cols = list(features.columns)

    if start_pos >= len(data):
        raise ValueError(f"start_pos {start_pos} out of range for {len(data)} rows.")

    train_end = start_pos
    if model is None:
        model = LightGBMModel(config.model, task=config.prediction.task)
        model.fit(data[feature_cols].iloc[:train_end], data["target"].iloc[:train_end])
    else:
        logger.info("Using pre-loaded registry model for runtime session.")

    ctx = _LiveContext(
        config=config,
        feature_cols=feature_cols,
        risk_manager=RiskManager(config.risk, config.prediction),
        trader=PaperTrader(config.risk),
        portfolio=PortfolioState.initial(config.risk.initial_capital),
        model=model,
        pipeline=pipeline,
        market=market,
        data=data,
        retrain_every=max(0, retrain_every),
    )

    last = len(data) if end_pos is None else min(end_pos, len(data))
    close = market["close"]

    for pos in range(start_pos, last):
        ts = data.index[pos]
        price = float(close.loc[ts])
        raw_pred = float(ctx.model.predict(data[feature_cols].iloc[[pos]])[0])
        decision = ctx.risk_manager.decide(raw_pred, ctx.portfolio)
        prev_equity = ctx.trader.equity
        ctx.trader.rebalance(str(ts), price, decision)
        period_ret = float(data["forward_return"].iloc[pos]) / config.prediction.horizon
        ctx.trader.mark_to_market(period_ret)
        lost = ctx.trader.equity < prev_equity and ctx.trader.position_weight != 0
        ctx.portfolio.after_bar(
            ctx.trader.equity,
            lost_trade=lost,
            cooldown_bars=config.risk.cooldown_bars,
        )
        ctx.bars_since_retrain += 1
        if ctx.retrain_every and ctx.bars_since_retrain >= ctx.retrain_every:
            _maybe_retrain(ctx, pos)
            ctx.bars_since_retrain = 0

    return LiveLoopResult(
        bars_processed=last - start_pos,
        fills=len(ctx.trader.fills),
        final_equity=ctx.trader.equity,
        retrain_count=ctx.retrain_count,
    )


def _maybe_retrain(ctx: _LiveContext, pos: int) -> None:
    """Refit on all data up to ``pos`` (causal expanding window)."""
    x_train = ctx.data[ctx.feature_cols].iloc[:pos]
    y_train = ctx.data["target"].iloc[:pos]
    if len(x_train) < ctx.config.walk_forward.initial_train_period:
        logger.info("Skipping inline retrain: only %d rows available.", len(x_train))
        return
    ctx.model = LightGBMModel(ctx.config.model, task=ctx.config.prediction.task)
    ctx.model.fit(x_train, y_train)
    ctx.retrain_count += 1
    logger.info("Inline retrain #%d on %d rows.", ctx.retrain_count, len(x_train))


def run_scheduled_retrain(config: AppConfig, *, min_new_samples: int = 50) -> int:
    """Run the SQLite/parquet retrain job; return 0 on success, 1 when skipped/failed."""
    result = run_retrain(config, min_new_samples=min_new_samples)
    if result.skipped:
        logger.warning("Retrain skipped: %s", result.reason)
        return 1
    return 0
