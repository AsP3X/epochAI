"""Near-real-time bar loop shared by paper-trade, live replay, and WebSocket modes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.paper_trader import PaperTrader
from epoch_ai.execution.portfolio_state import PortfolioState
from epoch_ai.execution.risk import RiskManager
from epoch_ai.execution.safety import SafetyScorer
from epoch_ai.features.pipeline import FeaturePipeline, build_target, forward_return
from epoch_ai.learning.retrain_job import run_retrain
from epoch_ai.logging_system.schemas import OutcomeLog, PredictionLog
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.models.base import BaseModel
from epoch_ai.models.factory import build_model
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
class _PendingPrediction:
    prediction_id: int
    entry_pos: int
    entry_price: float
    raw_prediction: float


@dataclass(slots=True)
class _LiveContext:
    config: AppConfig
    feature_cols: list[str]
    risk_manager: RiskManager
    safety_scorer: SafetyScorer | None
    trader: PaperTrader
    portfolio: PortfolioState
    model: BaseModel
    pipeline: FeaturePipeline
    market: pd.DataFrame
    data: pd.DataFrame
    retrain_every: int
    store: PredictionStore | None = None
    model_version: str = "unknown"
    pending: list[_PendingPrediction] | None = None
    bars_since_retrain: int = 0
    retrain_count: int = 0


def run_bar_loop(
    config: AppConfig,
    market: pd.DataFrame,
    *,
    start_pos: int,
    end_pos: int | None = None,
    retrain_every: int = 0,
    model: BaseModel | None = None,
    store: PredictionStore | None = None,
    model_version: str = "unknown",
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
        model = build_model(config.model, task=config.prediction.task)
        model.fit(data[feature_cols].iloc[:train_end], data["target"].iloc[:train_end])
    else:
        logger.info("Using pre-loaded registry model for runtime session.")

    ctx = _LiveContext(
        config=config,
        feature_cols=feature_cols,
        risk_manager=RiskManager(config.risk, config.prediction, config.safety),
        safety_scorer=SafetyScorer(config.safety) if config.safety.enabled else None,
        trader=PaperTrader(config.risk),
        portfolio=PortfolioState.initial(config.risk.initial_capital),
        model=model,
        pipeline=pipeline,
        market=market,
        data=data,
        retrain_every=max(0, retrain_every),
        store=store,
        model_version=model_version,
        pending=[] if store is not None else None,
    )

    last = len(data) if end_pos is None else min(end_pos, len(data))
    close = market["close"]
    symbol = config.primary_symbol

    for pos in range(start_pos, last):
        _resolve_pending_outcomes(ctx, pos, close)
        ts = data.index[pos]
        price = float(close.loc[ts])
        raw_pred = float(ctx.model.predict(data[feature_cols].iloc[[pos]])[0])
        feat_row = data[feature_cols].iloc[pos]
        safety = ctx.safety_scorer.assess(feat_row) if ctx.safety_scorer else None
        decision = ctx.risk_manager.decide(raw_pred, ctx.portfolio, safety=safety)
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
        if ctx.store is not None and ctx.pending is not None:
            feature_row = {
                k: float(v) for k, v in data[feature_cols].iloc[pos].to_dict().items()
            }
            pred_id = ctx.store.log_prediction(
                PredictionLog(
                    timestamp=str(ts),
                    symbol=symbol,
                    model_version=ctx.model_version,
                    horizon=config.prediction.horizon,
                    prediction=raw_pred,
                    confidence=decision.confidence,
                    signal=decision.signal,
                    entry_price=price,
                    features=feature_row,
                )
            )
            ctx.pending.append(
                _PendingPrediction(
                    prediction_id=pred_id,
                    entry_pos=pos,
                    entry_price=price,
                    raw_prediction=raw_pred,
                )
            )
        ctx.bars_since_retrain += 1
        if ctx.retrain_every and ctx.bars_since_retrain >= ctx.retrain_every:
            _maybe_retrain(ctx, pos)
            ctx.bars_since_retrain = 0

    if ctx.pending:
        _resolve_pending_outcomes(ctx, last - 1, close)

    if store is not None:
        store.close()

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
    ctx.model = build_model(ctx.config.model, task=ctx.config.prediction.task)
    ctx.model.fit(x_train, y_train)
    ctx.retrain_count += 1
    logger.info("Inline retrain #%d on %d rows.", ctx.retrain_count, len(x_train))


def _resolve_pending_outcomes(ctx: _LiveContext, current_pos: int, close: pd.Series) -> None:
    """Log realised outcomes once the prediction horizon has elapsed."""
    if not ctx.pending or ctx.store is None:
        return
    horizon = ctx.config.prediction.horizon
    threshold = ctx.config.prediction.threshold
    still_pending: list[_PendingPrediction] = []

    for pending in ctx.pending:
        if current_pos - pending.entry_pos < horizon:
            still_pending.append(pending)
            continue
        resolve_pos = pending.entry_pos + horizon
        resolve_ts = ctx.data.index[resolve_pos]
        exit_price = float(close.loc[resolve_ts])
        forward_ret = exit_price / pending.entry_price - 1.0
        ctx.store.log_outcome(
            OutcomeLog(
                prediction_id=pending.prediction_id,
                resolve_timestamp=str(resolve_ts),
                forward_return=forward_ret,
                realized_label=int(forward_ret > threshold),
                exit_price=exit_price,
                context={"runtime_session": True},
            )
        )

    ctx.pending = still_pending


def run_scheduled_retrain(config: AppConfig, *, min_new_samples: int = 50) -> int:
    """Run the SQLite/parquet retrain job; return 0 on success, 1 when skipped/failed."""
    result = run_retrain(config, min_new_samples=min_new_samples)
    if result.skipped:
        logger.warning("Retrain skipped: %s", result.reason)
        return 1
    return 0
