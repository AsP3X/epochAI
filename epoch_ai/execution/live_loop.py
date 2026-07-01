"""Near-real-time bar loop shared by paper-trade, live replay, and WebSocket modes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.action_log import ActionLog
from epoch_ai.execution.paper_trader import PaperTrader
from epoch_ai.execution.policy.executor import decide_trading_action, load_ppo_policy
from epoch_ai.execution.policy.ppo_policy import PPOPolicy
from epoch_ai.execution.policy.trunk_policy import runtime_trunk_embedding
from epoch_ai.execution.portfolio_state import PortfolioState
from epoch_ai.execution.risk import RiskManager
from epoch_ai.execution.safety import SafetyScorer
from epoch_ai.execution.session_state import SessionState
from epoch_ai.features.pipeline import FeaturePipeline, build_target, forward_return
from epoch_ai.learning.retrain_job import run_retrain
from epoch_ai.logging_system.multi_horizon_log import (
    PendingHorizonLog,
    log_multi_horizon_bar,
    resolve_pending_horizons,
)
from epoch_ai.logging_system.schemas import PredictionLog
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.models.base import BaseModel, MultiHeadModel
from epoch_ai.models.factory import build_model
from epoch_ai.services.types import build_multi_horizon_from_structured
from epoch_ai.utils.logging import get_logger
from epoch_ai.utils.timeframe import timeframe_to_minutes

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
    pending: list[PendingHorizonLog] | None = None
    ppo: PPOPolicy | None = None
    action_log: ActionLog | None = None
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
        portfolio=_load_portfolio(config),
        model=model,
        pipeline=pipeline,
        market=market,
        data=data,
        retrain_every=max(0, retrain_every),
        store=store,
        model_version=model_version,
        pending=[] if store is not None else None,
        ppo=load_ppo_policy(config) if config.trading.policy_backend.startswith("learned") else None,
        action_log=ActionLog(config.trading.action_log_path),
    )

    last = len(data) if end_pos is None else min(end_pos, len(data))
    close = market["close"]
    symbol = config.primary_symbol

    for pos in range(start_pos, last):
        _resolve_pending_outcomes(ctx, pos, close)
        ts = data.index[pos]
        price = float(close.loc[ts])
        # Sequence backends (TCN) need a trailing lookback window; the target bar is the
        # last row of the slice, so we read predictions/structured arrays at ``soffset``.
        seq_lb = getattr(ctx.model, "sequence_lookback", None)
        if seq_lb:
            win_start = max(0, pos - int(seq_lb) + 1)
            model_in = data[feature_cols].iloc[win_start : pos + 1]
            soffset = len(model_in) - 1
        else:
            model_in = data[feature_cols].iloc[[pos]]
            soffset = 0
        raw_pred = float(ctx.model.predict(model_in)[-1])
        feat_row = data[feature_cols].iloc[pos]
        safety = ctx.safety_scorer.assess(feat_row) if ctx.safety_scorer else None
        multi = None
        if (
            config.trading.policy_backend != "threshold"
            and isinstance(ctx.model, MultiHeadModel)
            and ctx.model.multi_head_spec_ is not None
        ):
            structured = ctx.model.predict_structured(model_in)
            multi = build_multi_horizon_from_structured(
                structured,
                soffset,
                as_of=pd.Timestamp(ts),
                last_close=price,
                model_version=ctx.model_version,
                symbol=symbol,
                timeframe=config.timeframe,
                horizons=list(ctx.model.multi_head_spec_.horizons),
                horizon_label_fn=config.prediction.horizon_label,
                bar_minutes=timeframe_to_minutes(config.timeframe),
            )
        trunk_emb = None
        if ctx.ppo is not None:
            trunk_emb = runtime_trunk_embedding(config, ctx.model, model_in)
        decision = decide_trading_action(
            config,
            raw_prediction=raw_pred,
            multi=multi,
            portfolio=ctx.portfolio,
            ppo=ctx.ppo,
            safety=safety,
            trunk_embedding=trunk_emb,
        )
        prev_equity = ctx.trader.equity
        fill = ctx.trader.rebalance(str(ts), price, decision)
        period_ret = float(data["forward_return"].iloc[pos]) / config.prediction.horizon
        ctx.trader.mark_to_market(period_ret)
        lost = ctx.trader.equity < prev_equity and ctx.trader.position_weight != 0
        ctx.portfolio.after_bar(
            ctx.trader.equity,
            lost_trade=lost,
            cooldown_bars=config.risk.cooldown_bars,
            position_weight=ctx.trader.position_weight,
        )
        if ctx.action_log is not None:
            ctx.action_log.log_step(
                timestamp=str(ts),
                symbol=symbol,
                model_version=ctx.model_version,
                policy_backend=config.trading.policy_backend,
                raw_prediction=raw_pred,
                decision=decision,
                equity=ctx.trader.equity,
                position_weight=ctx.trader.position_weight,
                multi=multi,
                fill_fee=fill.fee if fill is not None else None,
                bar_return=period_ret,
            )
        SessionState.from_portfolio(ctx.portfolio).save(config.trading.session_state_path)
        if ctx.store is not None and ctx.pending is not None:
            feature_row = {
                k: float(v) for k, v in data[feature_cols].iloc[pos].to_dict().items()
            }
            if isinstance(ctx.model, MultiHeadModel) and ctx.model.multi_head_spec_ is not None:
                # Reuse the forecast computed for the policy decision when available;
                # only the threshold backend leaves ``multi`` unset here.
                if multi is None:
                    structured = ctx.model.predict_structured(model_in)
                    multi = build_multi_horizon_from_structured(
                        structured,
                        soffset,
                        as_of=pd.Timestamp(ts),
                        last_close=price,
                        model_version=ctx.model_version,
                        symbol=symbol,
                        timeframe=config.timeframe,
                        horizons=list(ctx.model.multi_head_spec_.horizons),
                        horizon_label_fn=config.prediction.horizon_label,
                        bar_minutes=timeframe_to_minutes(config.timeframe),
                    )
                ctx.pending.extend(
                    log_multi_horizon_bar(
                        ctx.store,
                        multi,
                        signal=decision.signal,
                        base_features=feature_row,
                        entry_price=price,
                        entry_index=pos,
                    )
                )
            else:
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
                    PendingHorizonLog(
                        prediction_id=pred_id,
                        entry_index=pos,
                        entry_price=price,
                        horizon=config.prediction.horizon,
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


def _load_portfolio(config: AppConfig) -> PortfolioState:
    saved = SessionState.load(config.trading.session_state_path)
    if saved is not None:
        return saved.to_portfolio()
    return PortfolioState.initial(config.risk.initial_capital)


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
    """Log realised outcomes once each prediction horizon has elapsed."""
    if not ctx.pending or ctx.store is None:
        return
    ctx.pending = resolve_pending_horizons(
        ctx.pending,
        current_index=current_pos,
        close=close,
        index=ctx.data.index,
        threshold=ctx.config.prediction.threshold,
        store=ctx.store,
        context={"runtime_session": True},
    )


def run_scheduled_retrain(config: AppConfig, *, min_new_samples: int = 50) -> int:
    """Run the SQLite/parquet retrain job; return 0 on success, 1 when skipped/failed."""
    result = run_retrain(config, min_new_samples=min_new_samples)
    if result.skipped:
        logger.warning("Retrain skipped: %s", result.reason)
        return 1
    return 0
