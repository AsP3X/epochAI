"""Hard caps on any policy action (learned or baseline)."""

from __future__ import annotations

from epoch_ai.config.settings import RiskConfig, TradingConfig
from epoch_ai.execution.portfolio_state import PortfolioState


def apply_guardrails(
    target_weight: float,
    portfolio: PortfolioState,
    trading: TradingConfig,
    risk: RiskConfig,
) -> float:
    """Clamp or flatten an action when caps or kill-switch conditions trigger."""
    if portfolio.drawdown() >= trading.max_drawdown_kill:
        return 0.0
    if (
        risk.max_drawdown_halt is not None
        and portfolio.drawdown() >= risk.max_drawdown_halt
    ):
        return 0.0
    if (
        risk.max_daily_loss is not None
        and portfolio.session_loss() >= risk.max_daily_loss
    ):
        return 0.0
    if portfolio.cooldown_remaining > 0:
        return 0.0
    if (
        abs(portfolio.position_weight) > 1e-9
        and portfolio.bars_in_position >= trading.max_hold_bars
    ):
        return 0.0

    cap = trading.max_position_fraction * risk.max_leverage
    weight = float(max(-cap, min(cap, target_weight)))
    if not risk.allow_short and weight < 0:
        weight = 0.0
    return weight
