"""Tests for risk manager halts and sizing."""

from __future__ import annotations

from epoch_ai.config.settings import PredictionConfig, RiskConfig, TradingConfig
from epoch_ai.execution.policy.guardrails import apply_guardrails
from epoch_ai.execution.portfolio_state import PortfolioState
from epoch_ai.execution.risk import RiskManager


def test_risk_per_trade_scales_weight():
    risk = RiskConfig(risk_per_trade=0.02, max_leverage=3.0, long_threshold=0.5)
    mgr = RiskManager(risk, PredictionConfig(task="classification"))
    decision = mgr.decide(0.9)
    assert decision.signal == 1
    assert 0 < abs(decision.target_weight) <= risk.max_leverage


def test_min_confidence_flattens():
    risk = RiskConfig(min_confidence=0.8, long_threshold=0.5)
    mgr = RiskManager(risk, PredictionConfig(task="classification"))
    decision = mgr.decide(0.52)
    assert decision.signal == 0
    assert decision.target_weight == 0.0


def test_drawdown_halt():
    risk = RiskConfig(max_drawdown_halt=0.1, long_threshold=0.5)
    mgr = RiskManager(risk, PredictionConfig(task="classification"))
    portfolio = PortfolioState.initial(10_000.0)
    portfolio.equity = 8500.0
    portfolio.peak_equity = 10_000.0
    decision = mgr.decide(0.9, portfolio)
    assert decision.halted
    assert decision.signal == 0


def test_cooldown_halt():
    risk = RiskConfig(long_threshold=0.5, cooldown_bars=2)
    mgr = RiskManager(risk, PredictionConfig(task="classification"))
    portfolio = PortfolioState.initial(10_000.0)
    portfolio.cooldown_remaining = 1
    decision = mgr.decide(0.9, portfolio)
    assert decision.halted


def test_guardrails_max_hold_forces_flat():
    trading = TradingConfig(max_hold_bars=5, max_position_fraction=1.0)
    risk = RiskConfig(max_leverage=1.0)
    portfolio = PortfolioState.initial(10_000.0)
    portfolio.position_weight = 0.5
    portfolio.bars_in_position = 5
    weight = apply_guardrails(0.5, portfolio, trading, risk)
    assert weight == 0.0
