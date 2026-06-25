"""Tests for paper trader execution."""

from __future__ import annotations

from epoch_ai.config.settings import RiskConfig
from epoch_ai.execution.paper_trader import PaperTrader
from epoch_ai.execution.risk import RiskDecision


def test_rebalance_charges_fees():
    trader = PaperTrader(RiskConfig(initial_capital=10_000.0))
    decision = RiskDecision(signal=1, confidence=0.8, target_weight=0.5)
    fill = trader.rebalance("2020-01-01", 100.0, decision)
    assert fill is not None
    assert trader.equity < 10_000.0
    assert trader.position_weight == 0.5


def test_no_trade_when_flat():
    trader = PaperTrader(RiskConfig())
    decision = RiskDecision(signal=0, confidence=0.0, target_weight=0.0)
    assert trader.rebalance("2020-01-01", 100.0, decision) is None
    assert trader.equity == trader.risk.initial_capital
