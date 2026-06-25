"""Tests for treasury profit allocation."""

from __future__ import annotations

from epoch_ai.execution.treasury import Treasury


def test_treasury_reserve_fraction(tmp_path):
    treasury = Treasury(
        trading_capital=10_000.0,
        reserve_fraction=0.25,
        state_path=str(tmp_path / "treasury.json"),
    )
    snap = treasury.allocate_session_pnl(1_000.0)
    assert snap.last_reserved == 250.0
    assert snap.last_reinvested == 750.0
    assert treasury.trading_capital == 10_750.0
    assert treasury.reserved_wins == 250.0


def test_treasury_loss_reinvests(tmp_path):
    treasury = Treasury(
        trading_capital=10_000.0,
        reserve_fraction=0.5,
        state_path=str(tmp_path / "treasury.json"),
    )
    snap = treasury.allocate_session_pnl(-500.0)
    assert treasury.trading_capital == 9_500.0
    assert treasury.reserved_wins == 0.0
    assert snap.last_session_pnl == -500.0


def test_treasury_persist_reload(tmp_path):
    path = str(tmp_path / "treasury.json")
    t1 = Treasury(trading_capital=5_000.0, reserve_fraction=0.2, state_path=path)
    t1.allocate_session_pnl(200.0)
    t2 = Treasury.load_or_create(initial_capital=5_000.0, reserve_fraction=0.2, state_path=path)
    assert t2.trading_capital == t1.trading_capital
    assert t2.reserved_wins == t1.reserved_wins
