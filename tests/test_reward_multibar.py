"""Tests for the multi-bar decision-cadence reward in TradingReplayEnv.step."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.policy.env import TradingReplayEnv
from epoch_ai.execution.policy.guardrails import apply_guardrails
from epoch_ai.learning.policy_promotion import replay_metrics
from tests.test_policy_env import _policy_config


def _rising_market(n: int = 50) -> pd.DataFrame:
    # Human: strictly-rising close => every pct_change return is > 0, so a full long
    #        held across the block must accumulate a positive block return.
    # Agent: deterministic; returns[0]==0 (first pct_change NaN->0), returns[1:]>0.
    close = pd.Series(np.linspace(100.0, 200.0, n))
    return pd.DataFrame({"close": close})


def _zero_cost_config(**rl_overrides) -> AppConfig:
    rl = {
        "reward_mode": "multi_bar",
        "reward_horizon": 5,
        "turnover_penalty": 0.0,
        "sharpe_scale": 1.0,
        "drawdown_penalty": 0.0,
    }
    rl.update(rl_overrides)
    return _policy_config(
        rl=rl,
        risk={
            "max_leverage": 2.0,
            "initial_capital": 10_000.0,
            "fee_rate": 0.0,
            "slippage": 0.0,
        },
        trading={
            "max_position_fraction": 0.5,
            "max_drawdown_kill": 0.20,
            "max_hold_bars": 10,
            "reliability_floor": 0.0,
            "funding_rate_per_bar": 0.0,
        },
    )


def test_multibar_holds_and_accumulates():
    config = _zero_cost_config(reward_horizon=5)
    env = TradingReplayEnv.from_market(config, _rising_market())
    env.reset()

    cap = config.trading.max_position_fraction * config.risk.max_leverage
    start_pos = env._pos
    _, reward, _, info = env.step(cap)

    # One decision was held for exactly reward_horizon bars.
    assert info["bars_held"] == 5
    assert env._pos - start_pos == 5

    # Hand-compute the accumulated block return (fees/funding = 0, full long).
    eq = config.risk.initial_capital
    for i in range(5):
        eq = eq + cap * float(env.returns[i]) * eq
    expected_block_ret = (eq - config.risk.initial_capital) / config.risk.initial_capital
    # drawdown_penalty=0, turnover_penalty=0, sharpe_scale=1 -> reward == block_ret.
    assert reward > 0.0
    assert np.isclose(reward, expected_block_ret, atol=1e-9)


def test_multibar_turnover_penalty_reduces_reward():
    cap = 0.5 * 2.0

    env_zero = TradingReplayEnv.from_market(
        _zero_cost_config(turnover_penalty=0.0), _rising_market()
    )
    env_zero.reset()
    _, reward_zero, _, _ = env_zero.step(cap)

    env_pen = TradingReplayEnv.from_market(
        _zero_cost_config(turnover_penalty=0.1), _rising_market()
    )
    env_pen.reset()
    _, reward_pen, _, _ = env_pen.step(cap)

    # Flipping flat -> full long incurs turnover_penalty * |cap - 0| > 0.
    assert reward_pen < reward_zero


def test_per_bar_mode_advances_one_bar():
    config = _zero_cost_config(reward_mode="per_bar")
    env = TradingReplayEnv.from_market(config, _rising_market())
    env.reset()

    cap = config.trading.max_position_fraction * config.risk.max_leverage
    start_pos = env._pos
    env.step(cap)
    assert env._pos - start_pos == 1


def _dip_market() -> pd.DataFrame:
    # Human: close RISES to 120 then DIPS to 108 (all inside one reward_horizon block)
    #        then recovers to 112, ending the block ABOVE the intra-block trough. A step
    #        that samples equity only at the block boundary misses the 120 -> 108 drop.
    # Agent: deterministic; intra-block peak-to-trough = (120-108)/120 = 0.10.
    close = pd.Series([100.0, 105.0, 110.0, 120.0, 108.0, 112.0, 113.0, 114.0])
    return pd.DataFrame({"close": close})


def _max_dd(curve: list[float]) -> float:
    arr = np.asarray(curve, dtype=float)
    peak = np.maximum.accumulate(arr)
    return float(((peak - arr) / peak).max())


def test_equity_path_reports_every_held_bar():
    # per_bar step exposes a 1-element equity_path; multi_bar exposes one entry per bar.
    per_bar = _zero_cost_config(reward_mode="per_bar")
    env = TradingReplayEnv.from_market(per_bar, _rising_market())
    env.reset()
    cap = per_bar.trading.max_position_fraction * per_bar.risk.max_leverage
    _, _, _, info = env.step(cap)
    assert info["equity_path"] == [info["equity"]]

    multi = _zero_cost_config(reward_horizon=6)
    env = TradingReplayEnv.from_market(multi, _dip_market())
    env.reset()
    _, _, _, info = env.step(cap)
    assert len(info["equity_path"]) == info["bars_held"]
    # Last per-bar equity equals the block-boundary equity.
    assert np.isclose(info["equity_path"][-1], info["equity"])


def test_replay_metrics_per_bar_drawdown_captures_intrablock_dip():
    config = _zero_cost_config(reward_horizon=6)
    market = _dip_market()
    cap = config.trading.max_position_fraction * config.risk.max_leverage

    def weight_fn(_env: TradingReplayEnv) -> float:
        return cap

    metrics = replay_metrics(TradingReplayEnv.from_market(config, market), weight_fn)

    # Reconstruct the per-bar curve (what replay_metrics builds) and the coarse
    # boundary-only curve (what the old block-boundary sampling would have seen).
    env = TradingReplayEnv.from_market(config, market)
    env.reset()
    start_eq = env.portfolio.equity
    per_bar_curve = [start_eq]
    boundary_curve = [start_eq]
    total_bars = 0
    while not env.done:
        w = apply_guardrails(
            weight_fn(env), env.portfolio, env.config.trading, env.config.risk
        )
        _, _, done, info = env.step(w)
        per_bar_curve.extend(info["equity_path"])
        boundary_curve.append(info["equity"])
        total_bars += info["bars_held"]
        if done:
            break

    boundary_dd = _max_dd(boundary_curve)
    per_bar_dd = _max_dd(per_bar_curve)

    # The honest per-bar drawdown must expose the intra-block dip the boundary misses.
    assert boundary_dd == 0.0
    assert per_bar_dd > boundary_dd
    assert np.isclose(metrics.max_drawdown, per_bar_dd, atol=1e-9)
    assert math.isfinite(metrics.sharpe)
    # Curve spans every consumed bar plus the starting equity.
    assert len(per_bar_curve) == total_bars + 1
