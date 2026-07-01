"""Tests for the multi-bar decision-cadence reward in TradingReplayEnv.step."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.policy.env import TradingReplayEnv
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
