"""Regression tests for the multi-timescale policy observation vector."""

from __future__ import annotations

import pandas as pd

from epoch_ai.execution.policy.observation import build_observation, observation_dim
from epoch_ai.execution.portfolio_state import PortfolioState
from epoch_ai.services.types import MultiHorizonPredictionResult, build_horizon_forecast
from tests.test_policy_env import _policy_config


def test_observation_is_multitimescale():
    # Human: the observation must simultaneously carry a FAST head (horizon 1) and a
    #        slower MULTI-bar head (horizon 12) so the policy can reason across
    #        timescales; unreliable heads are masked to a neutral (0.5, 0.0, 0.0) triple.
    # Agent: CONFIG trading.decision_horizons=[1,6,12]; build_observation emits reliable
    #        triples for reliable heads, neutral triple for unreliable, then 4 portfolio scalars.
    config = _policy_config(
        prediction={"horizon": 12, "horizons": [1, 6, 12]},
        trading={
            "max_position_fraction": 0.5,
            "max_drawdown_kill": 0.20,
            "max_hold_bars": 10,
            "reliability_floor": 0.1,
            "decision_horizons": [1, 6, 12],
        },
    )
    portfolio = PortfolioState.initial(10_000.0)

    fast = build_horizon_forecast(
        as_of=pd.Timestamp("2020-01-01"),
        last_close=100.0,
        horizon=1,
        horizon_label="1b",
        bar_minutes=15,
        p_up=0.8,
        q10=-0.001,
        q50=0.004,
        q90=0.006,
        reliability_floor=0.0,
    )
    # Unreliable middle head: build with an impossibly high floor so reliable is False.
    unreliable = build_horizon_forecast(
        as_of=pd.Timestamp("2020-01-01"),
        last_close=100.0,
        horizon=6,
        horizon_label="6b",
        bar_minutes=15,
        p_up=0.51,
        q10=-0.05,
        q50=0.0,
        q90=0.05,
        reliability_floor=0.99,
    )
    multi_head = build_horizon_forecast(
        as_of=pd.Timestamp("2020-01-01"),
        last_close=100.0,
        horizon=12,
        horizon_label="12b",
        bar_minutes=15,
        p_up=0.75,
        q10=-0.002,
        q50=0.01,
        q90=0.008,
        reliability_floor=0.0,
    )
    assert fast.reliable is True
    assert multi_head.reliable is True
    assert unreliable.reliable is False

    multi = MultiHorizonPredictionResult(
        as_of="2020-01-01",
        last_close=100.0,
        model_version="test",
        symbol="BTC/USDT",
        timeframe="15m",
        horizons=[fast, unreliable, multi_head],
    )
    obs = build_observation(multi, portfolio, config)

    # Length matches the declared observation size (3 per head + 4 portfolio scalars).
    assert obs.shape == (observation_dim(config),)
    assert observation_dim(config) == 3 * 3 + 4

    # FAST head (index 0): reliable triple surfaces the real forecast values.
    assert obs[0] == fast.p_up
    assert obs[1] == fast.exp_return
    assert obs[2] == fast.confidence

    # Middle head (index 1): unreliable -> masked to the neutral triple.
    assert list(obs[3:6]) == [0.5, 0.0, 0.0]

    # MULTI-bar head (index 2): reliable triple surfaces the real forecast values.
    assert obs[6] == multi_head.p_up
    assert obs[7] == multi_head.exp_return
    assert obs[8] == multi_head.confidence

    # Exactly 4 portfolio scalars close out the vector.
    assert len(obs[9:]) == 4
