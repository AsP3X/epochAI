"""Tests for trading replay env and guardrails."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epoch_ai.config.settings import AppConfig, RiskConfig, TradingConfig
from epoch_ai.execution.policy.env import TradingReplayEnv
from epoch_ai.execution.policy.guardrails import apply_guardrails
from epoch_ai.execution.policy.observation import build_observation, observation_dim
from epoch_ai.execution.portfolio_state import PortfolioState
from epoch_ai.services.types import MultiHorizonPredictionResult, build_horizon_forecast


def _policy_config(**overrides) -> AppConfig:
    base = {
        "symbols": ["BTC/USDT"],
        "timeframe": "15m",
        "prediction": {"horizon": 8, "horizons": [8, 16]},
        "walk_forward": {"initial_train_period": 100},
        "trading": {
            "max_position_fraction": 0.5,
            "max_drawdown_kill": 0.20,
            "max_hold_bars": 10,
            "reliability_floor": 0.0,
        },
        "risk": {"max_leverage": 2.0, "initial_capital": 10_000.0},
    }
    base.update(overrides)
    return AppConfig.model_validate(base)


def test_observation_dim_matches_horizons():
    config = _policy_config()
    assert observation_dim(config) == len(config.prediction.horizons) * 3 + 4


def test_guardrails_clamp_position_and_drawdown_kill():
    trading = TradingConfig(max_position_fraction=0.5, max_drawdown_kill=0.10)
    risk = RiskConfig(max_leverage=2.0)
    portfolio = PortfolioState.initial(10_000.0)
    portfolio.equity = 8_500.0
    portfolio.peak_equity = 10_000.0

    assert apply_guardrails(1.5, portfolio, trading, risk) == 0.0

    portfolio.equity = 10_000.0
    portfolio.peak_equity = 10_000.0
    clamped = apply_guardrails(2.0, portfolio, trading, risk)
    assert abs(clamped) <= trading.max_position_fraction * risk.max_leverage + 1e-9


def test_env_step_respects_caps():
    config = _policy_config()
    close = pd.Series(np.linspace(100.0, 110.0, 200))
    market = pd.DataFrame({"close": close})
    env = TradingReplayEnv.from_market(config, market)
    env.reset()

    cap = config.trading.max_position_fraction * config.risk.max_leverage
    for _ in range(20):
        _, _, done, info = env.step(cap * 2.0)
        assert abs(info["weight"]) <= cap + 1e-9
        if done:
            break


def test_build_observation_uses_reliable_heads_only():
    config = _policy_config()
    portfolio = PortfolioState.initial(10_000.0)
    forecasts = [
        build_horizon_forecast(
            as_of=pd.Timestamp("2020-01-01"),
            last_close=100.0,
            horizon=8,
            horizon_label="8b",
            bar_minutes=15,
            p_up=0.7,
            q10=-0.01,
            q50=0.02,
            q90=0.03,
            reliability_floor=0.0,
        ),
        build_horizon_forecast(
            as_of=pd.Timestamp("2020-01-01"),
            last_close=100.0,
            horizon=16,
            horizon_label="16b",
            bar_minutes=15,
            p_up=0.4,
            q10=-0.02,
            q50=-0.01,
            q90=0.01,
            reliability_floor=0.99,
        ),
    ]
    multi = MultiHorizonPredictionResult(
        as_of="2020-01-01",
        last_close=100.0,
        model_version="test",
        symbol="BTC/USDT",
        timeframe="15m",
        horizons=forecasts,
    )
    obs = build_observation(multi, portfolio, config)
    assert obs.shape == (observation_dim(config),)
    assert obs[0] == 0.7
    assert obs[3] == 0.5
