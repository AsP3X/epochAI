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


def test_from_forecasts_uses_real_p_up_and_is_causal():
    # Agent: from_forecasts must surface the injected per-bar p_up (not the price proxy)
    #        and align returns causally (forecast at bar i earns the i->i+1 return).
    config = _policy_config()
    close = pd.Series(np.linspace(100.0, 120.0, 50))
    horizons = list(config.prediction.horizons)
    n = len(close)
    structured = {
        h: {
            "p_up": np.full(n, 0.9, dtype=float),
            "q10": np.full(n, -0.01, dtype=float),
            "q50": np.zeros(n, dtype=float),
            "q90": np.full(n, 0.01, dtype=float),
        }
        for h in horizons
    }
    env = TradingReplayEnv.from_forecasts(config, close, structured, horizons)
    env.reset()

    # Injected p_up flows through to the forecast emitted at the current bar.
    fc = env.current_forecast()
    assert fc.model_version == "replay-real"
    assert all(abs(h.p_up - 0.9) < 1e-6 for h in fc.horizons)

    # Causal return alignment: returns[i] == close[i+1]/close[i] - 1; last row padded to 0.
    expected = close.pct_change().shift(-1).fillna(0.0).to_numpy()
    assert np.allclose(env.returns, expected, atol=1e-6)
    assert env.returns[-1] == 0.0


def test_from_forecasts_baseline_trades_on_confident_signal():
    # Human: this is the regression guard for the holdout zero-trade bug -- a confident
    #        real forecast on a rising market must produce a non-flat, profitable baseline.
    # Agent: baseline_weight over from_forecasts env -> long, positive total_return.
    from epoch_ai.execution.policy.executor import baseline_weight
    from epoch_ai.learning.policy_promotion import replay_metrics

    config = _policy_config(trading={"reliability_floor": 0.1, "max_position_fraction": 0.5})
    close = pd.Series(np.linspace(100.0, 130.0, 60))
    horizons = list(config.prediction.horizons)
    n = len(close)
    # Decisive p_up + tight bands -> confidence clears the reliability floor.
    structured = {
        h: {
            "p_up": np.full(n, 0.95, dtype=float),
            "q10": np.full(n, -0.001, dtype=float),
            "q50": np.full(n, 0.005, dtype=float),
            "q90": np.full(n, 0.01, dtype=float),
        }
        for h in horizons
    }

    def baseline_fn(env: TradingReplayEnv) -> float:
        return baseline_weight(config, env.current_forecast(), env.portfolio)

    metrics = replay_metrics(
        TradingReplayEnv.from_forecasts(config, close, structured, horizons),
        baseline_fn,
    )
    assert metrics.total_return > 0.0


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
