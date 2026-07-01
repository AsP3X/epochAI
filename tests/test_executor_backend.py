"""Backend-selection tests for :func:`decide_trading_action`.

Covers the ``learned_with_baseline_fallback`` contract (learned drives the
decision when a policy exists, baseline otherwise) and proves the ``learned``
path has no reliability/threshold dead-band forcing flat.
"""

from __future__ import annotations

import pandas as pd

from epoch_ai.execution.policy.executor import decide_trading_action
from epoch_ai.execution.portfolio_state import PortfolioState
from epoch_ai.services.types import MultiHorizonPredictionResult, build_horizon_forecast
from tests.test_policy_env import _policy_config


class _StubPolicy:
    """Minimal PPO stand-in so tests avoid importing torch."""

    def __init__(self, action):
        self._a = action

    def act(self, obs, *, deterministic=False):
        return self._a


def _multi(config, *, p_up: float, reliability_floor: float = 0.0) -> MultiHorizonPredictionResult:
    # Agent: build one forecast per prediction horizon with tunable p_up + reliability
    #        gate; tight quantile bands keep confidence high for reliable cases.
    forecasts = [
        build_horizon_forecast(
            as_of=pd.Timestamp("2020-01-01"),
            last_close=100.0,
            horizon=h,
            horizon_label=f"{h}b",
            bar_minutes=15,
            p_up=p_up,
            q10=-0.001,
            q50=0.005,
            q90=0.01,
            reliability_floor=reliability_floor,
        )
        for h in config.prediction.horizons
    ]
    return MultiHorizonPredictionResult(
        as_of="2020-01-01",
        last_close=100.0,
        model_version="test",
        symbol="BTC/USDT",
        timeframe="15m",
        horizons=forecasts,
    )


def test_learned_fallback_uses_policy_when_present():
    # Human: with a policy artifact present the learned action must drive the
    #        decision (full long -> sized to the cap), not the baseline.
    config = _policy_config(trading={"policy_backend": "learned_with_baseline_fallback"})
    portfolio = PortfolioState.initial(10_000.0)
    multi = _multi(config, p_up=0.9)

    decision = decide_trading_action(
        config,
        raw_prediction=0.9,
        multi=multi,
        portfolio=portfolio,
        ppo=_StubPolicy(1.0),
    )

    cap = config.trading.max_position_fraction * config.risk.max_leverage
    assert decision.signal == 1
    assert decision.target_weight > 0
    assert abs(decision.target_weight - cap) < 1e-9


def test_learned_fallback_falls_back_to_baseline_when_no_policy(tmp_path):
    # Human: no policy + no artifact -> the decision must match the baseline path
    #        (confident forecast long; neutral forecast flat).
    config = _policy_config(
        trading={"policy_backend": "learned_with_baseline_fallback"},
        rl={"policy_path": str(tmp_path / "missing_policy.pt")},
    )
    portfolio = PortfolioState.initial(10_000.0)

    confident = decide_trading_action(
        config,
        raw_prediction=0.9,
        multi=_multi(config, p_up=0.9),
        portfolio=portfolio,
        ppo=None,
    )
    assert confident.signal == 1
    assert confident.target_weight > 0

    flat = decide_trading_action(
        config,
        raw_prediction=0.5,
        multi=_multi(config, p_up=0.5),
        portfolio=PortfolioState.initial(10_000.0),
        ppo=None,
    )
    assert flat.signal == 0
    assert abs(flat.target_weight) < 1e-9


def test_learned_path_has_no_reliability_deadband():
    # Human: even when every head is UNRELIABLE the learned policy still decides;
    #        there is no threshold dead-band forcing flat on the learned branch.
    config = _policy_config(trading={"policy_backend": "learned"})
    portfolio = PortfolioState.initial(10_000.0)
    # Impossibly high floor -> all heads unreliable (confidence below floor).
    multi = _multi(config, p_up=0.51, reliability_floor=0.99)
    assert all(not h.reliable for h in multi.horizons)

    decision = decide_trading_action(
        config,
        raw_prediction=0.51,
        multi=multi,
        portfolio=portfolio,
        ppo=_StubPolicy(1.0),
    )

    assert decision.signal == 1
    assert decision.target_weight > 0
