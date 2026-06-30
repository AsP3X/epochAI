"""Causal trading replay environment for PPO training on historical bars."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.policy.guardrails import apply_guardrails
from epoch_ai.execution.policy.observation import build_observation, observation_dim
from epoch_ai.execution.portfolio_state import PortfolioState
from epoch_ai.services.types import (
    MultiHorizonPredictionResult,
    build_horizon_forecast,
)
from epoch_ai.utils.timeframe import timeframe_to_minutes


@dataclass(slots=True)
class TradingReplayEnv:
    """Simple bar-replay env: obs from synthetic reliable forecasts + portfolio state."""

    config: AppConfig
    returns: np.ndarray
    p_up_series: np.ndarray
    obs_dim: int
    # Human: When set, current_forecast() emits the REAL model's per-bar/per-horizon
    #        forecast (p_up + return quantiles) instead of the price-only proxy. Used by
    #        the holdout acceptance benchmark so the policy is scored on the actual model.
    # Agent: dict horizon -> {"p_up","q10","q50","q90"} arrays aligned to ``returns`` rows;
    #        None => proxy mode (PPO training path). CAUSAL: see from_forecasts shift.
    structured_forecasts: dict[int, dict[str, np.ndarray]] | None = None
    _pos: int = 0
    portfolio: PortfolioState = field(init=False)
    _prev_equity: float = field(init=False)

    def __post_init__(self) -> None:
        self.portfolio = PortfolioState.initial(self.config.risk.initial_capital)
        self._prev_equity = self.portfolio.equity

    @classmethod
    def from_market(cls, config: AppConfig, market: pd.DataFrame) -> TradingReplayEnv:
        """Build env from OHLCV close returns and a rolling P(up) proxy."""
        close = market["close"].astype(float)
        rets = close.pct_change().fillna(0.0).to_numpy(dtype=np.float32)
        roll = close.pct_change(5).fillna(0.0)
        p_up = (0.5 + np.tanh(roll.to_numpy(dtype=np.float32) * 20.0) * 0.25).astype(
            np.float32
        )
        return cls(
            config=config,
            returns=rets,
            p_up_series=p_up,
            obs_dim=observation_dim(config),
        )

    @classmethod
    def from_forecasts(
        cls,
        config: AppConfig,
        close: pd.Series,
        structured: dict[int, dict[str, np.ndarray]],
        horizons: list[int],
    ) -> TradingReplayEnv:
        """Build env from a close series + the real model's per-bar structured forecasts.

        Unlike :meth:`from_market`, this scores the policy on the actual trained model's
        ``p_up``/quantiles. The return series is shifted so the position sized from the
        causal forecast at bar ``i`` earns the bar ``i -> i+1`` return (no look-ahead).

        Args:
            close: Close prices aligned 1:1 with the rows of ``structured`` (prediction rows).
            structured: ``predict_structured`` output: horizon -> {"p_up","q10","q50","q90"}.
            horizons: Horizons to expose to the policy (order preserved).
        """
        close = close.astype(float)
        # Human: forecast at row i is causal (features up to i); it must earn the return
        #        realized from bar i to i+1, so we shift pct_change() forward by one.
        # Agent: CAUSAL; returns[i] = (close[i+1]/close[i]-1); last row -> 0 (no next bar).
        rets = close.pct_change().shift(-1).fillna(0.0).to_numpy(dtype=np.float32)
        forecasts = {
            int(h): {
                "p_up": np.asarray(structured[h]["p_up"], dtype=np.float32).reshape(-1),
                "q10": np.asarray(structured[h]["q10"], dtype=np.float32).reshape(-1),
                "q50": np.asarray(structured[h]["q50"], dtype=np.float32).reshape(-1),
                "q90": np.asarray(structured[h]["q90"], dtype=np.float32).reshape(-1),
            }
            for h in horizons
        }
        # Agent: p_up_series feeds the proxy obs only; expose primary horizon for parity.
        primary = config.prediction.horizon
        primary = primary if primary in forecasts else next(iter(forecasts))
        return cls(
            config=config,
            returns=rets,
            p_up_series=forecasts[primary]["p_up"],
            obs_dim=observation_dim(config),
            structured_forecasts=forecasts,
        )

    @property
    def done(self) -> bool:
        return self._pos >= len(self.returns) - 1

    def reset(self) -> np.ndarray:
        self._pos = 0
        self.portfolio = PortfolioState.initial(self.config.risk.initial_capital)
        self._prev_equity = self.portfolio.equity
        return self._obs()

    def current_forecast(self) -> MultiHorizonPredictionResult:
        ts = pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=self._pos)
        bar_minutes = timeframe_to_minutes(self.config.timeframe)
        if self.structured_forecasts is not None:
            # Agent: real-model path; emit per-horizon p_up + return quantiles at this bar.
            #        reliability_floor=0.0 here so the baseline_policy gate (trading.
            #        reliability_floor) is the single source of the reliability filter.
            forecasts = [
                build_horizon_forecast(
                    as_of=ts,
                    last_close=1.0,
                    horizon=h,
                    horizon_label=self.config.prediction.horizon_label(h),
                    bar_minutes=bar_minutes,
                    p_up=float(block["p_up"][self._pos]),
                    q10=float(block["q10"][self._pos]),
                    q50=float(block["q50"][self._pos]),
                    q90=float(block["q90"][self._pos]),
                    reliability_floor=0.0,
                )
                for h, block in self.structured_forecasts.items()
            ]
            return MultiHorizonPredictionResult(
                as_of=str(ts),
                last_close=1.0,
                model_version="replay-real",
                symbol=self.config.primary_symbol,
                timeframe=self.config.timeframe,
                horizons=forecasts,
            )
        p_up = float(self.p_up_series[self._pos])
        horizons = (
            self.config.trading.decision_horizons or self.config.prediction.horizons
        )
        forecasts = [
            build_horizon_forecast(
                as_of=ts,
                last_close=1.0,
                horizon=h,
                horizon_label=self.config.prediction.horizon_label(h),
                bar_minutes=bar_minutes,
                p_up=p_up,
                q10=-0.001,
                q50=0.0,
                q90=0.001,
                reliability_floor=0.0,
            )
            for h in horizons
        ]
        return MultiHorizonPredictionResult(
            as_of=str(ts),
            last_close=1.0,
            model_version="replay",
            symbol=self.config.primary_symbol,
            timeframe=self.config.timeframe,
            horizons=forecasts,
        )

    def _obs(self) -> np.ndarray:
        return build_observation(self.current_forecast(), self.portfolio, self.config)

    def step(self, target_weight: float) -> tuple[np.ndarray, float, bool, dict]:
        """Apply action, advance one bar, return (obs, reward, done, info)."""
        trading = self.config.trading
        risk = self.config.risk
        weight = apply_guardrails(target_weight, self.portfolio, trading, risk)
        bar_ret = float(self.returns[self._pos])

        cost_rate = risk.fee_rate + risk.slippage
        delta = weight - self.portfolio.position_weight
        fee = abs(delta) * self.portfolio.equity * cost_rate
        funding = (
            abs(self.portfolio.position_weight)
            * self.portfolio.equity
            * trading.funding_rate_per_bar
        )

        pnl = self.portfolio.position_weight * bar_ret * self.portfolio.equity
        self.portfolio.equity = self.portfolio.equity + pnl - fee - funding
        self.portfolio.peak_equity = max(self.portfolio.peak_equity, self.portfolio.equity)

        if abs(weight) < 1e-9:
            self.portfolio.bars_in_position = 0
        elif abs(weight - self.portfolio.position_weight) < 1e-9 and abs(weight) > 1e-9:
            self.portfolio.bars_in_position += 1
        else:
            self.portfolio.bars_in_position = 1 if abs(weight) > 1e-9 else 0
        self.portfolio.position_weight = weight
        self.portfolio.bars_elapsed += 1

        step_ret = (self.portfolio.equity - self._prev_equity) / max(
            1e-9, self._prev_equity
        )
        reward = step_ret * self.config.rl.sharpe_scale
        reward -= self.config.rl.drawdown_penalty * self.portfolio.drawdown()

        self._prev_equity = self.portfolio.equity
        self._pos += 1
        done = self._pos >= len(self.returns) - 1
        return self._obs(), float(reward), done, {"equity": self.portfolio.equity, "weight": weight}
