"""Causal trading replay environment for PPO training on historical bars."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.policy.guardrails import apply_guardrails
from epoch_ai.execution.policy.observation import (
    build_embedding_observation,
    build_observation,
    observation_dim,
)
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
    # Human: shared-trunk (A.5) path. When set, _obs() emits the TCN trunk embedding at the
    #        current bar (plus portfolio scalars) instead of the forecast/proxy observation.
    # Agent: shape (n_rows, trunk_dim) aligned 1:1 to ``returns`` rows; None => forecast mode.
    embeddings: np.ndarray | None = None
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

    @classmethod
    def from_embeddings(
        cls,
        config: AppConfig,
        close: pd.Series,
        embeddings: np.ndarray,
    ) -> TradingReplayEnv:
        """Build a shared-trunk (A.5) env whose observation is the TCN trunk embedding.

        Mirrors :meth:`from_forecasts` causality: the embedding at row ``i`` is causal
        (features up to ``i``) and must earn the ``i -> i+1`` return, so the realized
        return is shifted forward by one bar (no look-ahead). ``structured_forecasts`` is
        left ``None`` so this env has no forecast dependency at all.

        Args:
            close: Close prices aligned 1:1 with the rows of ``embeddings``.
            embeddings: ``(n_rows, trunk_dim)`` trunk embeddings from ``model.embed``.
        """
        close = close.astype(float)
        # Human: identical shift to from_forecasts -- embedding at row i earns i->i+1 return.
        # Agent: CAUSAL; returns[i] = (close[i+1]/close[i]-1); last row -> 0 (no next bar).
        rets = close.pct_change().shift(-1).fillna(0.0).to_numpy(dtype=np.float32)
        emb = np.asarray(embeddings, dtype=np.float32)
        return cls(
            config=config,
            returns=rets,
            # Agent: p_up_series is unused in embedding obs mode; keep a neutral 0.5 proxy.
            p_up_series=np.full(len(rets), 0.5, dtype=np.float32),
            obs_dim=emb.shape[1] + 4,
            embeddings=emb,
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
        # Human: shared-trunk (A.5) obs uses the trunk embedding directly. Branch BEFORE any
        #        current_forecast() call so a pure embedding env (no forecasts) still works.
        # Agent: CONFIG rl.observation_mode == "embedding" + embeddings set => embedding obs.
        if self.config.rl.observation_mode == "embedding" and self.embeddings is not None:
            return build_embedding_observation(
                self.embeddings[self._pos], self.portfolio, self.config
            )
        return build_observation(self.current_forecast(), self.portfolio, self.config)

    def step(self, target_weight: float) -> tuple[np.ndarray, float, bool, dict]:
        """Apply action and return (obs, reward, done, info).

        Behaviour depends on ``config.rl.reward_mode``:

        * ``per_bar`` (legacy): advance exactly one bar; reward is the single-bar
          equity change (numerically identical to the historical implementation),
          minus an optional turnover penalty (default 0.0 => no-op).
        * ``multi_bar`` (default): treat one call as ONE decision held constant for
          up to ``config.rl.reward_horizon`` bars; reward is the accumulated block
          return net of the (once-charged) entry fee, giving a lower-noise signal.
        """
        rl = self.config.rl
        if rl.reward_mode == "per_bar":
            return self._step_per_bar(target_weight)
        return self._step_multi_bar(target_weight)

    def _step_per_bar(self, target_weight: float) -> tuple[np.ndarray, float, bool, dict]:
        """Legacy single-bar reward path (kept numerically identical to the original)."""
        trading = self.config.trading
        risk = self.config.risk
        weight = apply_guardrails(target_weight, self.portfolio, trading, risk)
        # Human: capture the pre-trade weight before we overwrite position_weight so the
        #        turnover term measures the actual change this decision requests.
        # Agent: prev_position_weight also drives the fee delta below; capture once.
        prev_position_weight = self.portfolio.position_weight
        bar_ret = float(self.returns[self._pos])

        cost_rate = risk.fee_rate + risk.slippage
        delta = weight - prev_position_weight
        fee = abs(delta) * self.portfolio.equity * cost_rate
        funding = (
            abs(prev_position_weight)
            * self.portfolio.equity
            * trading.funding_rate_per_bar
        )

        pnl = prev_position_weight * bar_ret * self.portfolio.equity
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
        # Human: turnover penalty discourages churn; defaults to 0.0 so legacy runs and
        #        tests are unaffected (this line is a no-op unless explicitly configured).
        # Agent: CONFIG rl.turnover_penalty; term == penalty * |weight - prev_weight|.
        reward -= self.config.rl.turnover_penalty * abs(weight - prev_position_weight)

        self._prev_equity = self.portfolio.equity
        self._pos += 1
        done = self._pos >= len(self.returns) - 1
        # Human: equity_path lets replay_metrics build an honest per-bar equity curve. In
        #        per_bar mode exactly one bar is consumed, so it is a 1-element list.
        # Agent: RETURNS equity AFTER the single bar consumed this step.
        return (
            self._obs(),
            float(reward),
            done,
            {
                "equity": self.portfolio.equity,
                "weight": weight,
                "equity_path": [self.portfolio.equity],
            },
        )

    def _step_multi_bar(self, target_weight: float) -> tuple[np.ndarray, float, bool, dict]:
        """Hold one decision for up to ``reward_horizon`` bars; reward is the block return.

        Causality: the action is chosen from the observation at the CURRENT ``_pos``.
        Only bars within the held block (``_pos`` .. ``_pos + n_bars``) affect equity, and
        the returned observation reflects the state AT the new ``_pos`` (the next decision
        point), so no bar beyond the block leaks into the observation.
        """
        trading = self.config.trading
        risk = self.config.risk
        rl = self.config.rl
        weight = apply_guardrails(target_weight, self.portfolio, trading, risk)
        # Human: guardrails + the entry fee are applied ONCE, at the decision boundary,
        #        not per held bar -- this is what makes the cadence "multi-bar".
        # Agent: prev_position_weight sizes the fee delta AND the turnover term below.
        prev_position_weight = self.portfolio.position_weight

        cost_rate = risk.fee_rate + risk.slippage
        # Human: denominator for the block return is equity BEFORE the entry fee, so the
        #        fee genuinely reduces the reward instead of cancelling out.
        # Agent: decision_equity_before_fee = equity pre-fee; decision_equity = post-fee.
        decision_equity_before_fee = self.portfolio.equity
        fee = abs(weight - prev_position_weight) * decision_equity_before_fee * cost_rate
        self.portfolio.equity = self.portfolio.equity - fee
        decision_equity = self.portfolio.equity

        # Human: hold the weight constant and roll equity forward one bar at a time up to
        #        reward_horizon bars, stopping early if the return series is exhausted.
        # Agent: CAUSAL; consumes returns[_pos] then advances _pos; never reads beyond it.
        # Human: record equity AFTER each held bar so replay_metrics can reconstruct an
        #        honest per-bar equity curve and see intra-block drawdowns (a step spans
        #        up to reward_horizon bars, so sampling only at the boundary hides dips).
        # Agent: equity_path appended inside the hold loop; one entry per consumed bar.
        n_bars_held = 0
        equity_path: list[float] = []
        for _ in range(rl.reward_horizon):
            bar_ret = float(self.returns[self._pos])
            pnl = weight * bar_ret * self.portfolio.equity
            funding = abs(weight) * self.portfolio.equity * trading.funding_rate_per_bar
            self.portfolio.equity = self.portfolio.equity + pnl - funding
            self.portfolio.peak_equity = max(
                self.portfolio.peak_equity, self.portfolio.equity
            )
            self._pos += 1
            self.portfolio.bars_elapsed += 1
            n_bars_held += 1
            equity_path.append(self.portfolio.equity)
            if self._pos >= len(self.returns) - 1:
                break

        # Human: block return includes the entry fee (subtracted in the numerator) and is
        #        measured against pre-fee equity; equals (final - pre_fee) / pre_fee.
        # Agent: reward = block_ret*sharpe_scale - dd_penalty*drawdown - turnover*|dw|.
        block_ret = (self.portfolio.equity - decision_equity - fee) / max(
            1e-9, decision_equity_before_fee
        )
        reward = block_ret * rl.sharpe_scale
        reward -= rl.drawdown_penalty * self.portfolio.drawdown()
        reward -= rl.turnover_penalty * abs(weight - prev_position_weight)

        self.portfolio.position_weight = weight
        self.portfolio.bars_in_position = n_bars_held if abs(weight) > 1e-9 else 0

        self._prev_equity = self.portfolio.equity
        done = self._pos >= len(self.returns) - 1
        return (
            self._obs(),
            float(reward),
            done,
            {
                "equity": self.portfolio.equity,
                "weight": weight,
                "bars_held": n_bars_held,
                "equity_path": equity_path,
            },
        )
