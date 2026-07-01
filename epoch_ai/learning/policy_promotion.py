"""Automated PPO policy train → evaluate → promote-if-better on a holdout tail."""

from __future__ import annotations

import math
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.execution.policy.env import TradingReplayEnv
from epoch_ai.execution.policy.executor import baseline_weight
from epoch_ai.execution.policy.guardrails import apply_guardrails
from epoch_ai.execution.policy.observation import build_observation, observation_dim
from epoch_ai.execution.policy.ppo_policy import PPOPolicy
from epoch_ai.features.pipeline import FeaturePipeline
from epoch_ai.learning.adaptation import resolved_holdout_bars
from epoch_ai.models.base import MultiHeadModel
from epoch_ai.models.registry import ModelRegistry
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


def _load_multi_head_champion(config: AppConfig) -> MultiHeadModel | None:
    """Load the promoted champion iff it is a usable multi-head predictor.

    Returns the loaded :class:`MultiHeadModel` when a multi-head champion exists in the
    registry, else ``None`` (no model yet, or a non-multi-head backend). Callers fall
    back to the price-only proxy env when this returns ``None``.

    Args:
        config: Resolved app config; reads ``model.model_dir`` and ``prediction.task``.

    Returns:
        The champion model, or ``None`` when unavailable / not multi-head.
    """
    # Agent: READS registry(config.model.model_dir); FileNotFoundError => no model yet.
    try:
        model, _ = ModelRegistry(config.model.model_dir).load(
            None, config.model, task=config.prediction.task
        )
    except FileNotFoundError:
        return None
    if isinstance(model, MultiHeadModel) and model.multi_head_spec_ is not None:
        return model
    return None


def _build_policy_env_from_model(
    config: AppConfig,
    market_slice: pd.DataFrame,
    model: MultiHeadModel,
) -> TradingReplayEnv:
    """Build a real-forecast replay env over ``market_slice`` using the champion model.

    The policy is trained/scored on the actual trained model's per-bar, per-horizon
    forecasts (``predict_structured``) instead of the price-only ``from_market`` proxy.

    Causality: features are computed causally by :class:`FeaturePipeline`; warmup NaN
    rows are dropped; ``close`` is aligned to the surviving prediction rows; and
    :meth:`TradingReplayEnv.from_forecasts` shifts the realized return forward by one bar
    so the forecast at bar ``i`` earns the ``i -> i+1`` return (no look-ahead).

    Args:
        config: Resolved app config (feature/prediction settings).
        market_slice: OHLCV slice to replay (must contain a ``close`` column).
        model: A trained multi-head model with a populated ``multi_head_spec_``.

    Returns:
        A :class:`TradingReplayEnv` in real-forecast mode (``structured_forecasts`` set).
    """
    # Agent: CAUSAL feature transform; dropna trims warmup rows before prediction.
    features = FeaturePipeline(config).transform(market_slice)
    features = features.dropna()
    # Agent: predict_structured handles sequence (TCN) windowing; rows align 1:1 with input.
    structured = model.predict_structured(features[list(features.columns)])
    close = market_slice.loc[features.index, "close"].astype(float)
    horizons = list(model.multi_head_spec_.horizons)
    return TradingReplayEnv.from_forecasts(config, close, structured, horizons)


@dataclass(slots=True)
class ReplayMetrics:
    """Paper-replay summary on a bar slice."""

    total_return: float
    sharpe: float
    risk_adjusted_return: float
    max_drawdown: float
    final_equity: float


@dataclass(slots=True)
class PolicyPromoteResult:
    """Outcome of one PPO train + promotion cycle."""

    challenger_path: str | None
    champion_path: str | None
    promoted: bool
    metric: str
    challenger_value: float = float("nan")
    champion_value: float = float("nan")
    baseline_value: float = float("nan")
    buy_hold_value: float = float("nan")
    train_bars: int = 0
    eval_bars: int = 0
    skipped: bool = False
    reason: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


def metric_value(metrics: ReplayMetrics, metric: str) -> float:
    """Select the scalar gate metric from a replay summary (higher is better)."""
    if metric == "sharpe":
        return metrics.sharpe
    if metric == "total_return":
        return metrics.total_return
    return metrics.risk_adjusted_return


def replay_metrics(
    env: TradingReplayEnv, weight_fn: Callable[[TradingReplayEnv], float]
) -> ReplayMetrics:
    """Simulate ``weight_fn`` over the env's return series.

    Metrics are computed on an HONEST per-bar equity curve rebuilt from each step's
    ``info["equity_path"]`` (one entry per bar consumed). This matters under
    ``reward_mode="multi_bar"``: a single ``env.step`` can span up to ``reward_horizon``
    bars, so sampling equity only at step boundaries would miss intra-block drawdowns
    (optimistic ``max_drawdown``) and compute Sharpe on block returns rather than per-bar
    returns. The curve starts at ``start_eq`` and is extended with every held bar.
    """
    env.reset()
    start_eq = env.portfolio.equity
    # Human: per-bar equity curve seeded with the starting equity; each step contributes
    #        one point per bar it consumed (per_bar => 1 point, multi_bar => up to N).
    # Agent: curve length == total bars consumed + 1; drives Sharpe + max_dd below.
    curve: list[float] = [start_eq]

    while not env.done:
        weight = apply_guardrails(
            weight_fn(env),
            env.portfolio,
            env.config.trading,
            env.config.risk,
        )
        _, _, done, info = env.step(weight)
        # Agent: defensive default keeps this working for any future step variant that
        #        omits equity_path (both current modes always provide it).
        curve.extend(float(eq) for eq in info.get("equity_path", [info["equity"]]))
        if done:
            break

    curve_arr = np.asarray(curve, dtype=float)
    # Per-bar returns from the honest equity curve (floor denominator to stay finite if
    # equity ever hits ~0, mirroring the original per-step guard).
    per_bar_rets = (
        np.diff(curve_arr) / np.maximum(1e-9, curve_arr[:-1])
        if len(curve_arr) > 1
        else np.array([])
    )
    sharpe = (
        float(per_bar_rets.mean() / (per_bar_rets.std() + 1e-9))
        if len(per_bar_rets)
        else 0.0
    )
    # Running max drawdown over the per-bar curve (catches intra-block dips).
    peak = np.maximum.accumulate(curve_arr)
    max_dd = float(((peak - curve_arr) / peak).max()) if len(curve_arr) else 0.0
    total_return = float((env.portfolio.equity - start_eq) / max(1e-9, start_eq))
    risk_adj = total_return / max(max_dd, 1e-6)
    return ReplayMetrics(
        total_return=total_return,
        sharpe=sharpe,
        risk_adjusted_return=risk_adj,
        max_drawdown=max_dd,
        final_equity=float(env.portfolio.equity),
    )


def decide_policy_promotion(
    *,
    challenger_value: float,
    champion_value: float | None,
    baseline_value: float,
    buy_hold_value: float,
    metric: str,
    min_improvement: float,
    require_beat_baseline: bool,
    require_beat_buy_hold: bool,
    min_absolute_metric: float = 0.0,
) -> tuple[bool, str]:
    """Gate promotion on champion improvement, an absolute floor, and optional benchmarks.

    Args:
        min_absolute_metric: Absolute floor on ``metric``. Even an otherwise-promotable
            challenger (including the bootstrap case with no champion) is refused when its
            value is below this floor, so a money-losing policy is never promoted.
        require_beat_baseline / require_beat_buy_hold: Now default report-only (see
            :class:`PolicyPromotionConfig`); still enforced as hard gates when a caller
            passes ``True``.
    """
    if challenger_value is None or math.isnan(challenger_value):
        return False, "challenger metric is undefined (NaN); keeping champion"

    if champion_value is None or math.isnan(champion_value):
        promote = True
        reason = "no usable champion policy; promoting challenger (bootstrap)"
    elif challenger_value <= champion_value:
        return False, (
            f"challenger does not beat champion on {metric} "
            f"({challenger_value:.6f} <= {champion_value:.6f})"
        )
    elif challenger_value - champion_value < min_improvement:
        return False, (
            f"challenger improvement {challenger_value - champion_value:.6f} "
            f"< {min_improvement:.6f}"
        )
    else:
        promote = True
        reason = f"challenger improves {metric} over champion"

    # Human: absolute floor is the primary guard now that benchmark beats are report-only.
    #        It applies to BOTH the champion-improvement path and the bootstrap path so we
    #        never promote a challenger whose metric is below the floor (e.g. losing money).
    # Agent: enforced after the champion/bootstrap decision, before optional benchmark gates.
    if promote and challenger_value < min_absolute_metric:
        return False, (
            f"challenger {metric}={challenger_value:.6f} below absolute floor "
            f"{min_absolute_metric:.6f}"
        )

    if promote and require_beat_baseline and challenger_value <= baseline_value:
        return False, (
            f"challenger {metric}={challenger_value:.6f} "
            f"does not beat baseline {baseline_value:.6f}"
        )
    if promote and require_beat_buy_hold and challenger_value <= buy_hold_value:
        return False, (
            f"challenger {metric}={challenger_value:.6f} "
            f"does not beat buy-and-hold {buy_hold_value:.6f}"
        )
    return promote, reason


def auto_train_and_promote_policy(
    config: AppConfig,
    *,
    n_bars: int | None = None,
) -> PolicyPromoteResult:
    """Train a challenger PPO on pre-holdout data; promote only if it beats benchmarks."""
    promo = config.rl.promotion
    if not promo.enabled:
        return PolicyPromoteResult(
            challenger_path=None,
            champion_path=None,
            promoted=False,
            metric=promo.metric,
            skipped=True,
            reason="rl.promotion.enabled is false",
        )

    market = HistoricalDownloader(config).load_or_download(config.primary_symbol, n_bars=n_bars)
    close = market["close"].astype(float)
    if len(close) < config.walk_forward.initial_train_period + 20:
        return PolicyPromoteResult(
            challenger_path=None,
            champion_path=promo.champion_path,
            promoted=False,
            metric=promo.metric,
            skipped=True,
            reason="insufficient history for policy train/holdout split",
        )

    eval_bars = min(
        promo.eval_bars or resolved_holdout_bars(config),
        max(1, len(close) - config.walk_forward.initial_train_period - 1),
    )
    holdout = close.iloc[-eval_bars:]
    train = close.iloc[: len(close) - eval_bars]

    # Human: prefer the champion model's REAL forecasts for both the train env and every
    #        holdout benchmark env. When no multi-head champion exists, fall back to the
    #        price-only proxy so bootstrap cycles still work.
    # Agent: CAUSAL split -- train_market is the pre-holdout slice, holdout_market the tail.
    champion_model = _load_multi_head_champion(config)
    holdout_market = market.iloc[-eval_bars:]
    train_market = market.iloc[: len(market) - eval_bars]

    if champion_model is not None:
        train_env = _build_policy_env_from_model(config, train_market, champion_model)

        def make_eval_env() -> TradingReplayEnv:
            return _build_policy_env_from_model(config, holdout_market, champion_model)
    else:
        train_env = TradingReplayEnv.from_market(config, pd.DataFrame({"close": train}))

        def make_eval_env() -> TradingReplayEnv:
            return TradingReplayEnv.from_market(config, pd.DataFrame({"close": holdout}))

    challenger = PPOPolicy(observation_dim(config), config.rl)
    challenger.train(train_env)

    challenger_path = Path(config.rl.policy_path)
    challenger_path.parent.mkdir(parents=True, exist_ok=True)
    challenger.save(challenger_path)

    def baseline_fn(env: TradingReplayEnv) -> float:
        return baseline_weight(config, env.current_forecast(), env.portfolio)

    def buy_hold_fn(_env: TradingReplayEnv) -> float:
        cap = config.trading.max_position_fraction * config.risk.max_leverage
        return cap

    def ppo_fn(env: TradingReplayEnv) -> float:
        obs = build_observation(env.current_forecast(), env.portfolio, config)
        cap = config.trading.max_position_fraction * config.risk.max_leverage
        return float(challenger.act(obs, deterministic=True) * cap)

    challenger_metrics = replay_metrics(make_eval_env(), ppo_fn)
    baseline_metrics = replay_metrics(make_eval_env(), baseline_fn)
    buy_hold_metrics = replay_metrics(make_eval_env(), buy_hold_fn)

    metric = promo.metric
    challenger_value = metric_value(challenger_metrics, metric)
    baseline_value = metric_value(baseline_metrics, metric)
    buy_hold_value = metric_value(buy_hold_metrics, metric)

    champion_path = Path(promo.champion_path)
    champion_value: float | None = None
    if champion_path.exists():
        try:
            champion = PPOPolicy.load(champion_path, config.rl)

            def champion_fn(env: TradingReplayEnv) -> float:
                obs = build_observation(env.current_forecast(), env.portfolio, config)
                cap = config.trading.max_position_fraction * config.risk.max_leverage
                return float(champion.act(obs, deterministic=True) * cap)

            champion_value = metric_value(
                replay_metrics(make_eval_env(), champion_fn),
                metric,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not score champion policy: %s", exc)
            champion_value = None

    promote, reason = decide_policy_promotion(
        challenger_value=challenger_value,
        champion_value=champion_value,
        baseline_value=baseline_value,
        buy_hold_value=buy_hold_value,
        metric=metric,
        min_improvement=promo.min_improvement,
        require_beat_baseline=promo.require_beat_baseline,
        require_beat_buy_hold=promo.require_beat_buy_hold,
        min_absolute_metric=promo.min_absolute_metric,
    )

    if promote:
        champion_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(challenger_path, champion_path)

    logger.info(
        "Policy auto-train: challenger=%s %s=%.6f baseline=%.6f buy_hold=%.6f -> %s [%s]",
        challenger_path,
        metric,
        challenger_value,
        baseline_value,
        buy_hold_value,
        "PROMOTED" if promote else "kept champion",
        reason,
    )

    return PolicyPromoteResult(
        challenger_path=str(challenger_path),
        champion_path=str(champion_path),
        promoted=promote,
        metric=metric,
        challenger_value=challenger_value,
        champion_value=float("nan") if champion_value is None else champion_value,
        baseline_value=baseline_value,
        buy_hold_value=buy_hold_value,
        train_bars=len(train),
        eval_bars=eval_bars,
        reason=reason,
        metrics={
            "challenger": challenger_value,
            "baseline": baseline_value,
            "buy_hold": buy_hold_value,
            "champion": float("nan") if champion_value is None else champion_value,
        },
    )
