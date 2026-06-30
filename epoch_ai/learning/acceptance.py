"""Final holdout acceptance report (predictor + policy benchmarks)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.execution.policy.env import TradingReplayEnv
from epoch_ai.execution.policy.executor import baseline_weight
from epoch_ai.execution.policy.observation import build_observation
from epoch_ai.execution.policy.ppo_policy import PPOPolicy
from epoch_ai.features.pipeline import FeaturePipeline, build_multi_horizon_targets, build_target
from epoch_ai.learning.adaptation import resolved_holdout_bars
from epoch_ai.learning.policy_promotion import replay_metrics
from epoch_ai.learning.promotion import _evaluate
from epoch_ai.learning.step_metrics import multi_horizon_classification_step_metrics
from epoch_ai.models.base import MultiHeadModel
from epoch_ai.models.registry import ModelRegistry


@dataclass(slots=True)
class AcceptanceReport:
    """Holdout-only evaluation summary."""

    holdout_bars: int
    predictor_metrics: dict[str, float] = field(default_factory=dict)
    policy_baseline: dict[str, float] = field(default_factory=dict)
    policy_buy_hold: dict[str, float] = field(default_factory=dict)
    policy_champion: dict[str, float] = field(default_factory=dict)
    skipped: bool = False
    reason: str = ""


def evaluate_holdout(config: AppConfig, *, n_bars: int | None = None) -> AcceptanceReport:
    """Score the promoted predictor and policy benchmarks on the untouched holdout."""
    holdout_bars = resolved_holdout_bars(config)
    market = HistoricalDownloader(config).load_or_download(config.primary_symbol, n_bars=n_bars)
    if len(market) <= holdout_bars + config.walk_forward.initial_train_period:
        return AcceptanceReport(
            holdout_bars=holdout_bars,
            skipped=True,
            reason="insufficient rows for holdout evaluation",
        )

    holdout_market = market.iloc[-holdout_bars:]
    features = FeaturePipeline(config).transform(holdout_market)
    multi = build_multi_horizon_targets(holdout_market, config.prediction)
    data = features.join(multi).dropna()

    predictor_metrics: dict[str, float] = {}
    registry = ModelRegistry(config.model.model_dir)
    try:
        model, _ = registry.load(None, config.model, task=config.prediction.task)
        if isinstance(model, MultiHeadModel) and model.multi_head_spec_ is not None:
            structured = model.predict_structured(data[features.columns])
            horizons = model.multi_head_spec_.horizons
            labels_by_h = {h: data[f"target_{h}"].to_numpy(dtype=float) for h in horizons}
            returns_by_h = {h: data[f"ret_{h}"].to_numpy(dtype=float) for h in horizons}
            predictor_metrics = multi_horizon_classification_step_metrics(
                structured,
                labels_by_h,
                returns_by_h,
                long_threshold=config.risk.long_threshold,
                short_threshold=config.risk.short_threshold,
                primary_horizon=config.prediction.horizon,
            )
        else:
            y = build_target(holdout_market, config.prediction)
            merged = features.join(y).dropna()
            predictor_metrics = _evaluate(
                model,
                merged[features.columns],
                merged["target"].to_numpy(),
                merged["target"].to_numpy(),
                config,
            )
    except FileNotFoundError:
        predictor_metrics = {}

    close = holdout_market["close"]
    holdout_df = pd.DataFrame({"close": close})

    def baseline_fn(env: TradingReplayEnv) -> float:
        return baseline_weight(config, env.current_forecast(), env.portfolio)

    def buy_hold_fn(_env: TradingReplayEnv) -> float:
        return config.trading.max_position_fraction * config.risk.max_leverage

    baseline = replay_metrics(
        TradingReplayEnv.from_market(config, holdout_df.copy()),
        baseline_fn,
    )
    buy_hold = replay_metrics(
        TradingReplayEnv.from_market(config, holdout_df.copy()),
        buy_hold_fn,
    )

    champion_metrics: dict[str, float] = {}
    champion_path = Path(config.rl.promotion.champion_path)
    if champion_path.exists():
        policy = PPOPolicy.load(champion_path, config.rl)
        cap = config.trading.max_position_fraction * config.risk.max_leverage

        def ppo_fn(env: TradingReplayEnv) -> float:
            obs = build_observation(env.current_forecast(), env.portfolio, config)
            return float(policy.act(obs, deterministic=True) * cap)

        replay = replay_metrics(
            TradingReplayEnv.from_market(config, holdout_df.copy()),
            ppo_fn,
        )
        champion_metrics = {
            "total_return": replay.total_return,
            "sharpe": replay.sharpe,
            "risk_adjusted_return": replay.risk_adjusted_return,
            "max_drawdown": replay.max_drawdown,
        }

    return AcceptanceReport(
        holdout_bars=holdout_bars,
        predictor_metrics=predictor_metrics,
        policy_baseline={
            "total_return": baseline.total_return,
            "sharpe": baseline.sharpe,
            "risk_adjusted_return": baseline.risk_adjusted_return,
            "max_drawdown": baseline.max_drawdown,
        },
        policy_buy_hold={
            "total_return": buy_hold.total_return,
            "sharpe": buy_hold.sharpe,
            "risk_adjusted_return": buy_hold.risk_adjusted_return,
            "max_drawdown": buy_hold.max_drawdown,
        },
        policy_champion=champion_metrics,
    )
