"""Joint / staged trunk policy training (A.5 Stage 1 frozen, Stage 2 joint fine-tune)."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.policy.env import TradingReplayEnv
from epoch_ai.execution.policy.ppo_policy import PPOPolicy, TrainStats
from epoch_ai.execution.policy.trunk_policy import build_embedding_env, build_trunk_policy
from epoch_ai.features.pipeline import (
    FeaturePipeline,
    build_multi_horizon_targets,
    build_target,
)
from epoch_ai.utils.logging import get_logger

if TYPE_CHECKING:
    from epoch_ai.models.tcn_model import TCNModel

logger = get_logger(__name__)


def _supervised_frame(market_slice: pd.DataFrame, config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aligned feature + multi-target frames for supervised trunk steps."""
    features = FeaturePipeline(config).transform(market_slice)
    y = build_target(market_slice, config.prediction)
    multi = build_multi_horizon_targets(market_slice, config.prediction)
    keep = ["target"]
    for h in config.prediction.horizons:
        keep.extend([f"ret_{h}", f"target_{h}"])
    data = features.join(y).join(multi).dropna(subset=keep)
    multi_cols = [c for c in data.columns if c.startswith(("ret_", "target_"))]
    return data[features.columns], data[multi_cols]


def joint_trunk_enabled(config: AppConfig) -> bool:
    """True when Stage-2 joint trunk fine-tuning is configured."""
    return (
        config.rl.observation_mode == "embedding"
        and not config.rl.trunk_frozen
        and config.rl.policy_loss_weight > 0.0
    )


def train_trunk_policy(
    config: AppConfig,
    train_market: pd.DataFrame,
    model: TCNModel,
    *,
    clone_model: bool = True,
) -> tuple[PPOPolicy, TCNModel, TrainStats]:
    """Train a PPO policy on the shared TCN trunk embedding (frozen or joint).

    Stage 1 (default): ``trunk_frozen=True`` — policy learns on precomputed embeddings;
    the TCN trunk weights are untouched.

    Stage 2 (joint): ``trunk_frozen=False`` and ``policy_loss_weight > 0`` — after each
    PPO update, run ``supervised_aux_steps`` supervised mini-batches on the trunk, then
    refresh the replay env embeddings so the policy sees the updated representation.

    Args:
        config: Resolved app config; ``rl.observation_mode`` must be ``embedding``.
        train_market: OHLCV slice for policy training (pre-holdout).
        model: Trained TCN champion with ``embed()`` and multi-head spec.
        clone_model: When True (default), deep-copy the model so joint steps do not
            mutate the registry champion unless promotion passes.

    Returns:
        ``(policy, work_model, train_stats)`` where ``work_model`` may differ from the
        input when joint fine-tuning ran.
    """
    if config.rl.observation_mode != "embedding":
        raise ValueError("train_trunk_policy requires rl.observation_mode='embedding'.")

    work_model: TCNModel = copy.deepcopy(model) if clone_model else model
    policy = build_trunk_policy(work_model.trunk_dim, config)
    env = build_embedding_env(config, train_market, work_model)

    joint = joint_trunk_enabled(config)
    if not joint:
        stats = policy.train(env)
        return policy, work_model, stats

    features, multi_targets = _supervised_frame(train_market, config)
    rl = config.rl
    logger.info(
        "Joint trunk policy training: policy_loss_weight=%.4f aux_steps=%d",
        rl.policy_loss_weight,
        rl.supervised_aux_steps,
    )

    def _refresh_env() -> TradingReplayEnv:
        return build_embedding_env(config, train_market, work_model)

    def _on_update(_update: int, current_env: TradingReplayEnv) -> None:
        nonlocal env
        loss = work_model.supervised_gradient_step(
            features,
            multi_targets,
            steps=rl.supervised_aux_steps,
            aux_weight=rl.prediction_aux_weight * rl.policy_loss_weight,
        )
        env = _refresh_env()
        logger.debug(
            "Joint trunk aux step mean_loss=%.6f refreshed_env_bars=%d",
            loss,
            len(env.returns),
        )

    stats = policy.train(env, on_update=_on_update)
    return policy, work_model, stats
