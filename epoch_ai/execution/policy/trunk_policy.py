"""Shared-trunk (A.5) policy: an actor-critic head over the TCN trunk embedding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.policy.env import TradingReplayEnv
from epoch_ai.execution.policy.observation import embedding_observation_dim
from epoch_ai.execution.policy.ppo_policy import PPOPolicy
from epoch_ai.features.pipeline import FeaturePipeline

if TYPE_CHECKING:
    # Human: embed()/trunk_dim are TCN-specific (not on MultiHeadModel/evolved_nn), so the
    #        contract is a TCNModel; imported only for typing to avoid a heavy runtime import.
    from epoch_ai.models.tcn_model import TCNModel


def build_trunk_policy(trunk_dim: int, config: AppConfig) -> PPOPolicy:
    """Build the shared-trunk policy/value head over the TCN trunk embedding (A.5).

    The head consumes the shared trunk embedding (width ``trunk_dim``) concatenated with
    the 4 portfolio scalars, and reuses :class:`PPOPolicy`'s config-sized actor-critic
    (``config.rl.hidden_sizes``). This is scaffolding only: the trunk itself is frozen
    here; joint trunk fine-tuning is a separate, GPU-gated task.

    Args:
        trunk_dim: Width of the model's trunk embedding (``model.trunk_dim``).
        config: Resolved app config; ``config.rl`` sizes the actor-critic.
    """
    return PPOPolicy(embedding_observation_dim(trunk_dim), config.rl)


def build_embedding_env(
    config: AppConfig,
    market_slice: pd.DataFrame,
    model: TCNModel,
) -> TradingReplayEnv:
    """Build a shared-trunk replay env over ``market_slice`` using the model's embeddings.

    Mirrors ``_build_policy_env_from_model`` but exposes the causal TCN trunk embedding
    (``model.embed``) as the observation instead of the per-horizon forecasts.

    Causality: features are computed causally by :class:`FeaturePipeline`; warmup NaN rows
    are dropped; ``close`` is aligned to the surviving rows; and
    :meth:`TradingReplayEnv.from_embeddings` shifts the realized return forward by one bar
    so the embedding at bar ``i`` earns the ``i -> i+1`` return (no look-ahead).

    Args:
        config: Resolved app config (feature settings, ``rl.observation_mode``).
        market_slice: OHLCV slice to replay (must contain a ``close`` column).
        model: A trained model exposing a causal ``embed`` and ``trunk_dim``.

    Returns:
        A :class:`TradingReplayEnv` in embedding mode (``embeddings`` set).
    """
    # Human: hard precondition -- the env only emits the embedding observation in
    #        embedding mode; a forecast-mode config would silently return a differently
    #        sized observation and break a trunk policy built for trunk_dim + 4.
    if config.rl.observation_mode != "embedding":
        raise ValueError(
            "build_embedding_env requires config.rl.observation_mode == 'embedding'; "
            f"got {config.rl.observation_mode!r}."
        )
    if not (hasattr(model, "embed") and hasattr(model, "trunk_dim")):
        raise TypeError(
            "build_embedding_env requires a model exposing embed()/trunk_dim (e.g. TCNModel)."
        )
    # Agent: CAUSAL feature transform; dropna trims warmup rows before embedding.
    features = FeaturePipeline(config).transform(market_slice).dropna()
    # Agent: model.embed handles sequence (TCN) windowing; rows align 1:1 with input.
    emb = model.embed(features[list(features.columns)])
    close = market_slice.loc[features.index, "close"].astype(float)
    return TradingReplayEnv.from_embeddings(config, close, emb)


def runtime_trunk_embedding(
    config: AppConfig,
    model: object,
    feature_window: pd.DataFrame,
) -> np.ndarray | None:
    """Return the latest causal trunk embedding row for live/replay policy obs."""
    if config.rl.observation_mode != "embedding":
        return None
    from epoch_ai.models.tcn_model import TCNModel

    if not isinstance(model, TCNModel) or model.multi_head_spec_ is None:
        return None
    cols = list(model.feature_names_ or feature_window.columns)
    frame = feature_window[cols]
    if frame.empty:
        return None
    emb = model.embed(frame)
    return np.asarray(emb[-1], dtype=np.float32)
