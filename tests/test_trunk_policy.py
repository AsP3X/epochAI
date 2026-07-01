"""Tests for the shared-trunk (A.5) embedding policy scaffolding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from epoch_ai.config.settings import RLConfig
from epoch_ai.execution.policy.env import TradingReplayEnv
from epoch_ai.execution.policy.observation import (
    build_embedding_observation,
    embedding_observation_dim,
)
from epoch_ai.execution.portfolio_state import PortfolioState
from tests.test_policy_env import _policy_config


def test_embedding_observation_dim_and_shape():
    # Agent: embedding obs == trunk_dim + 4 portfolio scalars; zeros embedding -> zeros head.
    config = _policy_config()
    assert embedding_observation_dim(16) == 20
    obs = build_embedding_observation(
        np.zeros(16), PortfolioState.initial(10_000.0), config
    )
    assert len(obs) == 20
    assert np.allclose(obs[:16], 0.0)


def test_from_embeddings_env_is_causal_and_embedding_obs():
    # Agent: from_embeddings mirrors from_forecasts shift(-1) causality; obs is embedding row.
    config = _policy_config(rl={"observation_mode": "embedding"})
    n = 40
    close = pd.Series(np.linspace(100.0, 120.0, n))
    embeddings = np.arange(n * 3).reshape(n, 3).astype(float)
    env = TradingReplayEnv.from_embeddings(config, close, embeddings)
    env.reset()

    obs = env._obs()
    assert len(obs) == 3 + 4
    assert np.allclose(obs[:3], embeddings[env._pos])

    assert len(env.returns) == len(embeddings)
    expected = close.pct_change().shift(-1).fillna(0.0).to_numpy()
    assert np.allclose(env.returns, expected, atol=1e-6)
    assert env.returns[-1] == 0.0


def test_observation_mode_config():
    # Agent: observation_mode is a Literal["forecast","embedding"]; bogus -> ValidationError.
    assert RLConfig(observation_mode="embedding").observation_mode == "embedding"
    assert RLConfig().observation_mode == "forecast"
    with pytest.raises(ValidationError):
        RLConfig(observation_mode="bogus")


def test_trunk_policy_trains_on_embedding_env(market, small_config, tmp_path):
    # Human: the shared-trunk policy head must train over the real TCN trunk embedding.
    #        We fit a tiny multi-head TCN, build an embedding env, and run a tiny PPO
    #        budget to confirm the scaffolding wires end-to-end without error.
    # Agent: CAUSAL; build_embedding_env uses model.embed; obs_dim == trunk_dim + 4.
    pytest.importorskip("torch")

    from epoch_ai.execution.policy.trunk_policy import (
        build_embedding_env,
        build_trunk_policy,
    )
    from epoch_ai.features.pipeline import (
        FeaturePipeline,
        build_multi_horizon_targets,
        build_target,
    )
    from epoch_ai.models.tcn_model import TCNModel

    cfg = small_config
    cfg.model.backend = "tcn"
    cfg.model.calibration = "none"
    cfg.model.val_fraction = 0.2
    cfg.model.refit_full_after_es = False
    cfg.model.tcn.lookback = 16
    cfg.model.tcn.channels = [16, 16]
    cfg.model.tcn.kernel_size = 3
    cfg.model.tcn.max_epochs = 8
    cfg.model.tcn.patience = 3
    cfg.model.tcn.batch_size = 128
    cfg.prediction.horizons = [4, 8]
    cfg.prediction.horizon = 8
    cfg.rl.observation_mode = "embedding"
    cfg.rl.device = "cpu"
    cfg.rl.total_updates = 1
    cfg.rl.rollout_steps = 16

    features = FeaturePipeline(cfg).transform(market)
    y = build_target(market, cfg.prediction)
    multi = build_multi_horizon_targets(market, cfg.prediction)
    keep = ["target"]
    for h in cfg.prediction.horizons:
        keep.extend([f"ret_{h}", f"target_{h}"])
    data = features.join(y).join(multi).dropna(subset=keep)
    multi_cols = [c for c in data.columns if c.startswith(("ret_", "target_"))]
    x, target, mt = data[features.columns], data["target"], data[multi_cols]

    model = TCNModel(cfg.model, task="classification")
    model.fit(
        x.iloc[:2400],
        target.iloc[:2400],
        prediction=cfg.prediction,
        multi_targets=mt.iloc[:2400],
    )

    env = build_embedding_env(cfg, market.iloc[:1200], model)
    policy = build_trunk_policy(model.trunk_dim, cfg)
    assert policy.obs_dim == model.trunk_dim + 4

    policy.train(env)
    assert env.embeddings is not None


def test_runtime_trunk_embedding_returns_latest_row(market, small_config):
    pytest.importorskip("torch")
    from epoch_ai.execution.policy.trunk_policy import runtime_trunk_embedding
    from epoch_ai.features.pipeline import (
        FeaturePipeline,
        build_multi_horizon_targets,
        build_target,
    )
    from epoch_ai.models.tcn_model import TCNModel

    cfg = small_config
    cfg.model.backend = "tcn"
    cfg.model.calibration = "none"
    cfg.model.val_fraction = 0.2
    cfg.model.refit_full_after_es = False
    cfg.model.tcn.lookback = 16
    cfg.model.tcn.channels = [16, 16]
    cfg.model.tcn.kernel_size = 3
    cfg.model.tcn.max_epochs = 6
    cfg.model.tcn.patience = 2
    cfg.model.tcn.batch_size = 128
    cfg.prediction.horizons = [4, 8]
    cfg.prediction.horizon = 8
    cfg.rl.observation_mode = "embedding"

    features = FeaturePipeline(cfg).transform(market)
    y = build_target(market, cfg.prediction)
    multi = build_multi_horizon_targets(market, cfg.prediction)
    keep = ["target"]
    for h in cfg.prediction.horizons:
        keep.extend([f"ret_{h}", f"target_{h}"])
    data = features.join(y).join(multi).dropna(subset=keep)
    multi_cols = [c for c in data.columns if c.startswith(("ret_", "target_"))]
    x, target, mt = data[features.columns], data["target"], data[multi_cols]

    model = TCNModel(cfg.model, task="classification")
    model.fit(
        x.iloc[:2400],
        target.iloc[:2400],
        prediction=cfg.prediction,
        multi_targets=mt.iloc[:2400],
    )
    window = x.iloc[:200]
    row = runtime_trunk_embedding(cfg, model, window)
    assert row is not None
    assert row.shape == (model.trunk_dim,)
    cfg.rl.observation_mode = "forecast"
    assert runtime_trunk_embedding(cfg, model, window) is None
