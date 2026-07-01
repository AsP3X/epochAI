"""Tests for PPO policy train/save/load/act."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epoch_ai.execution.policy.env import TradingReplayEnv
from epoch_ai.execution.policy.observation import observation_dim
from tests.test_policy_env import _policy_config

pytest.importorskip("torch")

from epoch_ai.execution.policy.ppo_policy import PPOPolicy  # noqa: E402


def test_ppo_train_save_load_act(tmp_path):
    config = _policy_config(
        rl={
            "total_updates": 2,
            "rollout_steps": 32,
            "device": "cpu",
            "policy_path": str(tmp_path / "ppo_policy.pt"),
        }
    )
    close = pd.Series(100.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.1, 300)))
    market = pd.DataFrame({"close": close})
    env = TradingReplayEnv.from_market(config, market.iloc[100:])

    policy = PPOPolicy(observation_dim(config), config.rl)
    stats = policy.train(env)
    assert stats.updates == 2

    path = tmp_path / "ppo_policy.pt"
    policy.save(path)
    loaded = PPOPolicy.load(path, config.rl)
    obs = env.reset()
    a1 = policy.act(obs, deterministic=True)
    a2 = loaded.act(obs, deterministic=True)
    assert -1.0 <= a1 <= 1.0
    assert abs(a1 - a2) < 1e-5


def test_ppo_act_is_deterministic():
    config = _policy_config(
        rl={"total_updates": 1, "rollout_steps": 8, "device": "cpu"}
    )
    policy = PPOPolicy(observation_dim(config), config.rl)
    obs = np.zeros(observation_dim(config), dtype=np.float32)
    assert policy.act(obs, deterministic=True) == policy.act(obs, deterministic=True)


def test_policy_trains_on_real_forecasts(market, small_config, tmp_path):
    # Human: policy training must consume the champion model's REAL per-bar forecasts,
    #        not the price-only tanh proxy. We register a tiny multi-head TCN champion,
    #        build the training env via the helper, and assert it exposes the real
    #        structured forecasts (model_version == "replay-real").
    # Agent: CAUSAL; from_forecasts shifts returns -1 internally. Fails pre-impl because
    #        the helper/loader do not exist yet (real-forecast wiring absent).
    from epoch_ai.features.pipeline import (
        FeaturePipeline,
        build_multi_horizon_targets,
        build_target,
    )
    from epoch_ai.learning.policy_promotion import (
        _build_policy_env_from_model,
        _load_multi_head_champion,
    )
    from epoch_ai.models.registry import ModelRegistry
    from epoch_ai.models.tcn_model import TCNModel

    cfg = small_config
    cfg.model.model_dir = str(tmp_path / "models")
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

    # Build a small multi-head training frame (features + per-horizon targets).
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
    ModelRegistry(cfg.model.model_dir).save(model, metadata={"train_rows": 2400})

    # The promoted champion loads back as a usable multi-head model.
    champion = _load_multi_head_champion(cfg)
    assert champion is not None
    assert champion.multi_head_spec_ is not None

    # The env built for policy training carries the real structured forecasts.
    env = _build_policy_env_from_model(cfg, market.iloc[:2400], champion)
    assert env.structured_forecasts is not None
    env.reset()
    assert env.current_forecast().model_version == "replay-real"
