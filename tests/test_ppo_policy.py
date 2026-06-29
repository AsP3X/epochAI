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
