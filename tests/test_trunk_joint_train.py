"""Tests for joint / staged shared-trunk policy training (Task 10)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from epoch_ai.learning.policy_promotion import _joint_brier_regression_reason
from epoch_ai.learning.trunk_joint_train import joint_trunk_enabled, train_trunk_policy
from tests.test_policy_env import _policy_config

pytest.importorskip("torch")


def test_joint_trunk_enabled_defaults_false():
    config = _policy_config(rl={"observation_mode": "embedding"})
    assert joint_trunk_enabled(config) is False


def test_joint_trunk_enabled_when_stage_two_configured():
    config = _policy_config(
        rl={
            "observation_mode": "embedding",
            "trunk_frozen": False,
            "policy_loss_weight": 0.5,
        }
    )
    assert joint_trunk_enabled(config) is True


def test_joint_brier_regression_reason():
    assert _joint_brier_regression_reason(0.20, 0.21, 0.02) is None
    reason = _joint_brier_regression_reason(0.20, 0.25, 0.02)
    assert reason is not None
    assert "regressed" in reason


def test_train_trunk_policy_frozen_skips_supervised_steps(market, small_config):
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
    cfg.rl.trunk_frozen = True
    cfg.rl.policy_loss_weight = 0.0
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

    with patch.object(TCNModel, "supervised_gradient_step", return_value=0.0) as mock_step:
        policy, work_model, stats = train_trunk_policy(
            cfg, market.iloc[:1200], model, clone_model=True
        )
    mock_step.assert_not_called()
    assert policy.obs_dim == work_model.trunk_dim + 4
    assert stats.updates >= 1
    assert work_model is not model


def test_train_trunk_policy_joint_runs_supervised_steps(market, small_config):
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
    cfg.rl.trunk_frozen = False
    cfg.rl.policy_loss_weight = 0.5
    cfg.rl.supervised_aux_steps = 1
    cfg.rl.prediction_aux_weight = 1.0
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

    with patch.object(
        TCNModel, "supervised_gradient_step", return_value=0.42
    ) as mock_step:
        _policy, _work_model, stats = train_trunk_policy(
            cfg, market.iloc[:1200], model, clone_model=True
        )
    assert mock_step.call_count == stats.updates
    assert stats.updates >= 1
