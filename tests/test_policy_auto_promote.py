"""Integration-style tests for policy auto-train and promotion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from epoch_ai.execution.policy.env import TradingReplayEnv
from epoch_ai.execution.policy.ppo_policy import PPOPolicy, TrainStats
from epoch_ai.learning.policy_promotion import (
    PolicyPromoteResult,
    ReplayMetrics,
    _joint_prediction_regression_reason,
    auto_train_and_promote_policy,
)
from tests.test_policy_env import _policy_config

pytest.importorskip("torch")


def test_joint_prediction_regression_reason_brier_and_auc():
    assert _joint_prediction_regression_reason(
        base_brier=0.20,
        cand_brier=0.21,
        base_auc=0.60,
        cand_auc=0.59,
        brier_tolerance=0.02,
        auc_tolerance=0.02,
    ) is None

    brier_reason = _joint_prediction_regression_reason(
        base_brier=0.20,
        cand_brier=0.25,
        base_auc=0.60,
        cand_auc=0.59,
        brier_tolerance=0.02,
        auc_tolerance=0.02,
    )
    assert brier_reason is not None
    assert "Brier" in brier_reason

    auc_reason = _joint_prediction_regression_reason(
        base_brier=0.20,
        cand_brier=0.21,
        base_auc=0.60,
        cand_auc=0.50,
        brier_tolerance=0.02,
        auc_tolerance=0.02,
    )
    assert auc_reason is not None
    assert "AUC" in auc_reason


def test_ppo_load_rejects_obs_dim_mismatch(tmp_path):
    config = _policy_config()
    policy = PPOPolicy(10, config.rl)
    path = tmp_path / "policy.pt"
    policy.save(path)

    with pytest.raises(ValueError, match="obs_dim"):
        PPOPolicy.load(path, config.rl, expected_obs_dim=20)


def test_auto_promote_vetoes_joint_brier_regression(small_config, tmp_path, market):
    cfg = small_config
    cfg.rl.promotion.champion_path = str(tmp_path / "champion.pt")
    cfg.rl.policy_path = str(tmp_path / "challenger.pt")
    cfg.rl.promotion.eval_bars = 80
    cfg.walk_forward.initial_train_period = 200

    close = market["close"].astype(float)
    n = len(close)

    challenger = PPOPolicy(13, cfg.rl)  # forecast obs dim for small_config
    work_model = MagicMock(name="work_model")
    champion_model = MagicMock(name="champion_model")
    stats = TrainStats(updates=1, mean_reward=0.01, final_equity=10_100.0)

    good_metrics = ReplayMetrics(
        total_return=0.2,
        sharpe=1.0,
        risk_adjusted_return=0.5,
        max_drawdown=0.1,
        final_equity=12_000.0,
    )

    def _fake_env(_config, market_slice, _model):
        close = market_slice["close"].astype(float).iloc[-200:]
        return TradingReplayEnv.from_market(_config, pd.DataFrame({"close": close}))

    with (
        patch(
            "epoch_ai.learning.policy_promotion.HistoricalDownloader"
        ) as mock_dl,
        patch(
            "epoch_ai.learning.policy_promotion._load_multi_head_champion",
            return_value=champion_model,
        ),
        patch(
            "epoch_ai.learning.policy_promotion.train_challenger_policy",
            return_value=(challenger, work_model, stats),
        ),
        patch(
            "epoch_ai.learning.policy_promotion._build_policy_env",
            side_effect=_fake_env,
        ),
        patch(
            "epoch_ai.learning.policy_promotion.replay_metrics",
            return_value=good_metrics,
        ),
        patch(
            "epoch_ai.learning.policy_promotion._holdout_predictor_quality",
            side_effect=[(0.20, 0.60), (0.25, 0.58)],
        ),
    ):
        mock_dl.return_value.load_or_download.return_value = market.iloc[: n - 1]
        result = auto_train_and_promote_policy(cfg, n_bars=n - 1)

    assert isinstance(result, PolicyPromoteResult)
    assert result.promoted is False
    assert result.reason is not None
    assert "Brier" in result.reason


def test_auto_promote_registers_joint_tcn_on_success(small_config, tmp_path, market):
    cfg = small_config
    cfg.rl.promotion.champion_path = str(tmp_path / "champion.pt")
    cfg.rl.policy_path = str(tmp_path / "challenger.pt")
    cfg.rl.promotion.eval_bars = 80
    cfg.walk_forward.initial_train_period = 200
    cfg.model.model_dir = str(tmp_path / "models")

    n = len(market)
    challenger = PPOPolicy(13, cfg.rl)
    work_model = MagicMock(name="work_model")
    champion_model = MagicMock(name="champion_model")
    stats = TrainStats(updates=1, mean_reward=0.01, final_equity=10_100.0)

    good_metrics = ReplayMetrics(
        total_return=0.2,
        sharpe=1.0,
        risk_adjusted_return=0.5,
        max_drawdown=0.1,
        final_equity=12_000.0,
    )

    mock_registry = MagicMock()
    mock_registry.save.return_value = "v_2"

    def _fake_env(_config, market_slice, _model):
        close = market_slice["close"].astype(float).iloc[-200:]
        return TradingReplayEnv.from_market(_config, pd.DataFrame({"close": close}))

    with (
        patch(
            "epoch_ai.learning.policy_promotion.HistoricalDownloader"
        ) as mock_dl,
        patch(
            "epoch_ai.learning.policy_promotion._load_multi_head_champion",
            return_value=champion_model,
        ),
        patch(
            "epoch_ai.learning.policy_promotion.train_challenger_policy",
            return_value=(challenger, work_model, stats),
        ),
        patch(
            "epoch_ai.learning.policy_promotion._build_policy_env",
            side_effect=_fake_env,
        ),
        patch(
            "epoch_ai.learning.policy_promotion.replay_metrics",
            return_value=good_metrics,
        ),
        patch(
            "epoch_ai.learning.policy_promotion._holdout_predictor_quality",
            side_effect=[(0.20, 0.60), (0.19, 0.61)],
        ),
        patch(
            "epoch_ai.learning.policy_promotion.ModelRegistry",
            return_value=mock_registry,
        ),
    ):
        mock_dl.return_value.load_or_download.return_value = market.iloc[: n - 1]
        result = auto_train_and_promote_policy(cfg, n_bars=n - 1)

    assert result.promoted is True
    mock_registry.save.assert_called_once_with(
        work_model,
        metadata={"source": "joint_trunk_policy_promotion"},
        retain_versions=cfg.model.retain_versions,
    )
    mock_registry.set_promoted.assert_called_once()
