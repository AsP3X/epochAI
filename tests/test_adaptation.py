"""Tests for action-log feedback and policy promotion gates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epoch_ai.execution.action_log import ActionLog, boost_weights_from_action_log, load_frame
from epoch_ai.execution.risk import RiskDecision
from epoch_ai.learning.adaptation import trim_training_rows, with_coarse_walk_forward
from epoch_ai.learning.policy_promotion import decide_policy_promotion
from tests.test_policy_env import _policy_config

pytest.importorskip("torch")


def test_action_log_round_trip(tmp_path):
    path = tmp_path / "action.jsonl"
    log = ActionLog(path)
    log.log_step(
        timestamp="2020-01-01T00:00:00",
        symbol="BTC/USDT",
        model_version="v_1",
        policy_backend="baseline",
        raw_prediction=0.55,
        decision=RiskDecision(1, 0.6, 0.5),
        equity=10_000.0,
        position_weight=0.5,
        bar_return=0.001,
    )
    frame = load_frame(path)
    assert len(frame) == 1
    assert frame.iloc[0]["decision_signal"] == 1


def test_boost_weights_from_action_log():
    weights = np.ones(4, dtype=float)
    action_df = pd.DataFrame({"timestamp": ["a", "c"]})
    boosted = boost_weights_from_action_log(
        weights,
        pd.Index(["a", "b", "c", "d"]),
        action_df,
        boost=2.0,
    )
    assert boosted[0] == 2.0
    assert boosted[1] == 1.0
    assert boosted[2] == 2.0


def test_trim_training_rows_respects_holdout():
    config = _policy_config(
        promotion={"eval_bars": 100},
        adaptation={"holdout_bars": 50},
    )
    assert trim_training_rows(config, 500) == 450


def test_with_coarse_walk_forward():
    config = _policy_config(adaptation={"enabled": True, "coarse_step_size": 999})
    coarse = with_coarse_walk_forward(config)
    assert coarse.walk_forward.step_size == 999


def test_decide_policy_promotion_requires_benchmark_beats():
    promote, _ = decide_policy_promotion(
        challenger_value=0.5,
        champion_value=0.4,
        baseline_value=0.6,
        buy_hold_value=0.3,
        metric="risk_adjusted_return",
        min_improvement=0.0,
        require_beat_baseline=True,
        require_beat_buy_hold=True,
    )
    assert promote is False

    promote, reason = decide_policy_promotion(
        challenger_value=0.8,
        champion_value=0.5,
        baseline_value=0.6,
        buy_hold_value=0.4,
        metric="risk_adjusted_return",
        min_improvement=0.05,
        require_beat_baseline=True,
        require_beat_buy_hold=True,
    )
    assert promote is True
    assert "improves" in reason


def test_promotion_requires_absolute_floor():
    # Bootstrap (no champion) below the absolute floor must NOT promote a losing policy.
    promote, reason = decide_policy_promotion(
        challenger_value=-0.2,
        champion_value=None,
        baseline_value=0.0,
        buy_hold_value=0.0,
        metric="risk_adjusted_return",
        min_improvement=0.0,
        require_beat_baseline=False,
        require_beat_buy_hold=False,
        min_absolute_metric=0.0,
    )
    assert promote is False
    assert "floor" in reason.lower() or "absolute" in reason.lower()

    # Bootstrap above the floor promotes.
    promote, _ = decide_policy_promotion(
        challenger_value=0.15,
        champion_value=None,
        baseline_value=0.0,
        buy_hold_value=0.0,
        metric="risk_adjusted_return",
        min_improvement=0.0,
        require_beat_baseline=False,
        require_beat_buy_hold=False,
        min_absolute_metric=0.0,
    )
    assert promote is True

    # The floor also applies when a champion exists and is otherwise beaten.
    promote, _ = decide_policy_promotion(
        challenger_value=-0.05,
        champion_value=-0.10,
        baseline_value=0.0,
        buy_hold_value=0.0,
        metric="risk_adjusted_return",
        min_improvement=0.0,
        require_beat_baseline=False,
        require_beat_buy_hold=False,
        min_absolute_metric=0.0,
    )
    assert promote is False


def test_baseline_beat_is_report_only_by_default():
    # A challenger that beats the champion and clears the floor is promoted even when it
    # does NOT beat baseline/buy-hold, given the new report-only (False) defaults.
    promote, reason = decide_policy_promotion(
        challenger_value=0.30,
        champion_value=0.20,
        baseline_value=0.90,  # baseline is better, but no longer a gate
        buy_hold_value=0.80,  # buy-hold is better, but no longer a gate
        metric="risk_adjusted_return",
        min_improvement=0.0,
        require_beat_baseline=False,
        require_beat_buy_hold=False,
        min_absolute_metric=0.0,
    )
    assert promote is True
    assert "improves" in reason


@pytest.mark.slow
def test_retrain_with_action_log_smoke(small_config, tmp_path):
    from epoch_ai.learning.retrain_job import run_retrain

    small_config.model.model_dir = str(tmp_path / "models")
    small_config.data.data_dir = str(tmp_path / "data")
    small_config.logging.db_path = str(tmp_path / "empty.sqlite")
    small_config.walk_forward.recency_half_life = 1000  # enable weights so boost applies
    small_config.adaptation.use_action_log_for_retrain = True
    small_config.adaptation.action_log_min_rows = 1
    small_config.adaptation.holdout_bars = 50
    small_config.promotion.eval_bars = 50

    log_path = tmp_path / "action.jsonl"
    log = ActionLog(log_path)
    log.log_step(
        timestamp="2020-01-01T00:00:00",
        symbol=small_config.primary_symbol,
        model_version="v_1",
        policy_backend="baseline",
        raw_prediction=0.5,
        decision=RiskDecision(0, 0.0, 0.0),
        equity=10_000.0,
        position_weight=0.0,
    )
    small_config.trading.action_log_path = str(log_path)

    result = run_retrain(small_config, min_new_samples=999, register=True, n_bars=4000)
    assert not result.skipped
    assert result.train_rows > 0
