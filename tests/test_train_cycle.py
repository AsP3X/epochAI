"""Tests for the train-cycle orchestration loop."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from epoch_ai.learning.train_cycle import (
    TrainCycleOptions,
    run_single_train_cycle,
    run_train_cycle_loop,
)
from tests.test_policy_env import _policy_config


def test_run_train_cycle_loop_respects_max_cycles():
    config = _policy_config()
    options = TrainCycleOptions(minutes=0, max_cycles=2, skip_download=True, skip_run=True)

    ok_iteration = MagicMock(ok=True, cycle=1, steps=[])

    with patch(
        "epoch_ai.learning.train_cycle.run_single_train_cycle",
        return_value=ok_iteration,
    ) as mock_run:
        summary = run_train_cycle_loop(config, options)

    assert summary.cycles_completed == 2
    assert summary.stopped_reason == "max_cycles=2"
    assert mock_run.call_count == 2


def test_run_train_cycle_loop_stops_on_minutes():
    config = _policy_config()
    options = TrainCycleOptions(
        minutes=0.0001,
        max_cycles=None,
        skip_download=True,
        skip_run=True,
        interval_minutes=0,
    )

    ok_iteration = MagicMock(ok=True, cycle=1, steps=[])

    with patch(
        "epoch_ai.learning.train_cycle.run_single_train_cycle",
        return_value=ok_iteration,
    ):
        summary = run_train_cycle_loop(config, options)

    assert summary.cycles_completed >= 1
    assert "minutes=" in summary.stopped_reason


def test_run_train_cycle_loop_stops_on_failure():
    config = _policy_config()
    options = TrainCycleOptions(minutes=0, max_cycles=5, skip_download=True, skip_run=True)

    fail_iteration = MagicMock(ok=False, cycle=1, steps=[MagicMock(name="train", ok=False)])

    with patch(
        "epoch_ai.learning.train_cycle.run_single_train_cycle",
        return_value=fail_iteration,
    ):
        summary = run_train_cycle_loop(config, options)

    assert summary.cycles_completed == 1
    assert "failed" in summary.stopped_reason


def test_single_cycle_skips_download_when_requested():
    config = _policy_config()
    options = TrainCycleOptions(skip_download=True, skip_run=True)

    with (
        patch("epoch_ai.learning.train_cycle.TrainingService") as mock_svc_cls,
        patch(
            "epoch_ai.learning.train_cycle.evaluate_holdout",
            return_value=MagicMock(skipped=False, holdout_bars=50, predictor_metrics={}, policy_champion={}),
        ),
        patch(
            "epoch_ai.learning.train_cycle._run_policy_training",
            return_value=(True, "policy ok"),
        ),
    ):
        mock_svc = mock_svc_cls.return_value
        mock_svc.train.return_value = MagicMock(
            model_version="v_1",
            walk_forward_steps=3,
        )
        result = run_single_train_cycle(config, options, cycle=1)

    assert result.ok
    step_names = [s.name for s in result.steps]
    assert step_names[0] == "download"
    assert "skipped" in result.steps[0].detail
    mock_svc.train.assert_called_once()


def test_cli_train_cycle_help(capsys):
    from epoch_ai.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["train-cycle", "--help"])
    assert exc.value.code == 0
    assert "train-cycle" in capsys.readouterr().out
