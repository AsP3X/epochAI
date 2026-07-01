"""CLI integration tests."""

from __future__ import annotations

import json

import pytest

from epoch_ai.cli import main


def test_cli_info():
    assert main(["info"]) == 0


def test_cli_info_with_set():
    assert main(["info", "--set", "timeframe=5m"]) == 0


def test_cli_train_cycle_accepts_loop_flags(tmp_path, monkeypatch):
    from unittest.mock import patch

    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "data:\n"
        "  use_synthetic_fallback: true\n"
        "  data_dir: artifacts/data\n"
        "walk_forward:\n"
        "  initial_train_period: 100\n"
    )
    with patch("epoch_ai.learning.train_cycle.run_train_cycle_loop") as mock_loop:
        from epoch_ai.learning.train_cycle import TrainCycleSummary

        mock_loop.return_value = TrainCycleSummary(
            cycles_completed=0,
            iterations=[],
            stopped_reason="minutes=10",
            elapsed_seconds=1.0,
        )
        rc = main(
            [
                "train-cycle",
                "--config",
                str(cfg),
                "--minutes",
                "0.001",
                "--max-cycles",
                "1",
                "--skip-download",
                "--skip-run",
            ]
        )
    assert rc == 1
    mock_loop.assert_called_once()


def test_cli_train_policy_accepts_observation_flags(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "data:\n"
        "  use_synthetic_fallback: true\n"
        "  data_dir: artifacts/data\n"
        "walk_forward:\n"
        "  initial_train_period: 100\n"
    )
    rc = main(
        [
            "train-policy",
            "--config",
            str(cfg),
            "--bars",
            "50",
            "--observation-mode",
            "embedding",
            "--no-trunk-frozen",
            "--policy-loss-weight",
            "0.1",
        ]
    )
    assert rc == 1


def test_cli_download(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "data:\n"
        "  use_synthetic_fallback: true\n"
        "  data_dir: artifacts/data\n"
    )
    assert main(["download", "--config", str(cfg), "--bars", "500"]) == 0


def test_cli_download_full_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
timeframe: 1h
data:
  use_synthetic_fallback: true
  data_dir: artifacts/data
  historical_start_date: "2025-01-01"
"""
    )
    assert main(["download", "--config", str(cfg), "--full-history"]) == 0


@pytest.mark.slow
def test_cli_backtest_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
symbols: ["BTC/USDT"]
data:
  use_synthetic_fallback: true
  data_dir: artifacts/data
  context_symbols: []
  synthesize_market_extensions: false
features:
  higher_timeframe: false
  macro: false
  onchain: false
  patterns: false
  manipulation: false
model:
  backend: lightgbm
  num_boost_round: 40
  early_stopping_rounds: null
walk_forward:
  initial_train_period: 800
  step_size: 400
  max_steps: 2
"""
    )
    out = tmp_path / "bt"
    code = main(
        [
            "backtest",
            "--config",
            str(cfg),
            "--bars",
            "2500",
            "--max-steps",
            "2",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert "learning_curve" in metrics
    assert (out / "learning_curve.json").exists()


def test_cli_set_invalid():
    with pytest.raises(ValueError, match="key=value"):
        main(["info", "--set", "notvalid"])


@pytest.mark.slow
def test_cli_auto_retrain_timed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
symbols: ["BTC/USDT"]
data:
  use_synthetic_fallback: true
  data_dir: artifacts/data
  context_symbols: []
  synthesize_market_extensions: false
features:
  higher_timeframe: false
  macro: false
  onchain: false
  patterns: false
  manipulation: false
model:
  backend: lightgbm
  model_dir: artifacts/models
  num_boost_round: 40
  early_stopping_rounds: null
walk_forward:
  initial_train_period: 800
promotion:
  eval_bars: 600
"""
    )
    # A tiny budget runs exactly one cycle (do-while), exercising the timed loop path.
    code = main(["auto-retrain", "--config", str(cfg), "--bars", "4000", "--minutes", "0.0001"])
    assert code == 0

    from epoch_ai.models.registry import ModelRegistry

    registry = ModelRegistry(str(tmp_path / "artifacts/models"))
    assert registry.latest_label() is not None
    assert registry.promoted_label() is not None  # bootstrap promotes the first challenger
