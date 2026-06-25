"""CLI integration tests."""

from __future__ import annotations

import json

import pytest

from epoch_ai.cli import main


def test_cli_info():
    assert main(["info"]) == 0


def test_cli_info_with_set():
    assert main(["info", "--set", "timeframe=5m"]) == 0


def test_cli_download(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("data:\n  data_dir: artifacts/data\n")
    assert main(["download", "--config", str(cfg), "--bars", "500"]) == 0


def test_cli_backtest_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
symbols: ["BTC/USDT"]
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
            "2000",
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
