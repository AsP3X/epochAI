"""Tests for tune promote."""

from __future__ import annotations

import json

from epoch_ai.tuning.promote import find_best_experiment, promote_best


def test_find_best_experiment(tmp_path):
    for name, sharpe in [("baseline", 0.5), ("better", 1.2), ("worse", -0.1)]:
        exp_dir = tmp_path / name
        exp_dir.mkdir()
        (exp_dir / "metrics.json").write_text(
            json.dumps({"strategy": {"sharpe": sharpe}, "overrides": {"timeframe": name}}),
            encoding="utf-8",
        )
    best, value, overrides = find_best_experiment(tmp_path, metric="sharpe")
    assert best == "better"
    assert value == 1.2
    assert overrides["timeframe"] == "better"


def test_promote_writes_config(tmp_path):
    base_cfg = tmp_path / "base.yaml"
    base_cfg.write_text("symbols: [BTC/USDT]\ntimeframe: 15m\n", encoding="utf-8")
    sweep = tmp_path / "sweeps"
    exp = sweep / "fast_steps"
    exp.mkdir(parents=True)
    (exp / "metrics.json").write_text(
        json.dumps(
            {
                "strategy": {"sharpe": 2.0},
                "overrides": {"walk_forward": {"step_size": 100}},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "promoted.yaml"
    result = promote_best(base_cfg, sweep, dest=out)
    assert result.experiment == "fast_steps"
    assert out.exists()
    assert "step_size: 100" in out.read_text(encoding="utf-8")
