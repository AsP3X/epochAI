"""Tests for walk-forward training progress reporting."""

from __future__ import annotations

from argparse import Namespace

import pytest

from epoch_ai.cli import cmd_progress
from epoch_ai.features.pipeline import FeaturePipeline, build_target, forward_return
from epoch_ai.learning.checkpoint import build_checkpoint, save_checkpoint
from epoch_ai.learning.progress_report import (
    estimate_total_walk_forward_steps,
    format_training_progress,
    gather_training_progress,
)
from epoch_ai.learning.progressive import ProgressiveLearningEngine


def _resolved_rows(market, config) -> int:
    features = FeaturePipeline(config).transform(market)
    y = build_target(market, config.prediction)
    fwd = forward_return(market, config.prediction.horizon)
    return len(features.join(y).join(fwd).dropna(subset=["target", "forward_return"]))


@pytest.mark.slow
def test_estimate_total_steps_matches_engine(market, small_config, tmp_path):
    wf = small_config.walk_forward
    small_config.walk_forward.max_steps = 3
    small_config.walk_forward.checkpoint_path = str(tmp_path / "wf.json")
    n = _resolved_rows(market, small_config)
    expected = estimate_total_walk_forward_steps(
        n,
        initial_train_period=wf.initial_train_period,
        step_size=wf.step_size,
        max_steps=3,
    )
    result = ProgressiveLearningEngine(small_config, register_models=False).run(
        market,
        FeaturePipeline(small_config).transform(market),
        resume=False,
        fresh=True,
    )
    assert expected == len(result.step_history)


def test_gather_progress_from_checkpoint(market, small_config, tmp_path):
    wf = small_config.walk_forward
    small_config.walk_forward.max_steps = None
    n = _resolved_rows(market, small_config)
    checkpoint_path = tmp_path / "wf.json"
    small_config.walk_forward.checkpoint_path = str(checkpoint_path)
    small_config.model.model_dir = str(tmp_path / "models")

    next_step = 3
    cutoff = wf.initial_train_period + next_step * wf.step_size
    state = build_checkpoint(
        step_idx=next_step,
        cutoff=cutoff,
        model_version="v_3",
        config=small_config,
        n_features=10,
        resolved_rows=n,
    )
    save_checkpoint(checkpoint_path, state)

    report = gather_training_progress(small_config)
    total = estimate_total_walk_forward_steps(
        n,
        initial_train_period=wf.initial_train_period,
        step_size=wf.step_size,
    )
    assert report.status == "in_progress"
    assert report.completed_steps == next_step
    assert report.total_steps == total
    assert report.remaining_steps == total - next_step
    assert report.next_step_idx == next_step
    assert report.cutoff == cutoff
    assert report.model_version == "v_3"

    text = format_training_progress(report)
    assert f"{next_step} /" in text
    assert "Steps remaining" in text
    assert "Resume with:" in text


def test_infer_progress_from_registry_when_no_checkpoint(small_config, tmp_path):
    import json

    model_dir = tmp_path / "models" / "v_5"
    model_dir.mkdir(parents=True)
    (model_dir / "model.txt").write_text("stub", encoding="utf-8")
    (model_dir / "metadata.json").write_text(
        json.dumps({"step": 4, "n_features": 8, "backend": "lightgbm"}),
        encoding="utf-8",
    )
    small_config.model.model_dir = str(tmp_path / "models")
    small_config.walk_forward.checkpoint_path = str(tmp_path / "missing.json")

    report = gather_training_progress(small_config)
    assert report.inferred_from_registry
    assert report.completed_steps == 5
    assert report.next_step_idx == 5
    assert report.model_version == "v_5"
    text = format_training_progress(report)
    assert "inferred from registry" in text
    assert "checkpoint seed --last-step 4" in text


def test_build_fraction_bar():
    from epoch_ai.utils.progress import build_fraction_bar

    bar = build_fraction_bar(5, 10, width=10)
    assert bar.startswith("[")
    assert bar.endswith("]")
    assert len(bar) == 12


def test_watch_training_progress_exits_on_interrupt(small_config, tmp_path, monkeypatch):
    from epoch_ai.learning.progress_report import watch_training_progress

    checkpoint_path = tmp_path / "wf.json"
    small_config.walk_forward.checkpoint_path = str(checkpoint_path)
    small_config.walk_forward.max_steps = None
    save_checkpoint(
        checkpoint_path,
        build_checkpoint(
            step_idx=2,
            cutoff=1600,
            model_version="v_2",
            config=small_config,
            n_features=8,
            resolved_rows=4000,
        ),
    )

    rendered: list[str] = []

    def capture(text: str, **kwargs) -> None:
        rendered.append(text)

    monkeypatch.setattr(
        "epoch_ai.utils.progress.render_live_text",
        capture,
    )
    monkeypatch.setattr(
        "epoch_ai.learning.progress_report.time.sleep",
        lambda _interval: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    code = watch_training_progress(small_config, interval=0.1)
    assert code == 130
    assert rendered
    assert "(live)" in rendered[0]
    assert "Press Ctrl+C to exit." in rendered[0]


def test_progress_cli_uses_checkpoint(tmp_path, small_config, capsys):
    checkpoint_path = tmp_path / "wf.json"
    small_config.walk_forward.checkpoint_path = str(checkpoint_path)
    save_checkpoint(
        checkpoint_path,
        build_checkpoint(
            step_idx=42,
            cutoff=10400,
            model_version="v_43",
            config=small_config,
            n_features=8,
            resolved_rows=5000,
        ),
    )

    import epoch_ai.cli as cli_mod

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cli_mod, "_load", lambda _args: small_config)
    try:
        code = cmd_progress(
            Namespace(
                config="config/config.yaml",
                set=[],
                symbol=None,
                bars=None,
                refresh_rows=False,
                max_steps=None,
                watch=False,
                interval=2.0,
            )
        )
    finally:
        monkeypatch.undo()

    assert code == 0
    out = capsys.readouterr().out
    assert "Walk-forward training progress" in out
    assert "42 /" in out
