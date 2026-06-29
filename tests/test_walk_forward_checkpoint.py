"""Tests for walk-forward training checkpoints (pause/resume)."""

from __future__ import annotations

import pytest

from epoch_ai.features.pipeline import FeaturePipeline
from epoch_ai.learning.checkpoint import (
    WalkForwardCheckpoint,
    build_checkpoint,
    checkpoint_fingerprint,
    clear_checkpoint,
    load_checkpoint,
    resolve_checkpoint_path,
    save_checkpoint,
    validate_checkpoint,
)
from epoch_ai.learning.progressive import ProgressiveLearningEngine

pytestmark = pytest.mark.slow
from epoch_ai.models.registry import ModelRegistry


def test_checkpoint_round_trip(tmp_path, small_config):
    small_config.walk_forward.checkpoint_path = str(tmp_path / "wf.json")
    state = build_checkpoint(
        step_idx=5,
        cutoff=2800,
        model_version="v_5",
        config=small_config,
        n_features=42,
        resolved_rows=4000,
    )
    path = resolve_checkpoint_path(small_config)
    save_checkpoint(path, state)
    loaded = load_checkpoint(path)
    assert loaded is not None
    assert loaded.step_idx == 5
    assert loaded.cutoff == 2800
    assert loaded.model_version == "v_5"
    assert loaded.fingerprint == checkpoint_fingerprint(small_config, 42)


def test_validate_rejects_feature_mismatch(tmp_path, small_config):
    small_config.walk_forward.checkpoint_path = str(tmp_path / "wf.json")
    state = build_checkpoint(
        step_idx=1,
        cutoff=1200,
        model_version=None,
        config=small_config,
        n_features=10,
        resolved_rows=4000,
    )
    validate_checkpoint(state, small_config, 10, 4000)
    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint(state, small_config, 11, 4000)


def test_legacy_fingerprint_still_resumes(tmp_path, small_config):
    from epoch_ai.learning.checkpoint import legacy_checkpoint_fingerprint

    small_config.walk_forward.checkpoint_path = str(tmp_path / "wf.json")
    small_config.walk_forward.retrain_frequency = 5
    legacy_fp = legacy_checkpoint_fingerprint(small_config, 10, retrain_frequency=1)
    state = WalkForwardCheckpoint(
        step_idx=3,
        cutoff=1600,
        model_version="v_3",
        fingerprint=legacy_fp,
        symbol=small_config.primary_symbol,
        resolved_rows=4000,
        updated_at="2026-01-01T00:00:00+00:00",
    )
    validate_checkpoint(state, small_config, 10, 4000)


def test_refresh_checkpoint_fingerprint(tmp_path, small_config):
    from epoch_ai.learning.checkpoint import (
        legacy_checkpoint_fingerprint,
        refresh_checkpoint_fingerprint,
    )

    path = tmp_path / "wf.json"
    small_config.walk_forward.checkpoint_path = str(path)
    legacy_fp = legacy_checkpoint_fingerprint(
        small_config, 10, retrain_frequency=1
    )
    save_checkpoint(
        path,
        WalkForwardCheckpoint(
            step_idx=50,
            cutoff=12000,
            model_version="v_50",
            fingerprint=legacy_fp,
            symbol=small_config.primary_symbol,
            resolved_rows=4000,
            updated_at="2026-01-01T00:00:00+00:00",
        ),
    )
    refreshed = refresh_checkpoint_fingerprint(path, small_config, 10)
    assert refreshed is not None
    assert refreshed.fingerprint == checkpoint_fingerprint(small_config, 10)
    assert load_checkpoint(path).fingerprint == checkpoint_fingerprint(small_config, 10)


def test_train_resume_continues_from_checkpoint(market, small_config, tmp_path):
    small_config.walk_forward.checkpoint_path = str(tmp_path / "wf.json")
    small_config.walk_forward.max_steps = 2
    small_config.model.model_dir = str(tmp_path / "models")
    features = FeaturePipeline(small_config).transform(market)

    first = ProgressiveLearningEngine(small_config, register_models=True).run(
        market,
        features,
        resume=False,
        fresh=True,
    )
    assert len(first.step_history) == 2
    assert first.step_history["step"].tolist() == [0, 1]

    checkpoint = load_checkpoint(resolve_checkpoint_path(small_config))
    assert checkpoint is not None
    assert checkpoint.step_idx == 2
    assert checkpoint.cutoff == small_config.walk_forward.initial_train_period + 2 * small_config.walk_forward.step_size

    small_config.walk_forward.max_steps = 4
    second = ProgressiveLearningEngine(small_config, register_models=True).run(
        market,
        features,
        resume=True,
        fresh=False,
    )
    assert second.resumed_from_step == 2
    assert len(second.step_history) == 2
    assert second.step_history["step"].tolist() == [2, 3]
    assert load_checkpoint(resolve_checkpoint_path(small_config)) is not None


def test_fresh_clears_checkpoint_and_restarts(market, small_config, tmp_path):
    small_config.walk_forward.checkpoint_path = str(tmp_path / "wf.json")
    small_config.walk_forward.max_steps = 1
    small_config.model.model_dir = str(tmp_path / "models")
    features = FeaturePipeline(small_config).transform(market)

    ProgressiveLearningEngine(small_config, register_models=True).run(
        market,
        features,
        resume=False,
        fresh=True,
    )
    assert load_checkpoint(resolve_checkpoint_path(small_config)) is not None

    result = ProgressiveLearningEngine(small_config, register_models=True).run(
        market,
        features,
        resume=True,
        fresh=True,
    )
    assert result.resumed_from_step is None
    assert result.step_history["step"].tolist() == [0]


def test_completed_run_clears_checkpoint(market, small_config, tmp_path):
    small_config.walk_forward.checkpoint_path = str(tmp_path / "wf.json")
    small_config.model.model_dir = str(tmp_path / "models")
    # Short history so the walk-forward exhausts data in ~3 steps (fast test).
    short = market.iloc[:1800].copy()
    small_config.walk_forward.max_steps = None
    small_config.walk_forward.initial_train_period = 800
    small_config.walk_forward.step_size = 400
    features = FeaturePipeline(small_config).transform(short)

    ProgressiveLearningEngine(small_config, register_models=False).run(
        short,
        features,
        resume=False,
        fresh=True,
    )
    assert not resolve_checkpoint_path(small_config).exists()


def test_resume_loads_registry_model(market, small_config, tmp_path):
    small_config.walk_forward.checkpoint_path = str(tmp_path / "wf.json")
    small_config.walk_forward.max_steps = 1
    model_dir = tmp_path / "models"
    small_config.model.model_dir = str(model_dir)
    features = FeaturePipeline(small_config).transform(market)

    ProgressiveLearningEngine(small_config, register_models=True).run(
        market,
        features,
        fresh=True,
    )
    checkpoint = load_checkpoint(resolve_checkpoint_path(small_config))
    assert checkpoint is not None
    assert checkpoint.model_version is not None

    registry = ModelRegistry(str(model_dir))
    model, _ = registry.load(checkpoint.model_version, small_config.model)
    assert model.feature_names_

    small_config.walk_forward.max_steps = 2
    resumed = ProgressiveLearningEngine(small_config, register_models=True).run(
        market,
        features,
        resume=True,
    )
    assert resumed.resumed_from_step == checkpoint.step_idx
    assert len(resumed.step_history) == 1


def test_clear_checkpoint_idempotent(tmp_path, small_config):
    path = tmp_path / "missing.json"
    clear_checkpoint(path)
    state = build_checkpoint(
        step_idx=0,
        cutoff=800,
        model_version=None,
        config=small_config,
        n_features=1,
        resolved_rows=1000,
    )
    save_checkpoint(path, state)
    clear_checkpoint(path)
    assert not path.exists()
    clear_checkpoint(path)


def test_train_interrupt_message(tmp_path, small_config, capsys):
    """Ctrl+C on train prints resume guidance without a traceback from cmd_train."""
    from argparse import Namespace

    from epoch_ai.cli import cmd_train

    small_config.walk_forward.checkpoint_path = str(tmp_path / "wf.json")
    save_checkpoint(
        tmp_path / "wf.json",
        build_checkpoint(
            step_idx=78,
            cutoff=17600,
            model_version="v_79",
            config=small_config,
            n_features=64,
            resolved_rows=3000,
        ),
    )

    class FakeService:
        def train(self, **_kwargs):
            raise KeyboardInterrupt

    def fake_load(_args):
        return small_config

    import epoch_ai.cli as cli_mod

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cli_mod, "TrainingService", lambda _cfg: FakeService())
    monkeypatch.setattr(cli_mod, "_load", fake_load)
    try:
        code = cmd_train(
            Namespace(
                config="config/config.yaml",
                set=[],
                symbol=None,
                max_steps=None,
                bars=None,
                log_predictions=False,
                no_register=False,
                no_resume=False,
                fresh=False,
            )
        )
    finally:
        monkeypatch.undo()

    assert code == 130
    captured = capsys.readouterr()
    assert "Training interrupted" in captured.out
    assert "step 78" in captured.out
    assert "v_79" in captured.out
