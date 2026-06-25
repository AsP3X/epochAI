"""Tests for training and runtime service layer."""

from __future__ import annotations

import pytest

from epoch_ai.models.registry import ModelRegistry
from epoch_ai.services.runtime import RuntimeService
from epoch_ai.services.training import TrainingService


def test_training_service_train(small_config, tmp_path):
    small_config.model.model_dir = str(tmp_path / "models")
    small_config.data.data_dir = str(tmp_path / "data")
    service = TrainingService(small_config)
    result = service.train(n_bars=2000, max_steps=2, register=True)
    assert result.model_version is not None
    assert result.walk_forward_steps >= 1
    assert len(service.list_models()) >= 1


def test_runtime_service_predict_and_run(market, small_config, tmp_path):
    small_config.model.model_dir = str(tmp_path / "models")
    small_config.data.data_dir = str(tmp_path / "data")

    train = TrainingService(small_config)
    train.train(n_bars=2000, max_steps=2, register=True)

    runtime = RuntimeService(small_config)
    pred = runtime.predict_market(market)
    assert 0.0 <= pred.raw_prediction <= 1.0
    assert pred.model_version.startswith("v_")

    result = runtime.run_session(n_bars=2000, live_bars=200, retrain_every=0)
    assert result.bars_processed > 0


def test_runtime_requires_trained_model(small_config, tmp_path):
    small_config.model.model_dir = str(tmp_path / "models")
    runtime = RuntimeService(small_config)
    assert runtime.status().models_available == 0
    with pytest.raises(FileNotFoundError):
        runtime.load_model()


def test_registry_load(small_config, tmp_path):
    small_config.model.model_dir = str(tmp_path / "models")
    small_config.data.data_dir = str(tmp_path / "data")
    TrainingService(small_config).train(n_bars=2000, max_steps=1, register=True)

    registry = ModelRegistry(small_config.model.model_dir)
    versions = registry.list_versions()
    assert len(versions) == 1
    model, meta = registry.load(None, small_config.model, task="classification")
    assert meta["label"] == versions[0]["label"]
    assert model.feature_names_ is not None
