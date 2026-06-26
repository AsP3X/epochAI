"""Tests for the evolved neural-network backend."""

from __future__ import annotations

import numpy as np
import pytest

from epoch_ai.features.pipeline import FeaturePipeline, build_target
from epoch_ai.models.factory import build_model, model_class
from epoch_ai.models.registry import ModelRegistry

pytest.importorskip("torch")

from epoch_ai.models.evolved_nn_model import EvolvedNNModel  # noqa: E402


def _xy(market, config):
    features = FeaturePipeline(config).transform(market)
    target = build_target(market, config.prediction)
    data = features.join(target).dropna()
    return data[features.columns], data["target"]


def _evolved_config(small_config):
    small_config.model.backend = "evolved_nn"
    small_config.model.evolution.fast_fit = True
    small_config.model.nn.max_epochs = 40
    small_config.model.nn.patience = 5
    small_config.model.val_fraction = 0.2
    small_config.model.calibration = "isotonic"
    return small_config


def test_factory_builds_evolved_nn(small_config):
    small_config.model.backend = "evolved_nn"
    model = build_model(small_config.model)
    assert isinstance(model, EvolvedNNModel)


def test_fit_predict_save_load(market, small_config, tmp_path):
    cfg = _evolved_config(small_config)
    x, y = _xy(market, cfg)
    model = EvolvedNNModel(cfg.model, task="classification")
    model.fit(x.iloc[:1500], y.iloc[:1500])

    preds = model.predict(x.iloc[1500:1600])
    assert preds.shape[0] == 100
    assert ((preds >= 0) & (preds <= 1)).all()

    path = tmp_path / "model.pt"
    model.save(str(path))
    loaded = EvolvedNNModel.load(str(path), cfg.model)
    reloaded = loaded.predict(x.iloc[1500:1600])
    assert np.allclose(preds, reloaded, atol=1e-5)


def test_registry_roundtrip(market, small_config, tmp_path):
    cfg = _evolved_config(small_config)
    x, y = _xy(market, cfg)
    model = EvolvedNNModel(cfg.model).fit(x.iloc[:1500], y.iloc[:1500])
    registry = ModelRegistry(str(tmp_path / "models"))
    label = registry.save(model, metadata={"train_rows": 1500})
    loaded, meta = registry.load(label, cfg.model)
    assert meta["backend"] == "evolved_nn"
    assert meta["model_file"] == "model.pt"
    preds = loaded.predict(x.iloc[1500:1600])
    assert ((preds >= 0) & (preds <= 1)).all()


def test_calibration_persisted(market, small_config, tmp_path):
    cfg = _evolved_config(small_config)
    x, y = _xy(market, cfg)
    model = EvolvedNNModel(cfg.model).fit(x.iloc[:2000], y.iloc[:2000])
    assert model.calibrator_ is not None

    path = tmp_path / "model.pt"
    model.save(str(path))
    assert path.with_name(path.name + ".calibration.json").exists()
    loaded = EvolvedNNModel.load(str(path), cfg.model)
    assert loaded.calibrator_ is not None


def test_feature_importance_non_empty(market, small_config):
    cfg = _evolved_config(small_config)
    x, y = _xy(market, cfg)
    model = EvolvedNNModel(cfg.model).fit(x.iloc[:2000], y.iloc[:2000])
    importance = model.feature_importance()
    assert len(importance) == x.shape[1]
    assert importance.sum() >= 0.0


def test_evolution_runs_without_fast_fit(market, small_config):
    """Evolution path completes with a small search budget."""
    cfg = _evolved_config(small_config)
    cfg.model.evolution.fast_fit = False
    cfg.model.evolution.population_size = 4
    cfg.model.evolution.generations = 1
    cfg.model.nn.max_epochs = 12
    cfg.model.nn.patience = 3
    x, y = _xy(market, cfg)

    model = EvolvedNNModel(cfg.model).fit(x.iloc[:1200], y.iloc[:1200])
    assert model.genome_ is not None
    assert model.best_iteration_ is not None


def test_model_class_lazy_import():
    cls = model_class("evolved_nn")
    assert cls.BACKEND == "evolved_nn"
