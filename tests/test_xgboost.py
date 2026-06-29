"""Tests for the optional XGBoost backend, the model factory and registry dispatch.

The factory/config tests run everywhere; the tests that actually train XGBoost are
skipped when the optional ``xgboost`` package is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from epoch_ai.features.pipeline import FeaturePipeline, build_target
from epoch_ai.models.factory import build_model, model_class, model_filename
from epoch_ai.models.lightgbm_model import LightGBMModel
from epoch_ai.models.registry import ModelRegistry

xgb = pytest.importorskip("xgboost")
from epoch_ai.models.xgboost_model import XGBoostModel  # noqa: E402


def _xy(market, config):
    features = FeaturePipeline(config).transform(market)
    target = build_target(market, config.prediction)
    data = features.join(target).dropna()
    return data[features.columns], data["target"]


# --------------------------------------------------------------------- factory
def test_factory_builds_each_backend(small_config):
    assert isinstance(build_model(small_config.model), LightGBMModel)
    small_config.model.backend = "xgboost"
    assert isinstance(build_model(small_config.model), XGBoostModel)


def test_model_filename_per_backend():
    assert model_filename("lightgbm") == "model.txt"
    assert model_filename("xgboost") == "model.json"


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="Unknown model backend"):
        model_class("does-not-exist")


# ------------------------------------------------------------------- training
def test_xgb_fit_predict_save_load(market, small_config, tmp_path):
    small_config.model.backend = "xgboost"
    x, y = _xy(market, small_config)
    model = build_model(small_config.model, task="classification")
    model.fit(x.iloc[:1500], y.iloc[:1500])

    preds = model.predict(x.iloc[1500:1600])
    assert preds.shape[0] == 100
    assert ((preds >= 0) & (preds <= 1)).all()

    path = tmp_path / "model.json"
    model.save(str(path))
    loaded = XGBoostModel.load(str(path), small_config.model)
    reloaded = loaded.predict(x.iloc[1500:1600])
    assert loaded.best_iteration_ == model.best_iteration_
    assert np.allclose(preds, reloaded, atol=1e-5)


def test_xgb_feature_importance(market, small_config):
    small_config.model.backend = "xgboost"
    x, y = _xy(market, small_config)
    model = build_model(small_config.model).fit(x.iloc[:1500], y.iloc[:1500])
    importance = model.feature_importance()
    assert len(importance) == x.shape[1]
    assert (importance >= 0).all()


def test_xgb_regression_task(market, small_config):
    small_config.model.backend = "xgboost"
    small_config.prediction.task = "regression"
    x, y = _xy(market, small_config)
    model = build_model(small_config.model, task="regression")
    model.fit(x.iloc[:1500], y.iloc[:1500])
    preds = model.predict(x.iloc[1500:1600])
    assert preds.shape[0] == 100
    assert np.isfinite(preds).all()


def test_xgb_calibration_persisted(market, small_config, tmp_path):
    small_config.model.backend = "xgboost"
    small_config.model.early_stopping_rounds = 20
    small_config.model.calibration = "isotonic"
    x, y = _xy(market, small_config)
    model = build_model(small_config.model, task="classification")
    model.fit(x.iloc[:1600], y.iloc[:1600])
    assert model.calibrator_ is not None

    path = tmp_path / "model.json"
    model.save(str(path))
    assert (tmp_path / "model.json.calibration.json").exists()
    loaded = XGBoostModel.load(str(path), small_config.model)
    assert loaded.calibrator_ is not None
    assert np.allclose(
        model.predict(x.iloc[1600:1700]),
        loaded.predict(x.iloc[1600:1700]),
        atol=1e-6,
    )


def test_xgb_gpu_request_falls_back_to_cpu(market, small_config):
    """Requesting CUDA where no GPU exists must still train (graceful fallback)."""
    small_config.model.backend = "xgboost"
    small_config.model.device = "cuda"
    x, y = _xy(market, small_config)
    model = build_model(small_config.model, task="classification")
    model.fit(x.iloc[:1500], y.iloc[:1500])  # must not raise even without a GPU
    preds = model.predict(x.iloc[1500:1600])
    assert ((preds >= 0) & (preds <= 1)).all()


def test_xgb_cuda_failure_retries_on_cpu(market, small_config, monkeypatch):
    """A CUDA XGBoostError is caught and retried on CPU (deterministic fallback)."""
    import epoch_ai.models.xgboost_model as mod

    small_config.model.backend = "xgboost"
    small_config.model.device = "cuda"
    x, y = _xy(market, small_config)
    real_train = mod.xgb.train
    calls: list[str] = []

    def fake_train(params, *args, **kwargs):
        device = str(params.get("device", "cpu"))
        calls.append(device)
        if not device.startswith("cpu"):
            raise mod.xgb.core.XGBoostError("CUDA driver not found")
        return real_train(params, *args, **kwargs)

    monkeypatch.setattr(mod.xgb, "train", fake_train)

    model = build_model(small_config.model, task="classification")
    model.fit(x.iloc[:1200], y.iloc[:1200])
    preds = model.predict(x.iloc[1200:1260])

    assert calls[0].startswith("cuda")  # first attempt requested CUDA
    assert calls[-1] == "cpu"           # and it fell back to CPU
    assert ((preds >= 0) & (preds <= 1)).all()


def test_xgb_device_resolution(small_config):
    small_config.model.backend = "xgboost"
    small_config.model.device = "cpu"
    assert XGBoostModel(small_config.model)._device() == "cpu"

    small_config.model.device = "gpu"
    assert XGBoostModel(small_config.model)._device() == "cuda"

    small_config.model.device = "cuda"
    small_config.model.gpu_device_id = 1
    assert XGBoostModel(small_config.model)._device() == "cuda:1"


# ------------------------------------------------------------------- registry
def test_registry_roundtrip_xgboost(market, small_config, tmp_path):
    """A registered XGBoost model is reloaded as XGBoost via metadata dispatch."""
    small_config.model.backend = "xgboost"
    small_config.model.model_dir = str(tmp_path / "models")
    x, y = _xy(market, small_config)
    model = build_model(small_config.model, task="classification").fit(
        x.iloc[:1500], y.iloc[:1500]
    )

    registry = ModelRegistry(small_config.model.model_dir)
    label = registry.save(model)

    meta = registry.list_versions()[-1]
    assert meta["backend"] == "xgboost"
    assert meta["model_file"] == "model.json"

    loaded, _ = registry.load(label, small_config.model, task="classification")
    assert isinstance(loaded, XGBoostModel)
    assert np.allclose(
        model.predict(x.iloc[1500:1600]),
        loaded.predict(x.iloc[1500:1600]),
        atol=1e-5,
    )
