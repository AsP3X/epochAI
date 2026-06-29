"""Tests for the LightGBM model wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from epoch_ai.features.pipeline import FeaturePipeline, build_target
from epoch_ai.models.calibration import ProbabilityCalibrator
from epoch_ai.models.lightgbm_model import LightGBMModel
from epoch_ai.models.registry import ModelRegistry


def _xy(market, config):
    features = FeaturePipeline(config).transform(market)
    target = build_target(market, config.prediction)
    data = features.join(target).dropna()
    return data[features.columns], data["target"]


def test_fit_predict_save_load(market, small_config, tmp_path):
    x, y = _xy(market, small_config)
    model = LightGBMModel(small_config.model, task="classification")
    model.fit(x.iloc[:1500], y.iloc[:1500])

    preds = model.predict(x.iloc[1500:1600])
    assert preds.shape[0] == 100
    assert ((preds >= 0) & (preds <= 1)).all()

    path = tmp_path / "model.txt"
    model.save(str(path))
    loaded = LightGBMModel.load(str(path), small_config.model)
    reloaded = loaded.predict(x.iloc[1500:1600])
    assert np.allclose(preds, reloaded, atol=1e-9)


def test_feature_importance(market, small_config):
    x, y = _xy(market, small_config)
    model = LightGBMModel(small_config.model).fit(x.iloc[:1500], y.iloc[:1500])
    importance = model.feature_importance()
    assert len(importance) == x.shape[1]
    assert (importance >= 0).all()


def test_calibration_fitted_and_persisted(market, small_config, tmp_path):
    """Early stopping + calibration enabled => calibrator survives save/load."""
    small_config.model.early_stopping_rounds = 20
    small_config.model.val_fraction = 0.2
    small_config.model.calibration = "isotonic"
    x, y = _xy(market, small_config)

    model = LightGBMModel(small_config.model, task="classification")
    model.fit(x.iloc[:2000], y.iloc[:2000])
    assert model.calibrator_ is not None

    preds = model.predict(x.iloc[2000:2100])
    assert ((preds >= 0) & (preds <= 1)).all()

    path = tmp_path / "model.txt"
    model.save(str(path))
    assert path.with_name(path.name + ".calibration.json").exists()

    loaded = LightGBMModel.load(str(path), small_config.model)
    assert loaded.calibrator_ is not None
    assert np.allclose(preds, loaded.predict(x.iloc[2000:2100]), atol=1e-9)


def test_refit_full_after_es_uses_all_rows(market, small_config):
    """Refitting on the full window (incl. the val tail) changes the deployed model."""
    small_config.model.early_stopping_rounds = 20
    small_config.model.val_fraction = 0.2
    x, y = _xy(market, small_config)

    small_config.model.refit_full_after_es = True
    refit = LightGBMModel(small_config.model, task="classification")
    refit.fit(x.iloc[:2000], y.iloc[:2000])
    preds_refit = refit.predict(x.iloc[2000:2100])

    small_config.model.refit_full_after_es = False
    split_only = LightGBMModel(small_config.model, task="classification")
    split_only.fit(x.iloc[:2000], y.iloc[:2000])
    preds_split = split_only.predict(x.iloc[2000:2100])

    assert refit.calibrator_ is not None
    assert ((preds_refit >= 0) & (preds_refit <= 1)).all()
    # Training on the extra (freshest) 20% of rows must move the predictions.
    assert not np.allclose(preds_refit, preds_split)


def test_device_params_builder(small_config):
    """The device param builder is a no-op on CPU/auto and wires ids on GPU."""
    small_config.model.device = "cpu"
    assert LightGBMModel(small_config.model)._device_params() == {}

    small_config.model.device = "auto"
    assert LightGBMModel(small_config.model)._device_params() == {}

    small_config.model.device = "gpu"
    small_config.model.gpu_platform_id = 0
    small_config.model.gpu_device_id = 1
    params = LightGBMModel(small_config.model)._device_params()
    assert params == {"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 1}

    small_config.model.device = "cuda"
    cuda = LightGBMModel(small_config.model)._device_params()
    assert cuda["device_type"] == "cuda"
    assert "gpu_platform_id" not in cuda  # platform id is OpenCL-only


def test_gpu_request_falls_back_to_cpu(market, small_config):
    """Requesting a GPU on a CPU-only build must still train (graceful fallback)."""
    small_config.model.device = "gpu"
    x, y = _xy(market, small_config)
    model = LightGBMModel(small_config.model, task="classification")
    model.fit(x.iloc[:1500], y.iloc[:1500])  # must not raise even without a GPU
    preds = model.predict(x.iloc[1500:1600])
    assert ((preds >= 0) & (preds <= 1)).all()


def test_gpu_failure_retries_on_cpu(market, small_config, monkeypatch):
    """A GPU LightGBMError is caught and retried on CPU (deterministic fallback)."""
    import epoch_ai.models.lightgbm_model as mod

    small_config.model.device = "gpu"
    x, y = _xy(market, small_config)
    real_train = mod.lgb.train
    calls: list[str] = []

    def fake_train(params, *args, **kwargs):
        device = params.get("device_type", "cpu")
        calls.append(device)
        if device != "cpu":
            raise mod.lgb.basic.LightGBMError("GPU Tree Learner was not enabled in this build")
        return real_train(params, *args, **kwargs)

    monkeypatch.setattr(mod.lgb, "train", fake_train)

    model = LightGBMModel(small_config.model, task="classification")
    model.fit(x.iloc[:1200], y.iloc[:1200])
    preds = model.predict(x.iloc[1200:1260])

    assert calls[0] == "gpu"        # first attempt requested the GPU
    assert calls[-1] == "cpu"       # and it fell back to CPU
    assert ((preds >= 0) & (preds <= 1)).all()


def test_balanced_class_weight_runs(market, small_config):
    """Balanced class weighting should train and still emit valid probabilities."""
    small_config.model.class_weight = "balanced"
    x, y = _xy(market, small_config)
    model = LightGBMModel(small_config.model, task="classification")
    model.fit(x.iloc[:1500], y.iloc[:1500])
    preds = model.predict(x.iloc[1500:1600])
    assert ((preds >= 0) & (preds <= 1)).all()


def test_probability_calibrator_monotone_and_serializable():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, size=500)
    # Labels correlated with raw so calibration is well-defined.
    labels = (raw + 0.2 * rng.standard_normal(500) > 0.5).astype(int)
    calib = ProbabilityCalibrator.fit(raw, labels, "isotonic")
    assert calib is not None

    restored = ProbabilityCalibrator.from_dict(calib.to_dict())
    grid = np.linspace(0, 1, 50)
    out = restored.transform(grid)
    assert ((out >= 0) & (out <= 1)).all()
    # Isotonic calibration is monotone non-decreasing.
    assert np.all(np.diff(out) >= -1e-9)


def test_calibrator_none_when_single_class():
    raw = np.linspace(0, 1, 100)
    labels = np.ones(100, dtype=int)
    assert ProbabilityCalibrator.fit(raw, labels, "isotonic") is None


def _stub_version_dir(base: Path, version: int) -> None:
    """Create a minimal registry version directory for prune tests."""
    label = f"v_{version}"
    version_dir = base / label
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "model.txt").write_text("stub", encoding="utf-8")
    (version_dir / "metadata.json").write_text(
        json.dumps({"label": label, "backend": "lightgbm", "model_file": "model.txt"}),
        encoding="utf-8",
    )


def test_registry_prune_keeps_latest_ten(tmp_path):
    model_dir = tmp_path / "models"
    registry = ModelRegistry(str(model_dir))
    for version in range(1, 16):
        _stub_version_dir(model_dir, version)

    removed = registry.prune_old_versions(keep=10)
    assert removed == [f"v_{i}" for i in range(1, 6)]
    remaining = registry._sorted_version_labels()
    assert remaining == [f"v_{i}" for i in range(6, 16)]


def test_registry_prune_never_deletes_protected(tmp_path):
    model_dir = tmp_path / "models"
    registry = ModelRegistry(str(model_dir))
    for version in range(1, 16):
        _stub_version_dir(model_dir, version)
    registry.set_promoted("v_3")

    registry.prune_old_versions(keep=10, protect={"v_4"})
    remaining = set(registry._sorted_version_labels())
    assert "v_3" in remaining
    assert "v_4" in remaining
    assert "v_15" in remaining
    assert "v_1" not in remaining
    assert "v_2" not in remaining


def test_registry_save_auto_prunes(market, small_config, tmp_path):
    small_config.model.model_dir = str(tmp_path / "models")
    small_config.model.retain_versions = 3
    x, y = _xy(market, small_config)
    registry = ModelRegistry(small_config.model.model_dir)
    labels = []
    for end in (1200, 1400, 1600, 1800, 2000):
        model = LightGBMModel(small_config.model, task="classification")
        model.fit(x.iloc[:end], y.iloc[:end])
        labels.append(
            registry.save(model, retain_versions=small_config.model.retain_versions)
        )

    remaining = registry._sorted_version_labels()
    assert len(remaining) == 3
    assert remaining == labels[-3:]
