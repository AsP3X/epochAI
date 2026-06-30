"""Tests for the causal TCN temporal backend."""

from __future__ import annotations

import numpy as np
import pytest

from epoch_ai.features.pipeline import (
    FeaturePipeline,
    build_multi_horizon_targets,
    build_target,
)
from epoch_ai.models.factory import build_model, model_class
from epoch_ai.models.registry import ModelRegistry

pytest.importorskip("torch")

from epoch_ai.models.tcn_model import TCNModel  # noqa: E402

pytestmark = pytest.mark.slow


def _tcn_config(small_config):
    small_config.model.backend = "tcn"
    small_config.model.calibration = "isotonic"
    small_config.model.val_fraction = 0.2
    small_config.model.refit_full_after_es = False
    small_config.model.tcn.lookback = 16
    small_config.model.tcn.channels = [16, 16]
    small_config.model.tcn.kernel_size = 3
    small_config.model.tcn.max_epochs = 20
    small_config.model.tcn.patience = 4
    small_config.model.tcn.batch_size = 128
    small_config.prediction.horizons = [4, 8]
    small_config.prediction.horizon = 8
    return small_config


def _xy_multi(market, config):
    features = FeaturePipeline(config).transform(market)
    y = build_target(market, config.prediction)
    multi = build_multi_horizon_targets(market, config.prediction)
    cols = ["target"]
    for h in config.prediction.horizons:
        cols.extend([f"ret_{h}", f"target_{h}"])
    data = features.join(y).join(multi).dropna(subset=cols)
    multi_cols = [c for c in data.columns if c.startswith(("ret_", "target_"))]
    return data[features.columns], data["target"], data[multi_cols]


def _fit(market, cfg, n: int = 2400):
    x, y, multi = _xy_multi(market, cfg)
    model = TCNModel(cfg.model, task="classification")
    model.fit(
        x.iloc[:n],
        y.iloc[:n],
        prediction=cfg.prediction,
        multi_targets=multi.iloc[:n],
    )
    return model, x, y


def test_factory_builds_tcn(small_config):
    small_config.model.backend = "tcn"
    model = build_model(small_config.model)
    assert isinstance(model, TCNModel)
    assert model_class("tcn").BACKEND == "tcn"


def test_sequence_lookback_exposed(small_config):
    cfg = _tcn_config(small_config)
    model = TCNModel(cfg.model)
    assert model.sequence_lookback == cfg.model.tcn.lookback


def test_fit_predict_multi_head(market, small_config):
    cfg = _tcn_config(small_config)
    model, x, _ = _fit(market, cfg)
    preds = model.predict(x.iloc[2400:2520])
    assert preds.shape[0] == 120
    assert ((preds >= 0) & (preds <= 1)).all()

    structured = model.predict_structured(x.iloc[2400:2520])
    assert set(structured) == set(cfg.prediction.horizons)
    for h in cfg.prediction.horizons:
        block = structured[h]
        assert block["p_up"].shape[0] == 120
        assert ((block["p_up"] >= 0) & (block["p_up"] <= 1)).all()


def test_save_load_roundtrip(market, small_config, tmp_path):
    cfg = _tcn_config(small_config)
    model, x, _ = _fit(market, cfg)
    preds = model.predict(x.iloc[2400:2480])

    path = tmp_path / "model.pt"
    model.save(str(path))
    assert path.with_name(path.name + ".tcn.json").exists()
    assert path.with_name(path.name + ".scaler.json").exists()

    loaded = TCNModel.load(str(path), cfg.model)
    reloaded = loaded.predict(x.iloc[2400:2480])
    assert np.allclose(preds, reloaded, atol=1e-5)


def test_registry_roundtrip(market, small_config, tmp_path):
    cfg = _tcn_config(small_config)
    model, x, _ = _fit(market, cfg)
    registry = ModelRegistry(str(tmp_path / "models"))
    label = registry.save(model, metadata={"train_rows": 2400})
    loaded, meta = registry.load(label, cfg.model)
    assert meta["backend"] == "tcn"
    assert meta["model_file"] == "model.pt"
    preds = loaded.predict(x.iloc[2400:2460])
    assert ((preds >= 0) & (preds <= 1)).all()


def test_calibration_persisted(market, small_config, tmp_path):
    cfg = _tcn_config(small_config)
    model, _, _ = _fit(market, cfg)
    assert model.multi_calibrator_ is not None
    path = tmp_path / "model.pt"
    model.save(str(path))
    assert path.with_name(path.name + ".calibration.json").exists()
    loaded = TCNModel.load(str(path), cfg.model)
    assert loaded.multi_calibrator_ is not None


def test_future_rows_cannot_change_past_predictions(market, small_config):
    """Causality: perturbing future bars must leave earlier predictions unchanged."""
    import torch

    torch.manual_seed(0)  # deterministic, non-degenerate fit for the sanity assertion
    cfg = _tcn_config(small_config)
    model, x, _ = _fit(market, cfg)
    frame = x.iloc[2400:2520]
    p1 = model.predict(frame)

    cut = 60
    perturbed = frame.copy()
    perturbed.iloc[cut:] = perturbed.iloc[cut:] + 50.0
    p2 = model.predict(perturbed)

    # Causality guarantee (independent of weights): predictions for bars before the
    # perturbation are byte-for-byte identical because their windows use only rows <= t.
    assert np.allclose(p1[:cut], p2[:cut], atol=1e-6)
    # Sanity: a large perturbation of the future bars does move their predictions.
    assert np.max(np.abs(p1[cut:] - p2[cut:])) > 1e-3


def test_lookback_context_tail_equivalence(market, small_config):
    """Predicting a sub-block with a lookback tail matches the full-region prediction.

    This is exactly how the walk-forward engine feeds sequence models: it prepends the
    preceding ``lookback - 1`` rows and trims them from the output.
    """
    cfg = _tcn_config(small_config)
    model, x, _ = _fit(market, cfg)
    lb = cfg.model.tcn.lookback
    start, end = 2400, 2520
    region = x.iloc[start:end]
    p_full = model.predict(region)

    offset = 40
    s2 = start + offset
    ctx = x.iloc[s2 - (lb - 1) : end]
    p_ctx = model.predict(ctx)[lb - 1 :]

    assert p_ctx.shape[0] == p_full[offset:].shape[0]
    assert np.allclose(p_full[offset:], p_ctx, atol=1e-6)


def test_load_uses_saved_arch_not_live_config(market, small_config, tmp_path):
    """A loaded model must rebuild from the saved architecture even if config drifted."""
    cfg = _tcn_config(small_config)
    model, x, _ = _fit(market, cfg)
    preds = model.predict(x.iloc[2400:2480])

    path = tmp_path / "model.pt"
    model.save(str(path))

    # Simulate the live config drifting away from the trained architecture.
    cfg.model.tcn.channels = [8]
    cfg.model.tcn.lookback = 8
    loaded = TCNModel.load(str(path), cfg.model)
    assert loaded.arch_["channels"] == [16, 16]
    assert loaded.sequence_lookback == 16
    reloaded = loaded.predict(x.iloc[2400:2480])
    assert np.allclose(preds, reloaded, atol=1e-5)


def test_feature_importance_optional(market, small_config):
    cfg = _tcn_config(small_config)
    cfg.model.tcn.compute_importance = False
    x, y, multi = _xy_multi(market, cfg)
    model = TCNModel(cfg.model).fit(
        x.iloc[:2400],
        y.iloc[:2400],
        prediction=cfg.prediction,
        multi_targets=multi.iloc[:2400],
    )
    assert model.feature_importance().sum() == 0.0
