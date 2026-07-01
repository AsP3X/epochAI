"""Tests for the TCN trunk-embedding accessor (``TCNModel.embed``).

The trunk embedding is the pre-head activation ``last`` (size ``channels[-1]``) that
the head consumes. Exposing it lets downstream consumers reuse the shared temporal
trunk without re-running the head. Embeddings must stay **causal** (row ``i`` depends
only on rows ``<= i``) and deterministic (eval net, dropout off).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from epoch_ai.features.pipeline import (  # noqa: E402
    FeaturePipeline,
    build_multi_horizon_targets,
    build_target,
)
from epoch_ai.models.tcn_model import TCNModel  # noqa: E402


def _fit_tiny_tcn(market, small_config):
    """Fit a tiny multi-head TCN (channels [16, 16], lookback 16) on the fixture.

    Mirrors ``tests/test_ppo_policy.py::test_policy_trains_on_real_forecasts``: build
    features + per-horizon targets from the synthetic ``market`` fixture, then train a
    small TCN for a few epochs. Returns ``(model, x)`` where ``x`` is the feature frame.
    """
    cfg = small_config
    cfg.model.backend = "tcn"
    cfg.model.calibration = "none"
    cfg.model.val_fraction = 0.2
    cfg.model.refit_full_after_es = False
    cfg.model.tcn.lookback = 16
    cfg.model.tcn.channels = [16, 16]
    cfg.model.tcn.kernel_size = 3
    cfg.model.tcn.max_epochs = 6
    cfg.model.tcn.patience = 3
    cfg.model.tcn.batch_size = 128
    cfg.prediction.horizons = [4, 8]
    cfg.prediction.horizon = 8

    features = FeaturePipeline(cfg).transform(market)
    y = build_target(market, cfg.prediction)
    multi = build_multi_horizon_targets(market, cfg.prediction)
    keep = ["target"]
    for h in cfg.prediction.horizons:
        keep.extend([f"ret_{h}", f"target_{h}"])
    data = features.join(y).join(multi).dropna(subset=keep)
    multi_cols = [c for c in data.columns if c.startswith(("ret_", "target_"))]
    x, target, mt = data[features.columns], data["target"], data[multi_cols]

    model = TCNModel(cfg.model, task="classification")
    model.fit(
        x.iloc[:2400],
        target.iloc[:2400],
        prediction=cfg.prediction,
        multi_targets=mt.iloc[:2400],
    )
    return model, x.iloc[:2400]


def test_embed_shape(market, small_config):
    # Human: the trunk embedding has one row per input bar and width == channels[-1].
    # Agent: RETURNS (len(x), trunk_dim); trunk_dim == channels[-1] == 16.
    model, x = _fit_tiny_tcn(market, small_config)
    emb = model.embed(x)
    assert emb.shape == (len(x), model.trunk_dim)
    assert model.trunk_dim == 16


def test_embed_is_deterministic(market, small_config):
    # Human: eval net (dropout off) -> two embed calls must be bit-for-bit identical.
    # Agent: deterministic inference; np.allclose over repeated embed(x).
    model, x = _fit_tiny_tcn(market, small_config)
    e1 = model.embed(x)
    e2 = model.embed(x)
    assert np.allclose(e1, e2)


def test_embed_is_causal(market, small_config):
    # Human: mutating rows strictly AFTER i must not change row i's embedding, because
    #        the causal TCN window for row i only ever reads rows <= i.
    # Agent: CAUSAL; perturb x.iloc[i+1:] -> assert np.allclose(e[i], e2[i]).
    model, x = _fit_tiny_tcn(market, small_config)
    e = model.embed(x)
    i = len(x) // 2
    # Human: perturb only rows strictly after i (float frame to avoid int-cast warning).
    x_mut = x.astype("float64").copy()
    rng = np.random.default_rng(123)
    tail = x_mut.iloc[i + 1 :]
    x_mut.iloc[i + 1 :] = tail.to_numpy() + rng.normal(0.0, 5.0, size=tail.shape)
    e2 = model.embed(x_mut)
    assert np.allclose(e[i], e2[i])


def test_predict_unchanged(market, small_config):
    # Human: the embed refactor must not alter the head path -- predict and
    #        predict_structured still return their usual shapes/keys.
    # Agent: sanity; predict len == len(x); predict_structured keys == horizons.
    model, x = _fit_tiny_tcn(market, small_config)
    preds = model.predict(x)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(x)

    structured = model.predict_structured(x)
    assert set(structured.keys()) == {4, 8}
    for block in structured.values():
        assert "p_up" in block
