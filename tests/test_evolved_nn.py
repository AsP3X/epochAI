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
    model = EvolvedNNModel(cfg.model).fit(
        x.iloc[:2000],
        y.iloc[:2000],
        compute_importance=True,
    )
    importance = model.feature_importance()
    assert len(importance) == x.shape[1]
    assert importance.sum() >= 0.0


def test_importance_skipped_by_default(market, small_config):
    cfg = _evolved_config(small_config)
    cfg.model.nn.compute_importance = False
    x, y = _xy(market, cfg)
    model = EvolvedNNModel(cfg.model).fit(x.iloc[:2000], y.iloc[:2000])
    assert model.feature_importance().sum() == 0.0


def test_resolve_device_auto_defaults_cpu(small_config):
    from epoch_ai.models.nn_trainer import resolve_device

    small_config.model.device = "auto"
    device = resolve_device(small_config.model)
    assert device.type in ("cpu", "cuda")


def test_parallel_evolution_completes(market, small_config):
    cfg = _evolved_config(small_config)
    cfg.model.evolution.fast_fit = False
    cfg.model.evolution.parallel_candidates = True
    cfg.model.evolution.population_size = 4
    cfg.model.evolution.generations = 1
    cfg.model.evolution.max_workers = 2
    cfg.model.nn.max_epochs = 8
    cfg.model.nn.patience = 2
    x, y = _xy(market, cfg)
    model = EvolvedNNModel(cfg.model).fit(x.iloc[:1200], y.iloc[:1200])
    assert model.genome_ is not None


def test_maybe_compile_skips_worker_threads(small_config):
    """Parallel evolution must not torch.compile inside thread-pool workers."""
    from concurrent.futures import ThreadPoolExecutor

    from epoch_ai.models.nn_genome import default_genome
    from epoch_ai.models.nn_trainer import _maybe_compile, build_mlp, resolve_device

    cfg = _evolved_config(small_config)
    cfg.model.nn.torch_compile = True
    device = resolve_device(cfg.model)
    genome = default_genome(cfg.model.nn)

    def _worker():
        base = build_mlp(8, genome, task="classification").to(device)
        compiled = _maybe_compile(base, cfg.model, device)
        return compiled is base

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert pool.submit(_worker).result()


def test_state_dict_keys_have_no_compile_prefix(market, small_config):
    """torch.compile wraps modules in _orig_mod; saved weights must stay prefix-free."""
    cfg = _evolved_config(small_config)
    cfg.model.nn.torch_compile = True
    x, y = _xy(market, cfg)
    model = EvolvedNNModel(cfg.model).fit(x.iloc[:1200], y.iloc[:1200])
    assert model.state_dict_ is not None
    assert all(not k.startswith("_orig_mod.") for k in model.state_dict_)
    # Predict path rebuilds an uncompiled module and must load these keys cleanly.
    preds = model.predict(x.iloc[1200:1260])
    assert ((preds >= 0) & (preds <= 1)).all()


def test_inference_model_cached_across_predicts(market, small_config):
    """predict() reuses one eval network instead of rebuilding every call."""
    cfg = _evolved_config(small_config)
    x, y = _xy(market, cfg)
    model = EvolvedNNModel(cfg.model).fit(x.iloc[:1500], y.iloc[:1500])
    first = model.predict(x.iloc[1500:1520])
    cached = model._infer_model
    assert cached is not None
    second = model.predict(x.iloc[1520:1540])
    assert model._infer_model is cached  # same object reused
    assert first.shape[0] == 20 and second.shape[0] == 20


def test_evolution_early_stop_patience(market, small_config):
    """Evolution halts once no improvement for the configured patience."""
    cfg = _evolved_config(small_config)
    cfg.model.evolution.fast_fit = False
    cfg.model.evolution.population_size = 4
    cfg.model.evolution.generations = 20
    cfg.model.evolution.early_stop_patience = 1
    cfg.model.nn.max_epochs = 10
    cfg.model.nn.patience = 2
    x, y = _xy(market, cfg)
    model = EvolvedNNModel(cfg.model).fit(x.iloc[:1200], y.iloc[:1200])
    assert model.genome_ is not None


def test_warm_start_from_seed_genome(market, small_config):
    from epoch_ai.models.nn_genome import default_genome

    cfg = _evolved_config(small_config)
    x, y = _xy(market, cfg)
    seed = default_genome(cfg.model.nn)
    first = EvolvedNNModel(cfg.model).fit(
        x.iloc[:1000],
        y.iloc[:1000],
        seed_genome=seed,
    )
    second = EvolvedNNModel(cfg.model).fit(
        x.iloc[:1200],
        y.iloc[:1200],
        seed_genome=first.genome_,
        seed_state=first.state_dict_,
    )
    assert second.genome_ is not None
    assert second.state_dict_ is not None


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


def test_train_genome_tolerates_trailing_singleton_batch():
    """Regression: train_rows % batch_size == 1 must not crash BatchNorm1d."""
    from epoch_ai.config.settings import ModelConfig
    from epoch_ai.models.nn_genome import NNGenome
    from epoch_ai.models.nn_trainer import train_genome

    n_train = 18689  # 18689 % 256 == 1 with default batch_size
    n_val = 200
    n = n_train + n_val
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, 24)).astype(np.float32)
    y = (rng.random(n) > 0.5).astype(np.float32)

    config = ModelConfig(
        backend="evolved_nn",
        val_fraction=0.15,
        nn={"max_epochs": 10, "patience": 1, "batch_size": 256},
    )
    genome = NNGenome(
        hidden_sizes=[128],
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        use_batch_norm=True,
    )
    result = train_genome(
        x,
        y,
        genome,
        config,
        task="classification",
        sample_weight=None,
        val_fraction=0.15,
        split=n_train,
    )
    assert result.best_epoch >= 1
    assert result.state_dict


def test_initialize_population_from_seed():
    from epoch_ai.config.settings import EvolutionConfig, NNConfig
    from epoch_ai.models.nn_genome import default_genome, initialize_population_from_seed

    rng = np.random.default_rng(0)
    nn = NNConfig()
    evolution = EvolutionConfig(population_size=6)
    seed = default_genome(nn)
    pop = initialize_population_from_seed(rng, nn, evolution, seed)
    assert len(pop) == 6
    assert pop[0] == seed


def test_model_class_lazy_import():
    cls = model_class("evolved_nn")
    assert cls.BACKEND == "evolved_nn"
