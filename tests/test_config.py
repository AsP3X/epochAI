"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest
import yaml

from epoch_ai.config.settings import AppConfig, load_config


def test_defaults_are_valid():
    config = AppConfig()
    assert config.primary_symbol == "BTC/USDT"
    assert config.prediction.horizon >= 1


def test_load_from_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"symbols": ["ETH/USDT"], "timeframe": "5m"}))
    config = load_config(path)
    assert config.primary_symbol == "ETH/USDT"
    assert config.timeframe == "5m"


def test_historical_start_date_concrete():
    data = AppConfig.model_validate({"data": {"historical_start_date": "2019-11-01"}}).data
    assert data.fetch_from_earliest() is False
    assert data.start_date_iso() == "2019-11-01"


def test_historical_start_date_earliest_sentinels():
    for sentinel in ("earliest", "AUTO", "all", "Max", ""):
        data = AppConfig.model_validate({"data": {"historical_start_date": sentinel}}).data
        assert data.fetch_from_earliest() is True
        assert data.start_date_iso() == "2017-01-01"


def test_invalid_horizon_rejected():
    with pytest.raises(ValueError):
        AppConfig.model_validate({"prediction": {"horizon": 0}})


def test_initial_train_must_exceed_horizon():
    with pytest.raises(ValueError):
        AppConfig.model_validate(
            {"prediction": {"horizon": 50}, "walk_forward": {"initial_train_period": 10}}
        )


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_new_model_and_feature_defaults():
    config = AppConfig()
    assert config.model.class_weight == "balanced"
    assert config.model.calibration == "isotonic"
    assert 0.0 <= config.model.val_fraction < 0.5
    assert config.backtest.horizon_aware is True
    assert config.features.ma_windows
    assert config.features.rsi_periods


def test_empty_feature_window_rejected():
    with pytest.raises(ValueError):
        AppConfig.model_validate({"features": {"ma_windows": []}})


def test_training_improvement_defaults():
    """New training knobs ship with safe, backward-compatible defaults."""
    config = AppConfig()
    assert config.walk_forward.embargo is None          # resolves to prediction.horizon
    assert config.prediction.neutral_band == 0.0        # dead-zone off by default
    assert config.model.refit_full_after_es is True     # keep freshest bars in final fit


def test_embargo_cannot_exceed_initial_train():
    with pytest.raises(ValueError):
        AppConfig.model_validate(
            {"walk_forward": {"initial_train_period": 100, "embargo": 100}}
        )


def test_model_device_defaults():
    model = AppConfig().model
    assert model.device == "cpu"
    assert model.gpu_platform_id == -1
    assert model.gpu_device_id == -1


def test_invalid_device_rejected():
    with pytest.raises(ValueError):
        AppConfig.model_validate({"model": {"device": "tpu"}})


def test_model_backend_default_is_evolved_nn():
    assert AppConfig().model.backend == "evolved_nn"
    assert AppConfig().model.evolution.enabled is True


def test_evolution_config_defaults():
    evo = AppConfig().model.evolution
    assert evo.population_size >= 2
    assert evo.generations >= 1
    assert evo.fast_fit is False


def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        AppConfig.model_validate({"model": {"backend": "catboost"}})


def test_shipped_config_yaml_backend():
    assert load_config("config/config.yaml").model.backend == "evolved_nn"


def test_promotion_defaults():
    promotion = AppConfig().promotion
    assert promotion.eval_bars == 2000
    assert promotion.metric == "oos_logloss"
    assert promotion.min_improvement == 0.0


def test_shipped_config_yaml_has_promotion():
    config = load_config("config/config.yaml")
    assert config.promotion.eval_bars >= 1
    assert config.promotion.metric in {
        "oos_logloss",
        "oos_brier",
        "oos_rmse",
        "oos_accuracy",
        "oos_auc",
        "oos_directional_accuracy",
    }


def test_invalid_val_fraction_rejected():
    with pytest.raises(ValueError):
        AppConfig.model_validate({"model": {"val_fraction": 0.9}})


def test_shipped_config_yaml_loads():
    """The example config must resolve with the new keys."""
    config = load_config("config/config.yaml")
    assert config.model.calibration == "sigmoid"
    assert config.walk_forward.recency_half_life == 2000
    assert "ETH/USDT" in config.data.context_symbols
    assert "SOL/USDT" in config.data.context_symbols
    assert config.features.cross_asset is True
    assert config.features.sentiment is True
    assert config.risk.long_threshold == 0.58
    assert config.risk.short_threshold == 0.42


def test_data_enrichment_defaults():
    data = AppConfig().data
    assert "ETH/USDT" in data.context_symbols
    assert "SOL/USDT" in data.context_symbols
    assert data.fetch_fear_greed is True
    assert data.fetch_open_interest is True
    assert data.fetch_spot_basis is True


def test_pattern_config_defaults():
    from epoch_ai.config.settings import FeatureConfig

    fc = FeatureConfig()
    assert fc.patterns is False
    assert fc.manipulation is False
    assert fc.pattern_lookbacks == [48, 96, 192]
    assert fc.pivot_confirm_bars == 3


def test_pattern_lookbacks_must_be_positive():
    from epoch_ai.config.settings import FeatureConfig

    with pytest.raises(ValueError, match="pattern_lookbacks"):
        FeatureConfig(pattern_lookbacks=[])


def test_safety_config_defaults():
    from epoch_ai.config.settings import SafetyConfig

    sc = SafetyConfig()
    assert sc.enabled is False
    assert sc.max_suspicion_score == 0.75
