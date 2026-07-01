"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest
import yaml

from epoch_ai.config.overrides import apply_overrides, parse_set_args
from epoch_ai.config.settings import AppConfig, EvolutionConfig, load_config


def test_defaults_are_valid():
    config = AppConfig()
    assert config.primary_symbol == "BTC/USDT"
    assert config.prediction.horizon >= 1
    assert config.data.use_synthetic_fallback is False


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
            {
                "prediction": {"horizon": 60, "horizons": [60]},
                "walk_forward": {"initial_train_period": 10},
            }
        )


def test_horizon_must_be_in_horizons():
    with pytest.raises(ValueError):
        AppConfig.model_validate({"prediction": {"horizon": 12, "horizons": [1, 5, 10]}})


def test_quantiles_must_include_median():
    with pytest.raises(ValueError):
        AppConfig.model_validate({"prediction": {"quantiles": [0.1, 0.9]}})


def test_multi_horizon_defaults():
    config = AppConfig()
    assert config.prediction.horizons == [1, 5, 10, 15, 30, 60]
    assert config.prediction.horizon == 60
    assert config.prediction.quantiles == [0.1, 0.5, 0.9]
    assert config.prediction.max_horizon == 60
    assert config.prediction.n_outputs == 24  # 6 horizons x (3 quantiles + 1 direction)
    # Default timeframe is 15m, so 60 candles = 900 minutes = 15h.
    assert config.prediction.horizon_label(60) == "15h"


def test_horizon_label_scales_with_timeframe():
    cfg_5m = AppConfig.model_validate(
        {"timeframe": "5m", "prediction": {"horizon": 12, "horizons": [1, 3, 6, 12, 24, 48]}}
    )
    labels = [cfg_5m.prediction.horizon_label(h) for h in [1, 3, 6, 12, 24, 48]]
    assert labels == ["5m", "15m", "30m", "1h", "2h", "4h"]

    cfg_1m = AppConfig.model_validate(
        {"timeframe": "1m", "prediction": {"horizon": 60, "horizons": [1, 5, 60]}}
    )
    assert cfg_1m.prediction.horizon_label(1) == "1m"
    assert cfg_1m.prediction.horizon_label(60) == "1h"


def test_embargo_resolves_to_max_horizon():
    config = AppConfig.model_validate(
        {
            "prediction": {"horizon": 15, "horizons": [1, 5, 15, 60]},
            "walk_forward": {"initial_train_period": 5000, "embargo": None},
        }
    )
    assert config.prediction.resolved_embargo(None) == 60
    assert config.prediction.resolved_embargo(10) == 10


def test_single_horizon_mode_from_empty_horizons():
    config = AppConfig.model_validate({"prediction": {"horizon": 8, "horizons": []}})
    assert config.prediction.horizons == [8]


def test_initial_train_must_exceed_horizon_legacy():
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
    assert model.device == "auto"
    assert model.gpu_platform_id == -1
    assert model.gpu_device_id == -1


def test_invalid_device_rejected():
    with pytest.raises(ValueError):
        AppConfig.model_validate({"model": {"device": "tpu"}})


def test_model_backend_default_is_evolved_nn():
    assert AppConfig().model.backend == "evolved_nn"
    assert AppConfig().model.evolution.enabled is True
    assert AppConfig().model.device == "auto"


def test_evolution_config_defaults():
    evo = AppConfig().model.evolution
    assert evo.population_size >= 2
    assert evo.generations >= 1
    assert evo.fast_fit is False
    assert evo.parallel_candidates is True
    assert evo.early_stop_patience is None
    assert evo.cuda_auto_workers is True
    assert evo.cuda_worker_cap_max == 12
    assert len(evo.cuda_worker_caps) == len(evo.cuda_worker_vram_gb) + 1
    assert evo.successive_halving is False
    assert 0.0 < evo.sh_proxy_epoch_fraction <= 1.0
    assert 0.0 < evo.sh_promote_fraction <= 1.0


def test_successive_halving_fraction_bounds_rejected():
    with pytest.raises(ValueError):
        EvolutionConfig.model_validate({"sh_proxy_epoch_fraction": 0.0})
    with pytest.raises(ValueError):
        EvolutionConfig.model_validate({"sh_promote_fraction": 1.5})


def test_evolution_cuda_worker_tiers_validation():
    with pytest.raises(ValueError, match="cuda_worker_caps"):
        EvolutionConfig.model_validate(
            {
                "cuda_worker_vram_gb": [8.0, 16.0],
                "cuda_worker_caps": [2, 4],
            }
        )


def test_cuda_performance_defaults():
    cuda = AppConfig().model.cuda
    assert cuda.allow_tf32 is True
    assert cuda.cudnn_benchmark is True
    assert cuda.matmul_precision == "high"


def test_tcn_config_defaults():
    tcn = AppConfig().model.tcn
    assert tcn.lookback >= 4
    assert tcn.channels and all(c > 0 for c in tcn.channels)
    assert tcn.kernel_size >= 2
    assert 0.0 <= tcn.dropout < 1.0
    assert tcn.max_epochs >= 5


def test_tcn_backend_accepted():
    config = AppConfig.model_validate({"model": {"backend": "tcn"}})
    assert config.model.backend == "tcn"


def test_tcn_empty_channels_rejected():
    from epoch_ai.config.settings import TCNConfig

    with pytest.raises(ValueError, match="channels"):
        TCNConfig.model_validate({"channels": []})


def test_tcn_large_capacity_config_validates():
    # Human: a powerful-GPU trunk (deeper/wider blocks + longer context) must be a
    #        first-class config, not something the schema rejects. This guards the
    #        documented GPU preset in config/config.yaml.
    # Agent: CONFIG model.tcn; asserts large capacity round-trips; no upper-bound rejection.
    config = AppConfig.model_validate(
        {
            "model": {
                "backend": "tcn",
                "tcn": {
                    "channels": [128, 128, 256, 256, 512],
                    "lookback": 192,
                    "batch_size": 1024,
                    "mixed_precision": True,
                },
            }
        }
    )
    tcn = config.model.tcn
    assert tcn.channels == [128, 128, 256, 256, 512]
    assert tcn.channels[-1] == 512
    assert len(tcn.channels) == 5  # five dilated residual blocks (deeper trunk)
    assert tcn.lookback == 192
    assert tcn.batch_size == 1024
    assert tcn.mixed_precision is True


def test_tcn_config_rejects_nonpositive():
    from pydantic import ValidationError

    from epoch_ai.config.settings import TCNConfig

    # Human: lower bounds keep a large-capacity trunk sane; lookback must be >= 1 bar
    #        and channel counts must be positive.
    # Agent: CONFIG model.tcn; lookback below ge and non-positive channels raise.
    with pytest.raises(ValidationError):
        TCNConfig.model_validate({"lookback": 0})
    with pytest.raises(ValidationError, match="channels"):
        TCNConfig.model_validate({"channels": [64, 0, 128]})


def test_tcn_backend_sets_slower_retrain_cadence():
    # tcn (like evolved_nn) defaults walk-forward retrain_frequency to 5 unless overridden.
    config = AppConfig.model_validate({"model": {"backend": "tcn"}})
    assert config.walk_forward.retrain_frequency == 5


def test_cuda_matmul_precision_rejects_unknown():
    from epoch_ai.config.settings import CudaPerformanceConfig

    with pytest.raises(ValueError, match="matmul_precision"):
        CudaPerformanceConfig.model_validate({"matmul_precision": "turbo"})


def test_registry_defaults_include_register_each_retrain():
    model = AppConfig().model
    assert model.register_each_retrain is True
    assert model.defer_registry_prune is False


def test_evolved_nn_default_retrain_frequency():
    assert AppConfig().walk_forward.retrain_frequency == 5


def test_lightgbm_default_retrain_frequency():
    config = AppConfig.model_validate({"model": {"backend": "lightgbm"}})
    assert config.walk_forward.retrain_frequency == 1


def test_nn_performance_defaults():
    nn = AppConfig().model.nn
    assert nn.min_layers == 1
    assert nn.max_layers == 3
    assert nn.fixed_hidden_sizes is None
    assert nn.compute_importance is True
    assert nn.mixed_precision is True
    assert nn.torch_compile is True
    assert nn.cuda_auto_batch is True
    assert nn.cuda_batches_per_epoch == 32


def test_nn_deep_layers_override_via_cli():
    overrides = parse_set_args(
        [
            "model.nn.min_layers=4",
            "model.nn.max_layers=6",
            "model.nn.hidden_size_max=1024",
            "model.nn.fixed_hidden_sizes=[512,384,256,128]",
        ]
    )
    config = AppConfig.model_validate(apply_overrides({}, overrides))
    nn = config.model.nn
    assert nn.min_layers == 4
    assert nn.max_layers == 6
    assert nn.hidden_size_max == 1024
    assert nn.fixed_hidden_sizes == [512, 384, 256, 128]


def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        AppConfig.model_validate({"model": {"backend": "catboost"}})


def test_shipped_config_yaml_backend():
    config = load_config("config/config.yaml")
    assert config.model.backend == "tcn"
    assert config.model.tcn.lookback == 96
    assert config.model.tcn.channels == [64, 64, 128, 128]


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
        "oos_brier_weighted",
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
    assert config.timeframe == "5m"
    assert config.model.calibration == "isotonic"
    assert config.walk_forward.recency_half_life == 15000
    assert config.prediction.horizon == 12
    assert config.prediction.horizons == [1, 3, 6, 12, 24, 48]
    assert config.prediction.quantiles == [0.1, 0.5, 0.9]
    assert config.prediction.neutral_band == 0.0005
    assert "ETH/USDT" in config.data.context_symbols
    assert "SOL/USDT" in config.data.context_symbols
    assert "BNB/USDT" in config.data.context_symbols
    assert config.features.cross_asset is True
    assert config.features.sentiment is True
    assert config.features.patterns is True
    assert config.adaptation.schedule_interval_hours == 24.0
    # Benchmark beats are now report-only by default; the absolute floor is the gate.
    assert config.rl.promotion.require_beat_baseline is False
    assert config.rl.promotion.require_beat_buy_hold is False
    assert config.rl.promotion.min_absolute_metric == 0.0
    assert config.features.manipulation is True
    assert config.features.higher_timeframe is True
    assert config.features.onchain is True
    assert config.risk.long_threshold == 0.58
    assert config.risk.short_threshold == 0.42
    assert config.data.use_synthetic_fallback is False
    assert config.model.evolution.fast_fit is True
    assert config.model.nn.fixed_hidden_sizes == [512, 384, 256, 128, 64]
    assert config.model.nn.torch_compile is False
    assert config.model.nn.cuda_batch_cap == 4096


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
    assert fc.higher_timeframe is True
    assert fc.macro is True
    assert fc.htf_timeframes == ["1h", "4h"]
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


def test_rl_reward_config_defaults():
    from epoch_ai.config.settings import AppConfig

    cfg = AppConfig().rl
    assert cfg.reward_mode in {"per_bar", "multi_bar"}
    assert cfg.reward_horizon >= 1
    assert cfg.turnover_penalty >= 0.0


def test_rl_reward_config_rejects_invalid_values():
    from pydantic import ValidationError

    from epoch_ai.config.settings import RLConfig

    with pytest.raises(ValidationError):
        RLConfig(reward_horizon=0)
    with pytest.raises(ValidationError):
        RLConfig(turnover_penalty=-1)
