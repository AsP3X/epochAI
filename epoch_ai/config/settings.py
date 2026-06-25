"""Strongly-typed application configuration.

Configuration is expressed as nested `Pydantic` models and is normally loaded from a
YAML file (see ``config/config.yaml``).  Every tunable knob of the system - symbols,
timeframe, prediction horizon, progressive walk-forward parameters, retraining
frequency, risk parameters and feature toggles - lives here so the whole system is
fully config-driven.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    """Settings controlling data acquisition and storage.

    Attributes:
        exchange: CCXT exchange id (e.g. ``"binanceusdm"`` for USDT-M futures).
        market_type: ``"spot"`` or ``"future"`` (derivatives carry funding/OI data).
        historical_start_date: ISO date for the *oldest* candle to fetch. The
            downloader walks forward from here to maximise historical depth.
        data_dir: Directory where raw/aligned parquet datasets are stored.
        use_synthetic_fallback: When ``True`` (default) the downloader generates a
            realistic synthetic dataset if the exchange is unreachable, guaranteeing
            the pipeline is runnable fully offline.
        synthetic_seed: RNG seed for reproducible synthetic data.
        rate_limit_ms: Politeness delay between paginated REST requests.
    """

    exchange: str = "binanceusdm"
    market_type: Literal["spot", "future"] = "future"
    historical_start_date: str = "2019-11-01"
    data_dir: str = "artifacts/data"
    use_synthetic_fallback: bool = True
    synthetic_seed: int = 7
    rate_limit_ms: int = 250


class FeatureConfig(BaseModel):
    """Toggles for the modular feature groups.

    Each flag enables/disables a registered feature group. Disabling unused groups
    keeps the feature matrix small and training fast.
    """

    technical: bool = True
    microstructure: bool = True
    derivatives: bool = True
    volatility: bool = True
    time: bool = True
    sentiment: bool = False
    onchain: bool = False
    dropna: bool = True


class PredictionConfig(BaseModel):
    """Defines the supervised-learning target.

    Attributes:
        horizon: Forward horizon, in candles, over which the outcome is measured.
        task: ``"classification"`` predicts P(up); ``"regression"`` predicts return.
        threshold: Forward return above which a candle is labelled "up"
            (classification) - a small positive value can encode a neutral band.
    """

    horizon: int = 12
    task: Literal["classification", "regression"] = "classification"
    threshold: float = 0.0


class ModelConfig(BaseModel):
    """LightGBM hyper-parameters and model-registry location."""

    model_dir: str = "artifacts/models"
    num_boost_round: int = 300
    early_stopping_rounds: int | None = 30
    params: dict[str, Any] = Field(
        default_factory=lambda: {
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": -1,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "verbosity": -1,
        }
    )


class WalkForwardConfig(BaseModel):
    """Progressive / expanding-window walk-forward parameters.

    Attributes:
        initial_train_period: Number of oldest candles used for the first model fit.
        step_size: How many candles to advance (and predict) on each iteration.
        retrain_frequency: Retrain every ``N`` steps (1 = retrain every step).
        expanding: ``True`` for an expanding window (full history); ``False`` for a
            rolling window of ``initial_train_period`` candles.
        recency_half_life: If set, sample weights decay with this half-life (in
            candles), emphasising recent regimes while still using full history.
        max_steps: Optional cap on the number of walk-forward steps (useful for
            quick smoke runs / demos).
    """

    initial_train_period: int = 2000
    step_size: int = 200
    retrain_frequency: int = 1
    expanding: bool = True
    recency_half_life: int | None = None
    max_steps: int | None = None


class RiskConfig(BaseModel):
    """Risk-management parameters used by the (separate) execution layer."""

    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.02
    max_leverage: float = 3.0
    long_threshold: float = 0.55
    short_threshold: float = 0.45
    fee_rate: float = 0.0004
    slippage: float = 0.0002
    allow_short: bool = True
    min_confidence: float = 0.0
    max_drawdown_halt: float | None = None
    max_daily_loss: float | None = None
    cooldown_bars: int = 0


class ExecutionConfig(BaseModel):
    """Live runtime and trade-execution settings (separate from model training)."""

    mode: Literal["paper", "live"] = "paper"
    live_enabled: bool = False
    dry_run: bool = True
    api_key_env: str = "EPOCH_AI_API_KEY"
    api_secret_env: str = "EPOCH_AI_API_SECRET"
    reserve_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of session profits set aside (not reinvested).",
    )
    treasury_state_path: str = "artifacts/treasury.json"
    min_buffer_bars: int = 500


class BacktestConfig(BaseModel):
    """Backtester settings."""

    use_vectorbt: bool = False
    annualization_factor: int | None = None


class LoggingConfig(BaseModel):
    """Prediction/outcome log store location."""

    db_path: str = "artifacts/logs/predictions.sqlite"


class TrackingConfig(BaseModel):
    """Optional MLflow experiment tracking."""

    enabled: bool = False
    tracking_uri: str = "artifacts/mlruns"
    experiment_name: str = "epoch-ai"


class AppConfig(BaseModel):
    """Root configuration object aggregating every sub-config."""

    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT"])
    timeframe: str = "15m"
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @property
    def primary_symbol(self) -> str:
        """The first configured symbol (the default focus pair)."""
        return self.symbols[0]

    @model_validator(mode="after")
    def _validate(self) -> AppConfig:
        if not self.symbols:
            raise ValueError("At least one symbol must be configured.")
        if self.prediction.horizon < 1:
            raise ValueError("prediction.horizon must be >= 1 candle.")
        if self.walk_forward.step_size < 1:
            raise ValueError("walk_forward.step_size must be >= 1 candle.")
        if self.walk_forward.initial_train_period < self.prediction.horizon + 1:
            raise ValueError(
                "walk_forward.initial_train_period must exceed prediction.horizon."
            )
        return self


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate an :class:`AppConfig` from a YAML file.

    Args:
        path: Path to the YAML config. When ``None`` an :class:`AppConfig` built
            entirely from defaults is returned.

    Returns:
        A validated :class:`AppConfig`.

    Raises:
        FileNotFoundError: If ``path`` is provided but does not exist.
    """
    if path is None:
        return AppConfig()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return AppConfig.model_validate(raw)
