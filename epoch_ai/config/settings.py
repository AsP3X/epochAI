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
        historical_start_date: ISO date for the *oldest* candle to fetch, or one of
            the sentinels ``"earliest"``/``"auto"`` to fetch from the very first
            candle the exchange offers (true full history). The downloader walks
            forward from here to maximise historical depth.
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
    context_symbols: list[str] = Field(
        default_factory=lambda: ["ETH/USDT"],
        description=(
            "Additional symbols whose OHLCV/derivatives are joined onto the primary "
            "frame as context columns (e.g. eth_close for cross-asset features)."
        ),
    )
    fetch_fear_greed: bool = Field(
        default=True,
        description="Fetch and join the Crypto Fear & Greed index (daily, ffilled).",
    )
    fetch_open_interest: bool = Field(
        default=True,
        description="Paginate open-interest history from the exchange when supported.",
    )
    fetch_spot_basis: bool = Field(
        default=True,
        description="Download spot reference close for perp basis features.",
    )
    spot_exchange: str = Field(
        default="binance",
        description="CCXT exchange id for spot reference OHLCV (basis features).",
    )
    spot_symbol: str | None = Field(
        default=None,
        description="Spot symbol for basis; defaults to the primary symbol.",
    )

    # Sentinel values that mean "start from the exchange's first available candle".
    _EARLIEST_SENTINELS = {"earliest", "auto", "all", "max", ""}
    # Concrete fallback used for synthetic generation / bar-count math in earliest mode.
    _EARLIEST_FALLBACK_DATE = "2017-01-01"

    def fetch_from_earliest(self) -> bool:
        """Whether the oldest-candle start date should be auto-detected."""
        return self.historical_start_date.strip().lower() in self._EARLIEST_SENTINELS

    def start_date_iso(self) -> str:
        """Resolve a concrete ISO start date (used for synthetic/bar-count math)."""
        if self.fetch_from_earliest():
            return self._EARLIEST_FALLBACK_DATE
        return self.historical_start_date


class FeatureConfig(BaseModel):
    """Toggles and window parameters for the modular feature groups.

    Each flag enables/disables a registered feature group. Disabling unused groups
    keeps the feature matrix small and training fast. The window lists make the
    indicator look-backs config-driven instead of hard-coded inside each group.
    """

    technical: bool = True
    microstructure: bool = True
    derivatives: bool = True
    volatility: bool = True
    time: bool = True
    sentiment: bool = False
    onchain: bool = False
    cross_asset: bool = True
    dropna: bool = True

    # Indicator look-back windows (config-driven; consumed by the feature groups).
    return_lags: list[int] = Field(
        default_factory=lambda: [1, 3, 6, 12, 24, 48],
        description="Look-back lags (in candles) for momentum/return features.",
    )
    ma_windows: list[int] = Field(
        default_factory=lambda: [10, 20, 50, 100, 200],
        description="Moving-average windows for SMA/EMA distance features.",
    )
    rsi_periods: list[int] = Field(
        default_factory=lambda: [7, 14, 28],
        description="RSI look-back periods.",
    )
    vol_windows: list[int] = Field(
        default_factory=lambda: [12, 24, 48, 96],
        description="Rolling realised-volatility windows.",
    )

    @model_validator(mode="after")
    def _validate_windows(self) -> FeatureConfig:
        # Window lists must be non-empty positive integers so feature groups emit
        # a stable set of columns; an empty list would silently drop a sub-family.
        for name in ("return_lags", "ma_windows", "rsi_periods", "vol_windows"):
            values = getattr(self, name)
            if not values or any(int(v) < 1 for v in values):
                raise ValueError(f"features.{name} must be a non-empty list of positive ints.")
        return self


class PredictionConfig(BaseModel):
    """Defines the supervised-learning target.

    Attributes:
        horizon: Forward horizon, in candles, over which the outcome is measured.
        task: ``"classification"`` predicts P(up); ``"regression"`` predicts return.
        threshold: Forward return above which a candle is labelled "up"
            (classification) - a small positive value can encode a neutral band.
        neutral_band: Symmetric dead-zone (in forward-return units) around
            ``threshold`` for classification. Bars whose realised forward return falls
            inside ``[threshold - neutral_band, threshold + neutral_band]`` are dropped
            (NaN target) instead of being labelled, so the model learns from decisive
            moves rather than near-zero noise. ``0`` keeps every bar (legacy behaviour).
    """

    horizon: int = 12
    task: Literal["classification", "regression"] = "classification"
    threshold: float = 0.0
    neutral_band: float = Field(
        default=0.0,
        ge=0.0,
        description="Dead-zone half-width around threshold; ambiguous bars are dropped.",
    )


class ModelConfig(BaseModel):
    """LightGBM hyper-parameters, calibration and model-registry location.

    Attributes:
        val_fraction: Fraction of the most-recent training rows held out (time-ordered)
            for early stopping and probability calibration. ``0`` disables both.
        class_weight: ``"balanced"`` derives ``scale_pos_weight`` from the training
            label balance (helps when "up" vs "down" labels are skewed); ``"none"``
            leaves LightGBM unweighted.
        calibration: Post-hoc probability calibration fit on the validation tail —
            ``"isotonic"`` (non-parametric, monotone), ``"sigmoid"`` (Platt scaling)
            or ``"none"``. Only applies to classification.
        refit_full_after_es: When early stopping is active, refit the booster on the
            **full** training window (including the held-out validation tail) for the
            early-stopping-selected number of rounds. This stops the deployed model
            from permanently discarding its most-recent ``val_fraction`` of bars while
            still choosing the iteration count honestly on out-of-sample data.
        backend: Gradient-boosting library — ``"lightgbm"`` (default, CPU-optimised) or
            ``"xgboost"`` (optional dependency; supports real CUDA-GPU training on NVIDIA
            cards via ``device="cuda"``). Both implement the same model interface and
            store open weights in the registry.
        device: Compute device — ``"cpu"`` (default, always available), ``"gpu"`` or
            ``"cuda"``. For LightGBM ``gpu`` is OpenCL; for XGBoost both ``gpu``/``cuda``
            map to CUDA. A GPU request that the installed build/host cannot satisfy
            **falls back to CPU** automatically so a model can always be trained.
        gpu_platform_id: OpenCL platform id for LightGBM ``device="gpu"`` (``-1`` = auto).
        gpu_device_id: OpenCL/CUDA device ordinal (``-1`` = auto). Pins a specific GPU on
            multi-GPU hosts (used by both backends).
    """

    model_dir: str = "artifacts/models"
    backend: Literal["lightgbm", "xgboost"] = Field(
        default="lightgbm",
        description="Gradient-boosting backend; xgboost enables real CUDA-GPU training.",
    )
    num_boost_round: int = 300
    early_stopping_rounds: int | None = 30
    val_fraction: float = Field(
        default=0.15,
        ge=0.0,
        lt=0.5,
        description="Time-ordered validation-tail fraction for early stopping + calibration.",
    )
    class_weight: Literal["none", "balanced"] = "balanced"
    calibration: Literal["none", "isotonic", "sigmoid"] = "isotonic"
    refit_full_after_es: bool = Field(
        default=True,
        description="Refit on the full training window for the ES-selected rounds.",
    )
    device: Literal["cpu", "gpu", "cuda"] = Field(
        default="cpu",
        description="Compute device; gpu/cuda fall back to cpu when unavailable.",
    )
    gpu_platform_id: int = Field(
        default=-1,
        description="OpenCL platform id for device='gpu' (-1 = auto).",
    )
    gpu_device_id: int = Field(
        default=-1,
        description="OpenCL/CUDA device id (-1 = auto).",
    )
    params: dict[str, Any] = Field(
        default_factory=lambda: {
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": -1,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            # Mild L1/L2 regularisation by default to curb overfitting on noisy
            # crypto features (previously 0.0 = unregularised).
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "min_gain_to_split": 0.0,
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
        embargo: Number of bars purged between the end of each training window and the
            start of the prediction window. Because the target is a forward return over
            ``prediction.horizon`` bars, the final ``horizon`` training labels otherwise
            overlap the prediction window and leak future information. ``None`` resolves
            to ``prediction.horizon`` (the correct gap); ``0`` disables purging (legacy).
        max_steps: Optional cap on the number of walk-forward steps (useful for
            quick smoke runs / demos).
    """

    initial_train_period: int = 2000
    step_size: int = 200
    retrain_frequency: int = 1
    expanding: bool = True
    recency_half_life: int | None = None
    embargo: int | None = Field(
        default=None,
        ge=0,
        description="Bars purged between train and prediction windows (None = horizon).",
    )
    max_steps: int | None = None


class PromotionConfig(BaseModel):
    """Challenger/champion gate for safe **automated** retraining.

    An automated retrain trains a *challenger* on data up to a recent cutoff, scores it
    and the current *champion* on a held-out tail of bars neither trained on, and only
    repoints the registry's promoted model when the challenger improves the chosen
    metric by at least ``min_improvement``. This keeps a self-updating loop from
    silently shipping a worse model.

    Attributes:
        eval_bars: Most-recent resolved bars held out to score challenger vs champion.
        metric: Decision metric (see :mod:`epoch_ai.learning.step_metrics`). Lower is
            better for ``oos_logloss``/``oos_brier``/``oos_rmse``; higher is better for
            the accuracy/AUC metrics.
        min_improvement: Minimum improvement (in metric units, sign-aware) the
            challenger must show over the champion to be promoted. The improvement
            must also be strictly positive, so ``0`` promotes only on a genuine
            (non-tie) improvement rather than on any non-worse result.
    """

    eval_bars: int = Field(
        default=2000,
        ge=1,
        description="Most-recent resolved bars held out to compare challenger vs champion.",
    )
    metric: Literal[
        "oos_logloss",
        "oos_brier",
        "oos_accuracy",
        "oos_auc",
        "oos_directional_accuracy",
        "oos_rmse",
    ] = "oos_logloss"
    min_improvement: float = Field(
        default=0.0,
        ge=0.0,
        description="Required sign-aware improvement over the champion to promote.",
    )


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
    cold_storage_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Additional fraction of session profits moved to cold storage.",
    )
    max_daily_profit_take: float | None = Field(
        default=None,
        ge=0.0,
        description="Cap total daily profit withdrawal (reserved + cold storage).",
    )
    treasury_state_path: str = "artifacts/treasury.json"
    kill_switch_path: str = "artifacts/kill_switch.json"
    audit_log_path: str = "artifacts/audit/trades.jsonl"
    metrics_path: str = "artifacts/metrics/runtime.jsonl"
    audit_enabled: bool = True
    metrics_enabled: bool = True
    calibration_min_accuracy: float | None = None
    calibration_min_samples: int = 30
    min_buffer_bars: int = 500

    @model_validator(mode="after")
    def _validate_allocation_fractions(self) -> ExecutionConfig:
        if self.reserve_fraction + self.cold_storage_fraction > 1.0:
            raise ValueError(
                "reserve_fraction + cold_storage_fraction must not exceed 1.0."
            )
        return self


class ApiConfig(BaseModel):
    """HTTP API server settings (FastAPI)."""

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class TelegramConfig(BaseModel):
    """Optional Telegram bot integration."""

    enabled: bool = False
    token_env: str = "EPOCH_AI_TELEGRAM_TOKEN"
    allowed_chat_ids: list[int] = Field(default_factory=list)


class BacktestConfig(BaseModel):
    """Backtester settings.

    Attributes:
        horizon_aware: When ``True`` the equity simulation holds each signal for the
            full ``prediction.horizon`` (overlapping positions are averaged), so the
            backtest measures the same horizon the model was trained to predict.
            When ``False`` it uses the legacy single-bar-ahead return.
    """

    use_vectorbt: bool = False
    annualization_factor: int | None = None
    horizon_aware: bool = True


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
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
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
        # The purge gap (defaulting to the horizon) must leave training rows behind.
        embargo = self.walk_forward.embargo
        resolved_embargo = self.prediction.horizon if embargo is None else embargo
        if self.walk_forward.initial_train_period <= resolved_embargo:
            raise ValueError(
                "walk_forward.initial_train_period must exceed the embargo gap "
                f"({resolved_embargo}); otherwise the first training window is empty."
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
