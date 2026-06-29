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
        default_factory=lambda: ["ETH/USDT", "SOL/USDT"],
        description=(
            "Additional symbols whose OHLCV/derivatives are joined onto the primary "
            "frame as context columns (e.g. eth_close, sol_funding_rate for cross-asset)."
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
    synthesize_market_extensions: bool = Field(
        default=False,
        description=(
            "Synthesise derived/macro/on-chain proxy columns when absent. OFF by "
            "default: the proxies are price-derived or pure noise and must not be "
            "fed to the model as if they were real feeds. Enable only for offline "
            "pipeline demos/tests that need the full column set populated."
        ),
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
    patterns: bool = Field(
        default=False,
        description="Enable classic chart-pattern geometry features (secondary signal).",
    )
    manipulation: bool = Field(
        default=False,
        description="Enable rug-pull/manipulation proxy features from OHLCV and derivatives.",
    )
    higher_timeframe: bool = Field(
        default=True,
        description="Enable higher-timeframe (1h/4h) context features on the bar grid.",
    )
    macro: bool = Field(
        default=True,
        description="Enable macro/cross-market features (dominance, DXY, VIX, etc.).",
    )
    dropna: bool = True

    htf_timeframes: list[str] = Field(
        default_factory=lambda: ["1h", "4h"],
        description="Higher-timeframe rules passed to pandas resample (e.g. 1h, 4h).",
    )

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
    pattern_lookbacks: list[int] = Field(
        default_factory=lambda: [48, 96, 192],
        description="Look-back windows (bars) for pattern geometry scoring.",
    )
    pivot_confirm_bars: int = Field(
        default=3,
        ge=1,
        description="Bars after a candidate pivot before it is treated as confirmed (causal lag).",
    )

    @model_validator(mode="after")
    def _validate_windows(self) -> FeatureConfig:
        # Window lists must be non-empty positive integers so feature groups emit
        # a stable set of columns; an empty list would silently drop a sub-family.
        for name in (
            "return_lags",
            "ma_windows",
            "rsi_periods",
            "vol_windows",
            "pattern_lookbacks",
        ):
            values = getattr(self, name)
            if not values or any(int(v) < 1 for v in values):
                raise ValueError(f"features.{name} must be a non-empty list of positive ints.")
        if not self.htf_timeframes:
            raise ValueError("features.htf_timeframes must be a non-empty list.")
        return self


class PredictionConfig(BaseModel):
    """Defines the supervised-learning target.

    Attributes:
        horizon: Primary forward horizon (in candles) used for legacy single-head
            paths, outcome logging, and backtest horizon-aware simulation.
        horizons: Multi-horizon candle counts for simultaneous forecasting (e.g.
            ``[1, 5, 10, 15, 30, 60]`` on a 1m base). When empty, resolves to
            ``[horizon]`` (single-horizon mode).
        quantiles: Quantile levels for return bands (must include ``0.5``).
        task: ``"classification"`` predicts P(up); ``"regression"`` predicts return.
        threshold: Forward return above which a candle is labelled "up"
            (classification) - a small positive value can encode a neutral band.
        neutral_band: Symmetric dead-zone (in forward-return units) around
            ``threshold`` for classification. Bars whose realised forward return falls
            inside ``[threshold - neutral_band, threshold + neutral_band]`` are dropped
            (NaN target) instead of being labelled, so the model learns from decisive
            moves rather than near-zero noise. ``0`` keeps every bar (legacy behaviour).
    """

    horizon: int = 60
    horizons: list[int] = Field(
        default_factory=lambda: [1, 5, 10, 15, 30, 60],
        description="Multi-horizon candle counts; empty list => single-horizon [horizon].",
    )
    quantiles: list[float] = Field(
        default_factory=lambda: [0.1, 0.5, 0.9],
        description="Quantile levels for return bands; must include 0.5 (median).",
    )
    task: Literal["classification", "regression"] = "classification"
    threshold: float = 0.0
    neutral_band: float = Field(
        default=0.0,
        ge=0.0,
        description="Dead-zone half-width around threshold; ambiguous bars are dropped.",
    )

    @model_validator(mode="after")
    def _normalize_horizons(self) -> PredictionConfig:
        if not self.horizons:
            self.horizons = [self.horizon]
        else:
            self.horizons = sorted(set(int(h) for h in self.horizons))
            if any(h < 1 for h in self.horizons):
                raise ValueError("prediction.horizons must contain positive integers.")
        if self.horizon not in self.horizons:
            raise ValueError("prediction.horizon must be one of prediction.horizons.")
        qs = sorted(set(float(q) for q in self.quantiles))
        if not qs or any(q <= 0.0 or q >= 1.0 for q in qs):
            raise ValueError("prediction.quantiles must be strictly between 0 and 1.")
        if 0.5 not in qs:
            raise ValueError("prediction.quantiles must include 0.5 (median).")
        self.quantiles = qs
        return self

    @property
    def max_horizon(self) -> int:
        """Longest configured horizon (purge/embargo and label tail)."""
        return max(self.horizons)

    def horizon_label(self, horizon: int) -> str:
        """Human label for a horizon candle count (e.g. ``60`` -> ``1hr`` on 1m base)."""
        labels = {1: "1m", 5: "5m", 10: "10m", 15: "15m", 30: "30m", 60: "1hr"}
        return labels.get(horizon, f"{horizon}b")

    @property
    def n_outputs(self) -> int:
        """Flat output width for multi-head models: horizons x (quantiles + direction)."""
        return len(self.horizons) * (len(self.quantiles) + 1)

    def resolved_embargo(self, embargo: int | None) -> int:
        """Purge gap between train and predict windows (defaults to ``max_horizon``)."""
        return self.max_horizon if embargo is None else embargo


class EvolutionConfig(BaseModel):
    """Evolutionary architecture search for ``evolved_nn`` backend."""

    enabled: bool = Field(
        default=True,
        description="When false, train the default MLP genome without evolution.",
    )
    population_size: int = Field(default=12, ge=2, description="Candidates per generation.")
    generations: int = Field(default=8, ge=1, description="Evolutionary generations per fit().")
    elite_fraction: float = Field(
        default=0.25,
        ge=0.05,
        le=0.5,
        description="Top fraction kept unchanged each generation.",
    )
    mutation_sigma: float = Field(
        default=0.2,
        ge=0.01,
        le=1.0,
        description="Relative mutation strength for offspring genomes.",
    )
    seed: int = Field(default=42, description="RNG seed for reproducible evolution.")
    fast_fit: bool = Field(
        default=False,
        description="Skip evolution and train a fixed default MLP (tests / quick smokes).",
    )
    parallel_candidates: bool = Field(
        default=True,
        description="Train population candidates concurrently (thread pool per device).",
    )
    max_workers: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Max parallel candidate trainers (null = auto: ~4 on CUDA, else CPU count)."
        ),
    )
    early_stop_patience: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Stop evolution early after this many generations without best-fitness "
            "improvement (null = run all generations)."
        ),
    )


class NNConfig(BaseModel):
    """PyTorch MLP training limits for ``evolved_nn``."""

    min_layers: int = Field(
        default=1,
        ge=1,
        le=12,
        description=(
            "Minimum hidden layers evolution may produce. Raise with max_layers to "
            "bias search toward deeper networks."
        ),
    )
    max_layers: int = Field(
        default=3,
        ge=1,
        le=12,
        description="Maximum hidden layers (depth ceiling for random genomes and mutation).",
    )
    hidden_size_min: int = Field(default=32, ge=8)
    hidden_size_max: int = Field(default=512, ge=32)
    fixed_hidden_sizes: list[int] | None = Field(
        default=None,
        description=(
            "Optional fixed layer widths (e.g. [256, 256, 128, 64]). When set, seeds "
            "default_genome and random_genome; evolution still mutates around this shape."
        ),
    )
    max_epochs: int = Field(default=200, ge=10)
    batch_size: int = Field(default=256, ge=16)
    patience: int = Field(default=15, ge=1, description="Early-stopping patience on val loss.")
    compute_importance: bool = Field(
        default=True,
        description=(
            "When true, permutation importance runs on the final walk-forward fit only "
            "(skipped on intermediate retrains for speed)."
        ),
    )
    mixed_precision: bool = Field(
        default=True,
        description="Use torch.autocast on CUDA during candidate training.",
    )
    torch_compile: bool = Field(
        default=True,
        description="Wrap MLP candidates with torch.compile when PyTorch 2+ is available.",
    )

    @model_validator(mode="after")
    def _validate_layer_bounds(self) -> NNConfig:
        if self.min_layers > self.max_layers:
            raise ValueError("model.nn.min_layers must be <= model.nn.max_layers")
        if self.hidden_size_min > self.hidden_size_max:
            raise ValueError("model.nn.hidden_size_min must be <= hidden_size_max")
        if self.fixed_hidden_sizes:
            depth = len(self.fixed_hidden_sizes)
            if depth < self.min_layers or depth > self.max_layers:
                raise ValueError(
                    "model.nn.fixed_hidden_sizes length must be between min_layers and max_layers"
                )
            for width in self.fixed_hidden_sizes:
                if width < self.hidden_size_min or width > self.hidden_size_max:
                    raise ValueError(
                        "each model.nn.fixed_hidden_sizes entry must be within "
                        "[hidden_size_min, hidden_size_max]"
                    )
        return self


class ModelConfig(BaseModel):
    """Model hyper-parameters, calibration and model-registry location.

    Attributes:
        val_fraction: Fraction of the most-recent training rows held out (time-ordered)
            for early stopping and probability calibration. ``0`` disables both.
        class_weight: ``"balanced"`` derives positive-class weight from label balance;
            ``"none"`` leaves the loss unweighted.
        calibration: Post-hoc probability calibration fit on the validation tail —
            ``"isotonic"`` (non-parametric, monotone), ``"sigmoid"`` (Platt scaling)
            or ``"none"``. Only applies to classification.
        refit_full_after_es: When early stopping is active, refit on the **full**
            training window for the early-stopping-selected epoch count.
        backend: ``"evolved_nn"`` (default, evolutionary PyTorch MLP),
            ``"lightgbm"``, or ``"xgboost"`` (optional GBM backends).
        evolution: Evolutionary search knobs (``evolved_nn`` only).
        nn: MLP training limits (``evolved_nn`` only).
        device: ``"auto"`` (default, CUDA when available), ``"cpu"``, ``"gpu"`` or ``"cuda"``.
        gpu_platform_id: OpenCL platform id for LightGBM ``device="gpu"`` (``-1`` = auto).
        gpu_device_id: OpenCL/CUDA device ordinal (``-1`` = auto).
        retain_versions: When set, prune oldest ``v_*`` directories after each
            :meth:`ModelRegistry.save`, keeping this many recent versions (plus any
            protected labels such as the champion or walk-forward checkpoint model).
            ``None`` disables automatic pruning.
    """

    model_dir: str = "artifacts/models"
    backend: Literal["evolved_nn", "lightgbm", "xgboost"] = Field(
        default="evolved_nn",
        description="Prediction backend; evolved_nn uses evolutionary PyTorch MLP.",
    )
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    nn: NNConfig = Field(default_factory=NNConfig)
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
    device: Literal["auto", "cpu", "gpu", "cuda"] = Field(
        default="auto",
        description="Compute device; auto picks CUDA when available; gpu/cuda fall back to cpu.",
    )
    gpu_platform_id: int = Field(
        default=-1,
        description="OpenCL platform id for device='gpu' (-1 = auto).",
    )
    gpu_device_id: int = Field(
        default=-1,
        description="OpenCL/CUDA device id (-1 = auto).",
    )
    retain_versions: int | None = Field(
        default=10,
        ge=1,
        description=(
            "Auto-prune registry to this many recent v_* versions after each save "
            "(champion and checkpoint models are always kept). None disables pruning."
        ),
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
        checkpoint_enabled: When ``True``, ``train`` persists progress after each step
            so a later run can resume with ``--resume`` (default).
        checkpoint_path: Optional JSON checkpoint file (``null`` = per-symbol file under
            ``artifacts/checkpoints/``).
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
    checkpoint_enabled: bool = True
    checkpoint_path: str | None = Field(
        default=None,
        description="Walk-forward resume checkpoint JSON (null = default per-symbol path).",
    )


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
        "oos_brier_weighted",
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


class SafetyConfig(BaseModel):
    """Pre-trade manipulation/rug-risk gate (execution layer only)."""

    enabled: bool = False
    max_suspicion_score: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Block or flatten when combined suspicion exceeds this score.",
    )
    scale_weight_by_suspicion: bool = Field(
        default=True,
        description="When True, linearly reduce target_weight as suspicion rises.",
    )
    block_on_missing_onchain: bool = Field(
        default=False,
        description="When True and symbol expects on-chain cols, missing data => max suspicion.",
    )


class RiskConfig(BaseModel):
    """Risk-management parameters used by the (separate) execution layer."""

    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.02
    max_leverage: float = 3.0
    long_threshold: float = 0.58
    short_threshold: float = 0.42
    fee_rate: float = 0.0004
    slippage: float = 0.0002
    allow_short: bool = True
    min_confidence: float = 0.0
    max_drawdown_halt: float | None = None
    max_daily_loss: float | None = None
    cooldown_bars: int = 0


class TradingConfig(BaseModel):
    """Learned/heuristic trading policy settings (execution layer only)."""

    policy_backend: Literal[
        "threshold",
        "baseline",
        "learned",
        "learned_with_baseline_fallback",
    ] = Field(
        default="baseline",
        description="threshold=RiskManager; baseline=ensemble; learned=PPO; fallback=learned then baseline.",
    )
    reliability_floor: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Drop forecast heads below this confidence for policy observations.",
    )
    max_position_fraction: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Hard cap on abs(target_weight) as a fraction of max_leverage.",
    )
    max_drawdown_kill: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Force flat when peak-to-trough drawdown exceeds this fraction.",
    )
    max_hold_bars: int = Field(
        default=1440,
        ge=1,
        description="Force-close positions held longer than this many bars (~1 day at 1m).",
    )
    funding_rate_per_bar: float = Field(
        default=0.0,
        description="Approximate perp funding accrual per bar on open positions.",
    )
    trade_frequency: Literal["selective", "active"] = Field(
        default="selective",
        description="Selective policies trade less often; active allows tighter dead bands.",
    )
    decision_horizons: list[int] = Field(
        default_factory=list,
        description="Horizons fed to the policy; empty uses all configured prediction horizons.",
    )
    action_log_path: str = "artifacts/logs/action_log.jsonl"
    session_state_path: str = "artifacts/session_state.json"


class PolicyPromotionConfig(BaseModel):
    """Challenger/champion gate for the learned PPO trading policy."""

    enabled: bool = True
    eval_bars: int | None = Field(
        default=None,
        ge=1,
        description="Holdout bars for policy eval (null = promotion.eval_bars).",
    )
    metric: Literal["risk_adjusted_return", "sharpe", "total_return"] = Field(
        default="risk_adjusted_return",
        description="Primary gate metric (higher is better).",
    )
    min_improvement: float = Field(
        default=0.0,
        ge=0.0,
        description="Required improvement over champion to promote.",
    )
    champion_path: str = "artifacts/policy/champion.pt"
    require_beat_baseline: bool = True
    require_beat_buy_hold: bool = True


class RLConfig(BaseModel):
    """PPO hyperparameters and artifact paths for the learned policy."""

    enabled: bool = False
    policy_path: str = "artifacts/policy/ppo_policy.pt"
    hidden_sizes: list[int] = Field(default_factory=lambda: [64, 32])
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    train_epochs: int = 4
    rollout_steps: int = 256
    total_updates: int = 50
    drawdown_penalty: float = 0.5
    sharpe_scale: float = 1.0
    device: Literal["auto", "cpu", "cuda"] = "auto"
    promotion: PolicyPromotionConfig = Field(default_factory=PolicyPromotionConfig)


class AdaptationConfig(BaseModel):
    """Post-initial coarse retrain cadence and action-log feedback knobs."""

    enabled: bool = True
    coarse_step_size: int = Field(
        default=4320,
        ge=1,
        description="Walk-forward step size for scheduled auto-retrain (~3 days at 1m).",
    )
    coarse_retrain_frequency: int = Field(
        default=1,
        ge=1,
        description="Retrain every N coarse steps during scheduled auto-retrain.",
    )
    schedule_interval_hours: float = Field(
        default=24.0,
        gt=0.0,
        description="Default sleep between schedule-retrain cycles.",
    )
    holdout_bars: int | None = Field(
        default=None,
        ge=1,
        description="Final holdout slice (null = promotion.eval_bars).",
    )
    use_action_log_for_retrain: bool = True
    action_log_min_rows: int = Field(
        default=50,
        ge=1,
        description="Minimum action-log rows before boosting live-experience weights.",
    )
    action_log_weight_boost: float = Field(
        default=2.0,
        ge=1.0,
        description="Multiply sample weights for bars present in the action log.",
    )

    def resolved_holdout_bars(self, promotion: PromotionConfig) -> int:
        """Holdout size shared by predictor retrain, policy eval, and acceptance."""
        return self.holdout_bars if self.holdout_bars is not None else promotion.eval_bars


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
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    adaptation: AdaptationConfig = Field(default_factory=AdaptationConfig)
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
        max_h = self.prediction.max_horizon
        if self.walk_forward.step_size < 1:
            raise ValueError("walk_forward.step_size must be >= 1 candle.")
        if self.walk_forward.initial_train_period < max_h + 1:
            raise ValueError(
                "walk_forward.initial_train_period must exceed prediction.max_horizon."
            )
        # The purge gap (defaulting to max horizon) must leave training rows behind.
        embargo = self.walk_forward.embargo
        resolved_embargo = self.prediction.resolved_embargo(embargo)
        if self.walk_forward.initial_train_period <= resolved_embargo:
            raise ValueError(
                "walk_forward.initial_train_period must exceed the embargo gap "
                f"({resolved_embargo}); otherwise the first training window is empty."
            )
        # Human: evolved_nn retrains are costly; default walk-forward cadence is slower
        #        than LightGBM unless the user explicitly configured walk_forward.
        # Agent: MUTATES retrain_frequency=5; ONLY when walk_forward not in fields_set.
        if self.model.backend == "evolved_nn" and "walk_forward" not in self.model_fields_set:
            self.walk_forward.retrain_frequency = 5
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
