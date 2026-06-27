"""Periodic retraining from logged predictions + optional historical parquet."""

from __future__ import annotations

from dataclasses import dataclass

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.features.pipeline import FeaturePipeline, build_target
from epoch_ai.learning.weighting import recency_weights
from epoch_ai.logging_system.joiner import build_training_dataset
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.models.factory import build_model
from epoch_ai.models.registry import ModelRegistry
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class RetrainResult:
    """Outcome of a retrain job run."""

    model_version: str | None
    train_rows: int
    skipped: bool
    reason: str = ""


def run_retrain(
    config: AppConfig,
    *,
    min_new_samples: int = 50,
    register: bool = True,
    n_bars: int | None = None,
) -> RetrainResult:
    """Retrain a model from SQLite logs and/or cached historical data.

    Priority:
    1. Joined prediction/outcome rows from :class:`PredictionStore` when enough exist.
    2. Otherwise fall back to a fresh fit on downloaded/synthetic parquet history.

    Args:
        config: Application configuration.
        min_new_samples: Minimum joined log rows required before using the store path.
        register: Persist the fitted model to :class:`ModelRegistry` when ``True``.
        n_bars: Optional bar cap for the parquet fallback path.

    Returns:
        A :class:`RetrainResult` describing the run.
    """
    store = PredictionStore(config.logging.db_path)
    try:
        logged = build_training_dataset(store, config.primary_symbol)
        if len(logged) >= min_new_samples:
            # Ensure chronological order so recency weighting decays into the past.
            if "timestamp" in logged.columns:
                logged = logged.sort_values("timestamp")
            feature_cols = [
                c for c in logged.columns if c not in {"timestamp", "target", "forward_return"}
            ]
            x = logged[feature_cols]
            y = logged["target"]
            source = "sqlite_logs"
        else:
            downloader = HistoricalDownloader(config)
            market = downloader.load_or_download(config.primary_symbol, n_bars=n_bars)
            features = FeaturePipeline(config).transform(market)
            y = build_target(market, config.prediction)
            data = features.join(y).dropna(subset=["target"])
            if len(data) < config.walk_forward.initial_train_period:
                return RetrainResult(
                    model_version=None,
                    train_rows=len(data),
                    skipped=True,
                    reason=(
                        f"Insufficient data ({len(data)} rows) for retrain; "
                        f"need >= {config.walk_forward.initial_train_period}."
                    ),
                )
            x = data[features.columns]
            y = data["target"]
            source = "parquet_history"
    finally:
        store.close()

    # Emphasise recent regimes consistently with the walk-forward engine; rows are
    # chronological (parquet history is time-sorted; logs sorted above).
    weights = recency_weights(len(x), config.walk_forward.recency_half_life)
    model = build_model(config.model, task=config.prediction.task)
    model.fit(x, y, sample_weight=weights)

    version: str | None = None
    if register:
        registry = ModelRegistry(config.model.model_dir)
        version = registry.save(
            model,
            metadata={
                "source": source,
                "train_rows": len(x),
                "symbol": config.primary_symbol,
            },
            retain_versions=config.model.retain_versions,
        )

    logger.info("Retrain complete (%s): %d rows -> %s", source, len(x), version or "memory-only")
    return RetrainResult(model_version=version, train_rows=len(x), skipped=False)
