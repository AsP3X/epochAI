"""Automated retrain → evaluate → promote-if-better pipeline (safe self-updating model).

This turns "retrain on a cadence" into a guarded MLOps loop:

    1. Hold out the most-recent ``promotion.eval_bars`` resolved bars.
    2. Train a **challenger** on data up to the holdout, minus the embargo gap so its
       forward-return labels never overlap (leak) the holdout.
    3. Register the challenger as a new immutable version (lineage is always kept).
    4. Score the challenger **and** the incumbent **champion** on the same untouched
       holdout, using the configured metric.
    5. Repoint the registry's promoted ("champion") model to the challenger **only**
       when it *strictly* improves the metric by at least ``promotion.min_improvement``
       (a tie never promotes, so identically-trained clones do not churn the registry).

Because the challenger is the model that gets scored *and* promoted, the deployed model
is exactly the one whose out-of-sample quality was measured (no train/deploy mismatch).
The cost is that the promoted model does not train on the freshest ``eval_bars``; the
next cycle folds those bars into training as the window rolls forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.features.pipeline import FeaturePipeline, build_target, forward_return
from epoch_ai.learning.step_metrics import classification_step_metrics, regression_step_metrics
from epoch_ai.learning.weighting import recency_weights
from epoch_ai.models.base import BaseModel
from epoch_ai.models.factory import build_model
from epoch_ai.models.registry import ModelRegistry
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)

#: Metrics where a smaller value is better (error/loss); all others are "higher better".
_LOWER_IS_BETTER = {"oos_logloss", "oos_brier", "oos_rmse"}


@dataclass(slots=True)
class AutoPromoteResult:
    """Outcome of one automated retrain + promotion cycle."""

    challenger_label: str | None
    champion_label: str | None
    promoted: bool
    metric: str
    challenger_value: float = float("nan")
    champion_value: float = float("nan")
    train_rows: int = 0
    eval_rows: int = 0
    skipped: bool = False
    reason: str = ""
    challenger_metrics: dict[str, float] = field(default_factory=dict)
    champion_metrics: dict[str, float] = field(default_factory=dict)


def metric_higher_is_better(metric: str) -> bool:
    """Whether a larger value of ``metric`` indicates a better model."""
    return metric not in _LOWER_IS_BETTER


def decide_promotion(
    champion_value: float | None,
    challenger_value: float,
    *,
    metric: str,
    min_improvement: float,
) -> tuple[bool, str]:
    """Decide whether the challenger should replace the champion.

    Args:
        champion_value: Champion's metric on the holdout, or ``None`` when there is no
            usable champion (bootstrap case).
        challenger_value: Challenger's metric on the holdout.
        metric: Metric name (drives the better-is-higher/lower direction).
        min_improvement: Required sign-aware improvement over the champion. The
            improvement must additionally be strictly positive, so a tie (0.000000)
            is never promoted even when this floor is ``0``.

    Returns:
        ``(promote, reason)``.
    """
    if challenger_value is None or math.isnan(challenger_value):
        return False, "challenger metric is undefined (NaN); keeping champion"
    if champion_value is None or math.isnan(champion_value):
        return True, "no usable champion; promoting challenger (bootstrap)"

    # Improvement is always expressed as a positive-is-better quantity.
    if metric_higher_is_better(metric):
        improvement = challenger_value - champion_value
    else:
        improvement = champion_value - challenger_value

    # A challenger must *strictly* improve the metric to justify repointing the
    # champion. Promoting on a tie (improvement == 0, e.g. an identically-trained
    # clone) only churns the registry without any out-of-sample benefit, so a
    # non-positive improvement never promotes even when ``min_improvement`` is 0.
    if improvement <= 0.0:
        return False, (
            f"challenger does not beat champion on {metric} "
            f"(improvement {improvement:.6f} <= 0.000000)"
        )
    if improvement >= min_improvement:
        return True, (
            f"challenger improves {metric} by {improvement:.6f} "
            f"(>= {min_improvement:.6f})"
        )
    return False, (
        f"challenger does not beat champion on {metric} "
        f"(improvement {improvement:.6f} < {min_improvement:.6f})"
    )


def _resolve_metric(metric: str, metrics: dict[str, float], task: str) -> str:
    """Fall back to a task-appropriate metric when the configured one is unavailable."""
    if metric in metrics:
        return metric
    fallback = "oos_rmse" if task == "regression" else "oos_logloss"
    logger.warning(
        "Configured promotion metric %r unavailable for task %r; using %r.",
        metric,
        task,
        fallback,
    )
    return fallback


def _evaluate(
    model: BaseModel,
    x_eval,
    labels: np.ndarray,
    returns: np.ndarray,
    config: AppConfig,
) -> dict[str, float]:
    """Score ``model`` on the holdout with the task-appropriate OOS metrics."""
    preds = np.asarray(model.predict(x_eval), dtype=float)
    if config.prediction.task == "classification":
        return classification_step_metrics(
            preds,
            labels,
            long_threshold=config.risk.long_threshold,
            short_threshold=config.risk.short_threshold,
        )
    return regression_step_metrics(preds, returns)


def auto_retrain_and_promote(
    config: AppConfig,
    *,
    n_bars: int | None = None,
) -> AutoPromoteResult:
    """Train a challenger, compare it to the champion on a fresh holdout, promote if better.

    Args:
        config: Application configuration (``promotion`` drives the gate).
        n_bars: Optional cap on history depth loaded for the cycle.

    Returns:
        An :class:`AutoPromoteResult` describing the cycle (skipped when there is not
        enough data to form an honest train/holdout split).
    """
    metric = config.promotion.metric
    horizon = config.prediction.horizon
    wf = config.walk_forward
    embargo = horizon if wf.embargo is None else int(wf.embargo)

    market = HistoricalDownloader(config).load_or_download(config.primary_symbol, n_bars=n_bars)
    features = FeaturePipeline(config).transform(market)
    y = build_target(market, config.prediction)
    fwd = forward_return(market, horizon)
    data = (
        features.join(y)
        .join(fwd)
        .dropna(subset=["target", "forward_return"])
    )
    feature_cols = list(features.columns)
    n = len(data)

    eval_bars = min(config.promotion.eval_bars, max(0, n - wf.initial_train_period - embargo))
    holdout_start = n - eval_bars
    train_end = holdout_start - embargo
    if eval_bars < 1 or train_end < wf.initial_train_period:
        return AutoPromoteResult(
            challenger_label=None,
            champion_label=None,
            promoted=False,
            metric=metric,
            skipped=True,
            reason=(
                f"Insufficient data: have {n} resolved rows; need "
                f">= initial_train_period ({wf.initial_train_period}) + embargo "
                f"({embargo}) + eval_bars (>=1)."
            ),
        )

    x_train = data[feature_cols].iloc[:train_end]
    y_train = data["target"].iloc[:train_end]
    weights = recency_weights(len(x_train), wf.recency_half_life)

    x_eval = data[feature_cols].iloc[holdout_start:]
    labels_eval = data["target"].iloc[holdout_start:].to_numpy()
    returns_eval = data["forward_return"].iloc[holdout_start:].to_numpy()

    registry = ModelRegistry(config.model.model_dir)
    # Capture the incumbent BEFORE registering the challenger, so "latest" fallback does
    # not accidentally resolve to the model we are about to add.
    champion_label = registry.resolve_label(None)

    challenger = build_model(config.model, task=config.prediction.task)
    challenger.fit(x_train, y_train, sample_weight=weights)
    challenger_metrics = _evaluate(challenger, x_eval, labels_eval, returns_eval, config)
    resolved_metric = _resolve_metric(metric, challenger_metrics, config.prediction.task)
    challenger_value = float(challenger_metrics.get(resolved_metric, float("nan")))

    challenger_label = registry.save(
        challenger,
        metadata={
            "source": "auto_retrain",
            "train_rows": int(len(x_train)),
            "eval_rows": int(eval_bars),
            "embargo": embargo,
            "eval_metric": resolved_metric,
            "eval_value": challenger_value,
            "eval_metrics": challenger_metrics,
        },
        retain_versions=config.model.retain_versions,
        protect=frozenset({champion_label}) if champion_label else None,
    )

    champion_metrics: dict[str, float] = {}
    champion_value: float | None = None
    if champion_label is not None:
        try:
            champ_model, _ = registry.load(
                champion_label, config.model, task=config.prediction.task
            )
            champion_metrics = _evaluate(champ_model, x_eval, labels_eval, returns_eval, config)
            champion_value = float(champion_metrics.get(resolved_metric, float("nan")))
        except Exception as exc:  # noqa: BLE001 - a broken champion must not block updates
            logger.warning("Could not score champion %s: %s", champion_label, exc)
            champion_value = None

    promote, reason = decide_promotion(
        champion_value,
        challenger_value,
        metric=resolved_metric,
        min_improvement=config.promotion.min_improvement,
    )
    if promote:
        registry.set_promoted(
            challenger_label,
            info={
                "metric": resolved_metric,
                "value": challenger_value,
                "previous": champion_label,
                "source": "auto_retrain",
            },
        )

    logger.info(
        "Auto-retrain: challenger=%s (%s=%.6f) champion=%s (%.6f) -> %s [%s]",
        challenger_label,
        resolved_metric,
        challenger_value,
        champion_label,
        float("nan") if champion_value is None else champion_value,
        "PROMOTED" if promote else "kept champion",
        reason,
    )

    return AutoPromoteResult(
        challenger_label=challenger_label,
        champion_label=champion_label,
        promoted=promote,
        metric=resolved_metric,
        challenger_value=challenger_value,
        champion_value=float("nan") if champion_value is None else champion_value,
        train_rows=int(len(x_train)),
        eval_rows=int(eval_bars),
        reason=reason,
        challenger_metrics=challenger_metrics,
        champion_metrics=champion_metrics,
    )
