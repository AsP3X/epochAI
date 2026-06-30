"""Walk-forward training progress without running inference."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from epoch_ai.config.settings import AppConfig
from epoch_ai.learning.checkpoint import (
    WalkForwardCheckpoint,
    load_checkpoint,
    resolve_checkpoint_path,
    validate_checkpoint,
)
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.models.registry import ModelRegistry
from epoch_ai.utils.progress import build_fraction_bar


@dataclass(slots=True)
class LogProgressSummary:
    """Aggregate stats from the SQLite prediction store."""

    predictions: int
    outcomes: int
    pending: int
    oos_accuracy: float | None
    oos_logloss: float | None


@dataclass(slots=True)
class TrainingProgressReport:
    """Snapshot of walk-forward position and remaining work."""

    symbol: str
    timeframe: str
    resolved_rows: int
    total_steps: int
    completed_steps: int
    remaining_steps: int
    percent_complete: float
    next_step_idx: int | None
    cutoff: int | None
    checkpoint_path: Path
    checkpoint: WalkForwardCheckpoint | None
    checkpoint_compatible: bool | None
    model_version: str | None
    registry_versions: int
    latest_registry_version: str | None
    promoted_version: str | None
    retain_versions: int | None
    log_summary: LogProgressSummary | None
    status: str
    inferred_from_registry: bool = False
    needs_refresh_rows: bool = False


def estimate_total_walk_forward_steps(
    resolved_rows: int,
    *,
    initial_train_period: int,
    step_size: int,
    max_steps: int | None = None,
) -> int:
    """Return how many walk-forward steps ``resolved_rows`` supports.

    Mirrors the termination logic in :class:`~epoch_ai.learning.progressive.ProgressiveLearningEngine`.
    """
    if resolved_rows <= initial_train_period or step_size < 1:
        return 0
    cutoff = initial_train_period
    step_idx = 0
    while cutoff < resolved_rows:
        if max_steps is not None and step_idx >= max_steps:
            break
        cutoff = min(cutoff + step_size, resolved_rows)
        step_idx += 1
    return step_idx


def _registry_n_features(registry: ModelRegistry, label: str | None) -> int | None:
    if not label:
        return None
    meta_path = registry.base_dir / label / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return int(meta["n_features"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        return None


def _infer_progress_from_registry(
    registry: ModelRegistry,
    config: AppConfig,
) -> tuple[int, int, str] | None:
    """Guess resume position from the latest registry model when no checkpoint exists."""
    latest = registry.latest_label()
    if not latest:
        return None
    meta_path = registry.base_dir / latest / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        step = meta.get("step")
        if step is None:
            return None
        next_step_idx = int(step) + 1
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        return None

    wf = config.walk_forward
    cutoff = wf.initial_train_period + next_step_idx * wf.step_size
    return next_step_idx, cutoff, latest


def count_resolved_rows(config: AppConfig, n_bars: int | None = None) -> tuple[int, int]:
    """Load cached market data and return ``(resolved_rows, n_features)``.

    Never hits the exchange — use ``download`` first if cache is missing or too small.
    """
    from epoch_ai.data.downloader import HistoricalDownloader
    from epoch_ai.data.training_policy import config_for_supervised_training
    from epoch_ai.features.pipeline import FeaturePipeline, build_target, forward_return
    from epoch_ai.services.training import resolve_training_bars

    cfg = config_for_supervised_training(config)

    n_bars = resolve_training_bars(cfg, n_bars, full_history=False)
    market = HistoricalDownloader(cfg).load_or_download(
        cfg.primary_symbol,
        n_bars=n_bars,
        fetch_if_missing=False,
    )
    features = FeaturePipeline(cfg).transform(market)
    y = build_target(market, cfg.prediction)
    fwd = forward_return(market, cfg.prediction.horizon)
    resolved = features.join(y).join(fwd).dropna(subset=["target", "forward_return"])
    return len(resolved), len(features.columns)


def _sqlite_log_summary(config: AppConfig) -> LogProgressSummary | None:
    db_path = Path(config.logging.db_path)
    if not db_path.exists():
        return None

    store = PredictionStore(str(db_path))
    try:
        counts = store.counts()
        preds = store.predictions_frame(config.primary_symbol)
        outs = store.outcomes_frame()
        if preds.empty or outs.empty:
            pending = counts["predictions"] - counts["outcomes"]
            return LogProgressSummary(
                predictions=counts["predictions"],
                outcomes=counts["outcomes"],
                pending=max(pending, 0),
                oos_accuracy=None,
                oos_logloss=None,
            )

        merged = preds.merge(outs, left_on="id", right_on="prediction_id", how="inner")
        if merged.empty:
            pending = counts["predictions"] - counts["outcomes"]
            return LogProgressSummary(
                predictions=counts["predictions"],
                outcomes=counts["outcomes"],
                pending=max(pending, 0),
                oos_accuracy=None,
                oos_logloss=None,
            )

        if config.prediction.task == "classification":
            import numpy as np

            pred_up = (merged["prediction"] >= 0.5).astype(int)
            accuracy = float((pred_up == merged["realized_label"]).mean())
            eps = 1e-15
            p = np.clip(merged["prediction"].to_numpy(), eps, 1.0 - eps)
            y = merged["realized_label"].to_numpy(dtype=float)
            logloss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        else:
            pred_up = (merged["prediction"] > 0.0).astype(int)
            accuracy = float((pred_up == merged["realized_label"]).mean())
            logloss = None

        pending = int((~preds["id"].isin(set(outs["prediction_id"]))).sum())
        return LogProgressSummary(
            predictions=counts["predictions"],
            outcomes=counts["outcomes"],
            pending=pending,
            oos_accuracy=accuracy,
            oos_logloss=logloss,
        )
    finally:
        store.close()


def gather_training_progress(
    config: AppConfig,
    *,
    n_bars: int | None = None,
    refresh_rows: bool = False,
    cached_resolved_rows: int | None = None,
) -> TrainingProgressReport:
    """Build a progress snapshot from checkpoint, registry, and optional SQLite logs."""
    wf = config.walk_forward
    checkpoint_path = resolve_checkpoint_path(config)
    checkpoint = load_checkpoint(checkpoint_path)
    registry = ModelRegistry(config.model.model_dir)

    resolved_rows: int | None = None
    n_features: int | None = None
    checkpoint_compatible: bool | None = None
    inferred_from_registry = False
    needs_refresh_rows = False

    if cached_resolved_rows is not None:
        resolved_rows = cached_resolved_rows
        n_features = _registry_n_features(registry, checkpoint.model_version if checkpoint else None)
    elif checkpoint is not None:
        resolved_rows = checkpoint.resolved_rows
        n_features = _registry_n_features(registry, checkpoint.model_version)
    elif refresh_rows:
        resolved_rows, n_features = count_resolved_rows(config, n_bars=n_bars)
    else:
        needs_refresh_rows = True

    if checkpoint is not None and n_features is not None:
        assert resolved_rows is not None
        try:
            validate_checkpoint(checkpoint, config, n_features, resolved_rows)
            checkpoint_compatible = True
        except ValueError:
            checkpoint_compatible = False

    inferred: tuple[int, int, str] | None = None
    if checkpoint is None:
        inferred = _infer_progress_from_registry(registry, config)

    total_steps = 0
    if resolved_rows is not None:
        total_steps = estimate_total_walk_forward_steps(
            resolved_rows,
            initial_train_period=wf.initial_train_period,
            step_size=wf.step_size,
            max_steps=wf.max_steps,
        )

    if checkpoint is None and inferred is None:
        completed_steps = 0
        next_step_idx = 0
        cutoff = wf.initial_train_period
        model_version = None
        status = "not_started"
    elif checkpoint is None and inferred is not None:
        next_step_idx, cutoff, model_version = inferred
        completed_steps = next_step_idx
        inferred_from_registry = True
        status = "in_progress"
    elif checkpoint is not None and checkpoint.completed:
        completed_steps = total_steps
        next_step_idx = None
        cutoff = None
        model_version = checkpoint.model_version
        status = "completed"
    else:
        assert checkpoint is not None
        completed_steps = checkpoint.step_idx
        next_step_idx = checkpoint.step_idx
        cutoff = checkpoint.cutoff
        model_version = checkpoint.model_version
        status = "in_progress"

    remaining_steps = max(total_steps - completed_steps, 0) if total_steps else 0
    percent_complete = (
        (completed_steps / total_steps * 100.0) if total_steps else 0.0
    )

    versions = registry.list_versions()
    latest = registry.latest_label()
    promoted = registry.promoted_label()

    return TrainingProgressReport(
        symbol=config.primary_symbol,
        timeframe=config.timeframe,
        resolved_rows=resolved_rows or 0,
        total_steps=total_steps,
        completed_steps=completed_steps,
        remaining_steps=remaining_steps,
        percent_complete=percent_complete,
        next_step_idx=next_step_idx,
        cutoff=cutoff,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        checkpoint_compatible=checkpoint_compatible,
        model_version=model_version,
        registry_versions=len(versions),
        latest_registry_version=latest,
        promoted_version=promoted,
        retain_versions=config.model.retain_versions,
        log_summary=_sqlite_log_summary(config),
        status=status,
        inferred_from_registry=inferred_from_registry,
        needs_refresh_rows=needs_refresh_rows and not inferred_from_registry,
    )


def _format_timestamp(iso_ts: str | None) -> str:
    if not iso_ts:
        return "unknown"
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    except ValueError:
        return iso_ts


def format_training_progress(
    report: TrainingProgressReport,
    *,
    watch: bool = False,
    eta_seconds: float | None = None,
    step_rate_per_min: float | None = None,
) -> str:
    """Render ``report`` as a human-readable multi-line summary."""
    title = (
        "=== Walk-forward training progress (live) ==="
        if watch
        else "=== Walk-forward training progress ==="
    )
    lines: list[str] = [title] if watch else ["", title]
    if watch:
        lines.append(f"Updated             : {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}")

    lines.append(f"Symbol              : {report.symbol} ({report.timeframe})")

    if report.total_steps:
        bar = build_fraction_bar(report.completed_steps, report.total_steps)
        bar_line = (
            f"{bar} {report.completed_steps:,}/{report.total_steps:,} "
            f"{report.percent_complete:5.1f}%"
        )
        if watch and eta_seconds is not None and eta_seconds > 0:
            from epoch_ai.utils.progress import _short_eta

            rate_text = (
                f"{step_rate_per_min:.2f} steps/min"
                if step_rate_per_min and step_rate_per_min > 0
                else "-- steps/min"
            )
            bar_line += f"  ETA {_short_eta(eta_seconds)} @ {rate_text}"
        lines.append(bar_line)
    else:
        lines.append(
            f"Walk-forward steps  : {report.completed_steps:,} completed "
            f"(total unknown - pass --refresh-rows)"
        )

    if report.resolved_rows:
        lines.append(f"Resolved rows       : {report.resolved_rows:,}")
    elif report.needs_refresh_rows:
        lines.append("Resolved rows       : unknown (pass --refresh-rows to compute)")

    if report.total_steps:
        if not watch:
            lines.append(
                f"Walk-forward steps  : {report.completed_steps:,} / {report.total_steps:,} "
                f"({report.percent_complete:.1f}%)"
            )
        lines.append(f"Steps remaining     : {report.remaining_steps:,}")

    if report.status == "not_started":
        lines.append("Status              : not started (no checkpoint on disk)")
    elif report.status == "completed":
        lines.append("Status              : completed")
    else:
        if report.inferred_from_registry:
            lines.append("Status              : in progress (inferred from registry; no checkpoint)")
        else:
            lines.append("Status              : in progress (resume available)")
        lines.append(f"Next step           : {report.next_step_idx}")
        if report.cutoff is not None:
            lines.append(f"Cutoff index        : {report.cutoff:,}")
        if report.model_version:
            lines.append(f"Checkpoint model    : {report.model_version}")

    lines.append(f"Checkpoint file     : {report.checkpoint_path}")
    if report.checkpoint is not None:
        lines.append(f"Checkpoint updated  : {_format_timestamp(report.checkpoint.updated_at)}")
        if report.checkpoint_compatible is False:
            lines.append(
                "Checkpoint config   : MISMATCH - run with --fresh or restore matching config"
            )
        elif report.checkpoint_compatible:
            lines.append("Checkpoint config   : compatible with current YAML")
    else:
        lines.append("Checkpoint on disk  : none")

    lines.append(
        f"Registry versions   : {report.registry_versions}"
        + (f" (retain {report.retain_versions})" if report.retain_versions else "")
    )
    if report.latest_registry_version:
        lines.append(f"Latest model        : {report.latest_registry_version}")
    if report.promoted_version:
        lines.append(f"Champion model      : {report.promoted_version}")

    if report.log_summary is not None:
        ls = report.log_summary
        lines.append(
            f"SQLite predictions  : {ls.predictions:,} "
            f"(outcomes {ls.outcomes:,}, pending {ls.pending:,})"
        )
        if ls.oos_accuracy is not None:
            acc_line = f"Logged OOS accuracy : {ls.oos_accuracy:.3f}"
            if ls.oos_logloss is not None:
                acc_line += f"  logloss {ls.oos_logloss:.4f}"
            lines.append(acc_line)

    if report.status == "in_progress":
        lines.extend(["", "Resume with:"])
        if report.inferred_from_registry and report.checkpoint is None:
            last_completed = (report.next_step_idx or 1) - 1
            lines.append(
                f"  python -m epoch_ai checkpoint seed --last-step {last_completed}"
            )
            lines.append("  python -m epoch_ai train --log-predictions --set model.device=cuda")
        else:
            lines.append("  python -m epoch_ai train --log-predictions --set model.device=cuda")
    elif report.status == "not_started":
        lines.extend(
            [
                "",
                "Start training with:",
                "  python -m epoch_ai train --log-predictions --set model.device=cuda",
            ]
        )

    if watch:
        lines.extend(["", "Press Ctrl+C to exit."])

    return "\n".join(lines)


def _resolve_cached_rows(
    config: AppConfig,
    *,
    n_bars: int | None,
    refresh_rows: bool,
) -> int | None:
    checkpoint = load_checkpoint(resolve_checkpoint_path(config))
    if refresh_rows:
        rows, _ = count_resolved_rows(config, n_bars=n_bars)
        return rows
    if checkpoint is not None:
        return checkpoint.resolved_rows
    return None


def watch_training_progress(
    config: AppConfig,
    *,
    interval: float = 2.0,
    n_bars: int | None = None,
    refresh_rows: bool = False,
) -> int:
    """Poll checkpoint/registry/SQLite and redraw progress until interrupted."""
    from epoch_ai.utils.progress import render_live_text

    if interval <= 0:
        raise ValueError("interval must be > 0")

    cached_rows = _resolve_cached_rows(config, n_bars=n_bars, refresh_rows=refresh_rows)
    stream = sys.stdout
    last_completed: int | None = None
    last_tick = time.monotonic()
    step_rate_per_min: float | None = None

    try:
        while True:
            report = gather_training_progress(
                config,
                n_bars=n_bars,
                refresh_rows=False,
                cached_resolved_rows=cached_rows,
            )
            if cached_rows is None and report.resolved_rows:
                cached_rows = report.resolved_rows

            now = time.monotonic()
            eta_seconds: float | None = None
            if (
                last_completed is not None
                and report.completed_steps > last_completed
                and report.total_steps
            ):
                delta_steps = report.completed_steps - last_completed
                delta_time = now - last_tick
                if delta_time > 0:
                    step_rate_per_min = (delta_steps / delta_time) * 60.0
                    if step_rate_per_min > 0:
                        eta_seconds = report.remaining_steps / (step_rate_per_min / 60.0)
            last_completed = report.completed_steps
            last_tick = now

            text = format_training_progress(
                report,
                watch=True,
                eta_seconds=eta_seconds,
                step_rate_per_min=step_rate_per_min,
            )
            render_live_text(text, stream=stream)

            if report.status == "completed":
                return 0

            time.sleep(interval)
    except KeyboardInterrupt:
        if stream.isatty():
            stream.write("\n")
            stream.flush()
        return 130
