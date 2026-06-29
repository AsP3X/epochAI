"""Walk-forward training checkpoints for pause/resume."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from epoch_ai.config.settings import AppConfig
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)

# Human: Base directory for auto-named checkpoints. Patched in tests for isolation so
#        runs never leak a checkpoint into the real artifacts/ tree.
# Agent: read at call time by resolve_checkpoint_path; CONFIG walk_forward.checkpoint_path overrides.
DEFAULT_CHECKPOINT_DIR = Path("artifacts/checkpoints")


@dataclass(slots=True)
class WalkForwardCheckpoint:
    """Serializable state for resuming progressive walk-forward training."""

    step_idx: int
    cutoff: int
    model_version: str | None
    fingerprint: str
    symbol: str
    resolved_rows: int
    updated_at: str
    completed: bool = False


def resolve_checkpoint_path(config: AppConfig) -> Path:
    """Return the checkpoint file path for ``config``."""
    wf = config.walk_forward
    if wf.checkpoint_path:
        return Path(wf.checkpoint_path)
    safe_symbol = config.primary_symbol.replace("/", "-")
    return DEFAULT_CHECKPOINT_DIR / f"walk_forward_{safe_symbol}.json"


def checkpoint_fingerprint(config: AppConfig, n_features: int) -> str:
    """Hash resume-critical settings so incompatible config changes are rejected.

    ``retrain_frequency`` is intentionally excluded: it only affects future retrains,
    not step index / cutoff / feature alignment.
    """
    wf = config.walk_forward
    payload = {
        "symbol": config.primary_symbol,
        "timeframe": config.timeframe,
        "horizon": config.prediction.horizon,
        "task": config.prediction.task,
        "model_backend": config.model.backend,
        "n_features": n_features,
        "walk_forward": {
            "initial_train_period": wf.initial_train_period,
            "step_size": wf.step_size,
            "expanding": wf.expanding,
            "embargo": wf.embargo,
        },
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def legacy_checkpoint_fingerprint(
    config: AppConfig,
    n_features: int,
    *,
    retrain_frequency: int,
) -> str:
    """Pre-v2 fingerprint that included ``retrain_frequency`` (for resume migration)."""
    wf = config.walk_forward
    payload = {
        "symbol": config.primary_symbol,
        "timeframe": config.timeframe,
        "horizon": config.prediction.horizon,
        "task": config.prediction.task,
        "model_backend": config.model.backend,
        "n_features": n_features,
        "walk_forward": {
            "initial_train_period": wf.initial_train_period,
            "step_size": wf.step_size,
            "retrain_frequency": retrain_frequency,
            "expanding": wf.expanding,
            "embargo": wf.embargo,
        },
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def checkpoint_fingerprint_matches(
    state: WalkForwardCheckpoint,
    config: AppConfig,
    n_features: int,
) -> bool:
    """Return ``True`` when ``state.fingerprint`` matches current or legacy formats."""
    if state.fingerprint == checkpoint_fingerprint(config, n_features):
        return True
    wf = config.walk_forward
    legacy_candidates = {
        legacy_checkpoint_fingerprint(config, n_features, retrain_frequency=wf.retrain_frequency),
        legacy_checkpoint_fingerprint(config, n_features, retrain_frequency=1),
        legacy_checkpoint_fingerprint(config, n_features, retrain_frequency=5),
    }
    return state.fingerprint in legacy_candidates


def refresh_checkpoint_fingerprint(
    path: Path,
    config: AppConfig,
    n_features: int,
) -> WalkForwardCheckpoint | None:
    """Rewrite ``path`` with the current fingerprint when resume-critical fields still match."""
    state = load_checkpoint(path)
    if state is None:
        return None
    if state.fingerprint == checkpoint_fingerprint(config, n_features):
        return state
    if not checkpoint_fingerprint_matches(state, config, n_features):
        return None
    state.fingerprint = checkpoint_fingerprint(config, n_features)
    save_checkpoint(path, state)
    logger.info(
        "Refreshed walk-forward checkpoint fingerprint at step %d (resume-safe config change).",
        state.step_idx,
    )
    return state


def save_checkpoint(path: Path, state: WalkForwardCheckpoint) -> None:
    """Persist ``state`` to ``path`` (atomic replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    tmp.replace(path)
    logger.info(
        "Saved walk-forward checkpoint at step %d (cutoff=%d, model=%s).",
        state.step_idx,
        state.cutoff,
        state.model_version or "none",
    )


def load_checkpoint(path: Path) -> WalkForwardCheckpoint | None:
    """Load a checkpoint from ``path``, or ``None`` when missing or invalid."""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return WalkForwardCheckpoint(
            step_idx=int(raw["step_idx"]),
            cutoff=int(raw["cutoff"]),
            model_version=raw.get("model_version"),
            fingerprint=str(raw["fingerprint"]),
            symbol=str(raw["symbol"]),
            resolved_rows=int(raw["resolved_rows"]),
            updated_at=str(raw["updated_at"]),
            completed=bool(raw.get("completed", False)),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Ignoring corrupt walk-forward checkpoint %s: %s", path, exc)
        return None


def clear_checkpoint(path: Path) -> None:
    """Remove ``path`` when it exists."""
    if path.exists():
        path.unlink()
        logger.info("Cleared walk-forward checkpoint %s.", path)


def validate_checkpoint(
    state: WalkForwardCheckpoint,
    config: AppConfig,
    n_features: int,
    resolved_rows: int,
) -> None:
    """Raise ``ValueError`` when ``state`` cannot be resumed safely."""
    expected = checkpoint_fingerprint(config, n_features)
    if not checkpoint_fingerprint_matches(state, config, n_features):
        raise ValueError(
            "Walk-forward checkpoint does not match current config/features "
            f"(checkpoint={state.fingerprint}, current={expected}). "
            "Restore matching config or run `python -m epoch_ai checkpoint refresh`."
        )
    if state.fingerprint != expected:
        logger.warning(
            "Checkpoint fingerprint uses a legacy format (resume-safe); "
            "it will be refreshed on the next saved step."
        )
    if state.symbol != config.primary_symbol:
        raise ValueError(
            f"Checkpoint symbol {state.symbol!r} != {config.primary_symbol!r}. "
            "Run with --fresh to start over."
        )
    if resolved_rows < state.resolved_rows:
        raise ValueError(
            f"Resolved rows ({resolved_rows}) are fewer than at checkpoint "
            f"({state.resolved_rows}). Restore cached data or run with --fresh."
        )
    if state.cutoff < config.walk_forward.initial_train_period:
        raise ValueError(
            f"Checkpoint cutoff ({state.cutoff}) is before "
            f"initial_train_period ({config.walk_forward.initial_train_period})."
        )
    if state.cutoff > resolved_rows:
        raise ValueError(
            f"Checkpoint cutoff ({state.cutoff}) exceeds resolved rows ({resolved_rows}). "
            "Run with --fresh to start over."
        )


def build_checkpoint(
    *,
    step_idx: int,
    cutoff: int,
    model_version: str | None,
    config: AppConfig,
    n_features: int,
    resolved_rows: int,
    completed: bool = False,
) -> WalkForwardCheckpoint:
    """Construct a checkpoint for the next resume point."""
    return WalkForwardCheckpoint(
        step_idx=step_idx,
        cutoff=cutoff,
        model_version=model_version,
        fingerprint=checkpoint_fingerprint(config, n_features),
        symbol=config.primary_symbol,
        resolved_rows=resolved_rows,
        updated_at=datetime.now(UTC).isoformat(),
        completed=completed,
    )


def seed_checkpoint_from_last_step(
    config: AppConfig,
    last_completed_step: int,
    *,
    model_version: str | None = None,
    n_bars: int | None = None,
) -> WalkForwardCheckpoint:
    """Write a resume checkpoint after manually stopping a pre-checkpoint train run.

    Use the ``Step N | ...`` line from logs as ``last_completed_step`` (e.g. 75 for
    ``Step 75 | ...``). The next ``train`` run resumes at step ``N + 1``.
    """
    from epoch_ai.data.downloader import HistoricalDownloader
    from epoch_ai.features.pipeline import FeaturePipeline, build_target, forward_return

    cfg = config
    if cfg.model.backend == "evolved_nn" and cfg.data.use_synthetic_fallback:
        cfg = cfg.model_copy(deep=True)
        cfg.data.use_synthetic_fallback = False

    market = HistoricalDownloader(cfg).load_or_download(cfg.primary_symbol, n_bars=n_bars)
    features = FeaturePipeline(cfg).transform(market)
    y = build_target(market, cfg.prediction)
    fwd = forward_return(market, cfg.prediction.horizon)
    resolved_rows = len(features.join(y).join(fwd).dropna(subset=["target", "forward_return"]))

    wf = cfg.walk_forward
    step_idx = last_completed_step + 1
    cutoff = wf.initial_train_period + step_idx * wf.step_size
    version = model_version or f"v_{last_completed_step + 1}"
    state = build_checkpoint(
        step_idx=step_idx,
        cutoff=cutoff,
        model_version=version,
        config=cfg,
        n_features=len(features.columns),
        resolved_rows=resolved_rows,
    )
    path = resolve_checkpoint_path(cfg)
    save_checkpoint(path, state)
    return state
