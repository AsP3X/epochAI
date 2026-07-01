"""Full training loop: download → train → holdout → policy → optional run."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from epoch_ai.config.settings import AppConfig
from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.execution.policy.env import TradingReplayEnv
from epoch_ai.execution.policy.observation import observation_dim
from epoch_ai.execution.policy.ppo_policy import PPOPolicy
from epoch_ai.learning.acceptance import evaluate_holdout
from epoch_ai.learning.policy_promotion import (
    _load_multi_head_champion,
    train_challenger_policy,
)
from epoch_ai.services.runtime import RuntimeService
from epoch_ai.services.training import TrainingService
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class TrainCycleStepResult:
    """Outcome of one step inside a cycle."""

    name: str
    ok: bool
    detail: str = ""


@dataclass(slots=True)
class TrainCycleIterationResult:
    """Outcome of one full train-cycle iteration."""

    cycle: int
    steps: list[TrainCycleStepResult] = field(default_factory=list)
    ok: bool = True

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.steps.append(TrainCycleStepResult(name=name, ok=ok, detail=detail))
        if not ok:
            self.ok = False


@dataclass(slots=True)
class TrainCycleSummary:
    """Aggregate result after the loop exits."""

    cycles_completed: int
    iterations: list[TrainCycleIterationResult]
    stopped_reason: str
    elapsed_seconds: float


@dataclass(slots=True)
class TrainCycleOptions:
    """Knobs for :func:`run_train_cycle_loop`."""

    minutes: float = 10.0
    max_cycles: int | None = None
    interval_minutes: float = 0.0
    bars: int | None = 6000
    live_bars: int = 300
    train_bars: int | None = None
    train_max_steps: int | None = None
    full_history_download: bool = True
    skip_download: bool = False
    skip_run: bool = False
    embedding_policy: bool = False
    fresh_train: bool = False
    log_predictions: bool = False
    long_threshold: float | None = None
    short_threshold: float | None = None
    policy_updates: int | None = None
    policy_rollout_steps: int | None = None


def _apply_thresholds(config: AppConfig, options: TrainCycleOptions) -> AppConfig:
    cfg = config.model_copy(deep=True)
    if options.long_threshold is not None:
        cfg.risk.long_threshold = options.long_threshold
    if options.short_threshold is not None:
        cfg.risk.short_threshold = options.short_threshold
    return cfg


def _run_policy_training(
    config: AppConfig,
    *,
    bars: int | None,
    observation_mode: str | None = None,
    policy_updates: int | None = None,
    policy_rollout_steps: int | None = None,
) -> tuple[bool, str]:
    """Train PPO on OOS replay; returns (ok, detail)."""
    cfg = config.model_copy(deep=True)
    if observation_mode is not None:
        cfg.rl.observation_mode = observation_mode  # type: ignore[assignment]
    cfg.rl.enabled = True
    if policy_updates is not None:
        cfg.rl.total_updates = policy_updates
    if policy_rollout_steps is not None:
        cfg.rl.rollout_steps = policy_rollout_steps

    market = HistoricalDownloader(cfg).load_or_download(cfg.primary_symbol, n_bars=bars)
    start = cfg.walk_forward.initial_train_period
    if len(market) <= start + 10:
        return False, f"need > {start + 10} bars for policy training; got {len(market)}"

    oos = market.iloc[start:]
    champion = _load_multi_head_champion(cfg)
    if champion is not None:
        policy, _work, stats = train_challenger_policy(cfg, oos, champion)
    else:
        env = TradingReplayEnv.from_market(cfg, oos)
        policy = PPOPolicy(observation_dim(cfg), cfg.rl)
        stats = policy.train(env)
    policy.save(cfg.rl.policy_path)
    mode = cfg.rl.observation_mode
    return True, (
        f"mode={mode} updates={stats.updates} equity={stats.final_equity:,.2f} "
        f"saved={cfg.rl.policy_path}"
    )


def run_single_train_cycle(
    config: AppConfig,
    options: TrainCycleOptions,
    *,
    cycle: int,
) -> TrainCycleIterationResult:
    """Run one iteration: download → train → holdout → policy → [embedding] → run."""
    result = TrainCycleIterationResult(cycle=cycle)
    cfg = _apply_thresholds(config, options)
    train_service = TrainingService(cfg)

    if not options.skip_download:
        try:
            downloader = HistoricalDownloader(cfg)
            df = downloader.load_or_download(
                cfg.primary_symbol,
                n_bars=None if options.full_history_download else options.bars,
                full_history=options.full_history_download,
            )
            result.add("download", True, f"{len(df):,} bars")
        except Exception as exc:  # noqa: BLE001
            logger.exception("train-cycle download failed")
            result.add("download", False, str(exc))
            return result
    else:
        result.add("download", True, "skipped")

    try:
        train_result = train_service.train(
            n_bars=options.train_bars,
            max_steps=options.train_max_steps,
            log_predictions=options.log_predictions,
            resume=not options.fresh_train,
            fresh=options.fresh_train,
            full_history=options.full_history_download and not options.skip_download,
        )
        result.add(
            "train",
            True,
            f"version={train_result.model_version or 'n/a'} "
            f"steps={train_result.walk_forward_steps}",
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("train-cycle train failed")
        result.add("train", False, str(exc))
        return result

    try:
        report = evaluate_holdout(cfg, n_bars=options.bars)
        if report.skipped:
            result.add("evaluate-holdout", False, report.reason)
            return result
        pred_brier = report.predictor_metrics.get("oos_brier")
        pol_rar = report.policy_champion.get("risk_adjusted_return")
        detail = f"holdout_bars={report.holdout_bars}"
        if pred_brier is not None:
            detail += f" oos_brier={pred_brier:.6f}"
        if pol_rar is not None:
            detail += f" policy_rar={pol_rar:.6f}"
        result.add("evaluate-holdout", True, detail)
    except Exception as exc:  # noqa: BLE001
        logger.exception("train-cycle holdout failed")
        result.add("evaluate-holdout", False, str(exc))
        return result

    ok, detail = _run_policy_training(
        cfg,
        bars=options.bars,
        observation_mode=None,
        policy_updates=options.policy_updates,
        policy_rollout_steps=options.policy_rollout_steps,
    )
    result.add("train-policy", ok, detail)
    if not ok:
        return result

    if options.embedding_policy:
        ok, detail = _run_policy_training(
            cfg,
            bars=options.bars,
            observation_mode="embedding",
            policy_updates=options.policy_updates,
            policy_rollout_steps=options.policy_rollout_steps,
        )
        result.add("train-policy-embedding", ok, detail)
        if not ok:
            return result

    if options.skip_run:
        result.add("run", True, "skipped")
        return result

    try:
        runtime = RuntimeService(cfg)
        if runtime.status().models_available == 0:
            result.add("run", False, "no models in registry")
            return result
        session = runtime.run_session(
            mode="replay",
            n_bars=options.bars,
            live_bars=options.live_bars,
            log_predictions=options.log_predictions,
        )
        result.add(
            "run",
            True,
            f"bars={session.bars_processed} fills={session.fills} "
            f"equity={session.final_equity:,.2f}",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("train-cycle run failed")
        result.add("run", False, str(exc))
    return result


def run_train_cycle_loop(
    config: AppConfig,
    options: TrainCycleOptions,
) -> TrainCycleSummary:
    """Repeat :func:`run_single_train_cycle` until time or cycle limit is reached."""
    t0 = time.monotonic()
    deadline: float | None = None
    if options.minutes > 0:
        deadline = t0 + options.minutes * 60.0

    iterations: list[TrainCycleIterationResult] = []
    cycle = 0
    stopped_reason = "completed"

    while True:
        cycle += 1
        if options.max_cycles is not None and cycle > options.max_cycles:
            stopped_reason = f"max_cycles={options.max_cycles}"
            break
        if deadline is not None and time.monotonic() >= deadline:
            stopped_reason = f"minutes={options.minutes}"
            break

        logger.info("train-cycle starting iteration %d", cycle)
        iteration = run_single_train_cycle(config, options, cycle=cycle)
        iterations.append(iteration)

        if not iteration.ok:
            stopped_reason = f"cycle {cycle} failed at step {iteration.steps[-1].name}"
            break

        if options.max_cycles is not None and cycle >= options.max_cycles:
            stopped_reason = f"max_cycles={options.max_cycles}"
            break
        if deadline is not None and time.monotonic() >= deadline:
            stopped_reason = f"minutes={options.minutes}"
            break
        if options.interval_minutes > 0:
            sleep_s = options.interval_minutes * 60.0
            if deadline is not None:
                sleep_s = min(sleep_s, max(0.0, deadline - time.monotonic()))
            if sleep_s > 0:
                logger.info("train-cycle sleeping %.1f min before next iteration", sleep_s / 60)
                time.sleep(sleep_s)

    elapsed = time.monotonic() - t0
    return TrainCycleSummary(
        cycles_completed=len(iterations),
        iterations=iterations,
        stopped_reason=stopped_reason,
        elapsed_seconds=elapsed,
    )
