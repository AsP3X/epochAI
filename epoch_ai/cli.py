"""Command-line orchestration for epoch_ai.

Sub-commands:

* ``train``        - train the AI (progressive walk-forward + model registry).
* ``run``          - run a trained model (paper/replay session from registry).
* ``download``     - fetch the longest possible history (or synthesize it offline).
* ``backtest``     - run the progressive historical-learning backtest.
* ``paper-trade``  - simulate near-real-time paper trading with periodic updates.
* ``live``         - WebSocket stream or historical replay live loop.
* ``retrain``      - periodic retrain from SQLite logs or parquet history.
* ``auto-retrain`` - retrain a challenger and promote it only if it beats the champion.
* ``tune``         - run a YAML sweep over config overrides.
* ``promote``      - promote the best tune experiment to a config file.
* ``export``       - export open-weights bundle + model card.
* ``serve``        - start the FastAPI HTTP API.
* ``telegram``     - start the optional Telegram bot.
* ``kill-switch``  - halt or resume live trading globally.
* ``schedule-retrain`` - periodic retrain loop.
* ``checkpoint``     - seed or inspect walk-forward resume state.
* ``progress``       - show walk-forward position and steps remaining.
* ``info``         - print the resolved configuration.
* ``predict``      - multi-horizon forecast table or JSON for the latest bar.
* ``train-policy`` - train PPO trading policy on out-of-sample bar replay.
* ``evaluate-holdout`` - score predictor + policy on the untouched final holdout.

Run ``python -m epoch_ai <command> --help`` for details.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from epoch_ai.backtesting.engine import Backtester
from epoch_ai.backtesting.reporting import format_importance_value, importance_metric_label
from epoch_ai.config.overrides import apply_overrides, parse_set_args
from epoch_ai.config.settings import AppConfig
from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.execution.live_loop import run_bar_loop, run_scheduled_retrain
from epoch_ai.features.pipeline import FeaturePipeline
from epoch_ai.learning.checkpoint import seed_checkpoint_from_last_step
from epoch_ai.learning.degradation import degradation_hints
from epoch_ai.logging_system.joiner import RetrainLogStats, retrain_log_stats
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.services.runtime import RuntimeService
from epoch_ai.services.training import TrainingService
from epoch_ai.tracking.mlflow_tracker import MLflowTracker
from epoch_ai.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _load_retrain_log_stats(config: AppConfig) -> RetrainLogStats:
    """Read joined-sample counts from the SQLite prediction store."""
    store = PredictionStore(config.logging.db_path)
    try:
        return retrain_log_stats(store, config.primary_symbol)
    finally:
        store.close()


def _print_retrain_log_summary(
    config: AppConfig,
    stats: RetrainLogStats,
    *,
    before: RetrainLogStats | None = None,
) -> None:
    """Show how many rows are eligible for ``retrain --min-new-samples``."""
    print("\n--- Retrain dataset (SQLite) ---")
    if before is not None and stats.joined_samples > before.joined_samples:
        added = stats.joined_samples - before.joined_samples
        print(f"  This session       : +{added:,} joined sample(s)")
    print(f"  Joined samples     : {stats.joined_samples:,}  (max --min-new-samples)")
    print(f"  Predictions logged : {stats.predictions:,}")
    print(f"  Pending outcomes   : {stats.pending:,}  (horizon not elapsed yet)")
    if stats.joined_samples > 0:
        print(
            f"  Retrain example    : python -m epoch_ai retrain "
            f"--min-new-samples {stats.joined_samples}"
        )
    else:
        print(
            "  Retrain example    : python -m epoch_ai retrain --min-new-samples 50 "
            "(falls back to parquet until enough joined samples exist)"
        )
    print(f"  Store path         : {config.logging.db_path}")


def _load(args: argparse.Namespace) -> AppConfig:
    path = Path(args.config)
    raw: dict = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    elif args.config != "config/config.yaml":
        raise FileNotFoundError(f"Config file not found: {path}")

    overrides = parse_set_args(getattr(args, "set", []) or [])
    if overrides:
        raw = apply_overrides(raw, overrides)

    config = AppConfig.model_validate(raw)
    if getattr(args, "symbol", None):
        config.symbols = [args.symbol]
    if getattr(args, "max_steps", None) is not None:
        config.walk_forward.max_steps = args.max_steps
    return config


def _print_train_interrupted(config: AppConfig) -> None:
    """User-facing summary when ``train`` is stopped with Ctrl+C."""
    from epoch_ai.learning.checkpoint import load_checkpoint, resolve_checkpoint_path

    print("\n=== Training interrupted ===")
    wf = config.walk_forward
    if not wf.checkpoint_enabled:
        print("Checkpoints are disabled; only fully completed steps are persisted elsewhere.")
        return

    path = resolve_checkpoint_path(config)
    state = load_checkpoint(path)
    if state is None or state.completed:
        print("No resume checkpoint on disk.")
        print("If you stopped mid-step, seed one from the last log line:")
        print("  python -m epoch_ai checkpoint seed --last-step <N>")
        return

    print(f"Progress saved at step {state.step_idx} (cutoff={state.cutoff}).")
    if state.model_version:
        print(f"Model checkpoint     : {state.model_version}")
    print(f"Checkpoint file        : {path}")
    print("\nResume with:")
    print("  python -m epoch_ai train --log-predictions --set model.device=cuda")


# --------------------------------------------------------------------- commands
def cmd_train(args: argparse.Namespace) -> int:
    """Train the AI via progressive walk-forward learning and register the model."""
    config = _load(args)
    service = TrainingService(config)
    try:
        result = service.train(
            n_bars=args.bars,
            max_steps=args.max_steps,
            log_predictions=args.log_predictions,
            register=not args.no_register,
            resume=not args.no_resume,
            fresh=args.fresh,
            full_history=args.full_history,
            refresh_data=args.refresh_data,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        _print_train_interrupted(config)
        return 130
    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    print("\n=== Training complete ===")
    print(f"Symbol            : {config.primary_symbol}")
    print(f"Model version     : {result.model_version or '(not registered)'}")
    print(f"Walk-forward steps: {result.walk_forward_steps}")
    if result.resumed_from_step is not None:
        print(f"Resumed from step : {result.resumed_from_step}")
    print(f"Final train rows  : {result.train_rows:,}")
    if not result.feature_importance.empty:
        metric = importance_metric_label(config.model.backend)
        print(f"Top features ({metric}):")
        for name, score in result.feature_importance.head(5).items():
            print(f"  {name:<28}{format_importance_value(float(score)):>14}")
    if args.log_predictions:
        _print_retrain_log_summary(config, _load_retrain_log_stats(config))
    return 0


def cmd_checkpoint_seed(args: argparse.Namespace) -> int:
    """Seed a walk-forward checkpoint from the last completed log step."""
    from epoch_ai.learning.checkpoint import resolve_checkpoint_path

    config = _load(args)
    state = seed_checkpoint_from_last_step(
        config,
        args.last_step,
        model_version=args.model_version,
        n_bars=args.bars,
    )
    path = resolve_checkpoint_path(config)
    print(f"Checkpoint written: {path}")
    print(f"  last completed step : {args.last_step}")
    print(f"  resume at step      : {state.step_idx}")
    print(f"  cutoff              : {state.cutoff}")
    print(f"  model_version       : {state.model_version}")
    print(f"  resolved_rows       : {state.resolved_rows}")
    print(f"  fingerprint         : {state.fingerprint}")
    print("\nResume with:")
    print("  python -m epoch_ai train --log-predictions --set model.device=cuda")
    return 0


def cmd_checkpoint_refresh(args: argparse.Namespace) -> int:
    """Refresh checkpoint fingerprint after resume-safe config changes."""
    from epoch_ai.learning.checkpoint import (
        load_checkpoint,
        refresh_checkpoint_fingerprint,
        resolve_checkpoint_path,
    )
    from epoch_ai.learning.progress_report import count_resolved_rows

    config = _load(args)
    path = resolve_checkpoint_path(config)
    state = load_checkpoint(path)
    if state is None:
        logger.error("No checkpoint at %s", path)
        return 1
    _, n_features = count_resolved_rows(config, n_bars=args.bars)
    refreshed = refresh_checkpoint_fingerprint(path, config, n_features)
    if refreshed is None:
        logger.error(
            "Checkpoint at %s is not compatible with current config/features.", path
        )
        return 1
    print(f"Checkpoint refreshed: {path}")
    print(f"  step          : {refreshed.step_idx}")
    print(f"  cutoff        : {refreshed.cutoff}")
    print(f"  model_version : {refreshed.model_version}")
    print(f"  fingerprint   : {refreshed.fingerprint}")
    return 0


def cmd_progress(args: argparse.Namespace) -> int:
    """Print walk-forward training position without running inference."""
    from epoch_ai.learning.progress_report import (
        format_training_progress,
        gather_training_progress,
        watch_training_progress,
    )

    config = _load(args)
    if args.watch:
        return watch_training_progress(
            config,
            interval=args.interval,
            n_bars=args.bars,
            refresh_rows=args.refresh_rows,
        )
    report = gather_training_progress(
        config,
        n_bars=args.bars,
        refresh_rows=args.refresh_rows,
    )
    print(format_training_progress(report))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run a trained model from the registry (paper/replay or live feed)."""
    config = _load(args)
    if args.long_threshold is not None:
        config.risk.long_threshold = args.long_threshold
    if args.short_threshold is not None:
        config.risk.short_threshold = args.short_threshold
    if args.policy is not None:
        config.trading.policy_backend = args.policy
    if args.reserve_fraction is not None:
        config.execution.reserve_fraction = args.reserve_fraction
    if args.confirm_live:
        config.execution.mode = "live"
        config.execution.live_enabled = True
        config.execution.dry_run = False

    runtime = RuntimeService(config)
    status = runtime.status()
    if status.models_available == 0:
        logger.error("No trained models in registry. Run `python -m epoch_ai train` first.")
        return 1

    stats_before = _load_retrain_log_stats(config) if args.log_predictions else None

    if args.live_stream:
        import asyncio

        try:
            result = asyncio.run(
                runtime.run_live_stream(
                    model_version=args.model_version,
                    log_predictions=args.log_predictions,
                )
            )
        except RuntimeError as exc:
            logger.error("%s — use --live-feed for offline simulation.", exc)
            return 1
    elif args.live_feed:
        result = runtime.run_live_feed(
            n_bars=args.bars,
            feed_bars=args.live_bars,
            model_version=args.model_version,
            log_predictions=args.log_predictions,
        )
    else:
        result = None

    if args.live_stream or args.live_feed:
        print("\n=== Live session complete ===")
        print(f"Model version     : {result.model_version}")
        print(f"Live ticks        : {result.ticks}")
        print(f"Trades (fills)    : {result.fills}")
        print(f"Final equity      : {result.final_equity:,.2f}")
        print(f"Trading capital   : {result.treasury.trading_capital:,.2f}")
        print(f"Reserved wins     : {result.treasury.reserved_wins:,.2f}")
        print(f"Session PnL       : {result.treasury.last_session_pnl:,.2f}")
        if result.treasury.last_reserved > 0:
            print(f"Set aside (wins)  : {result.treasury.last_reserved:,.2f}")
            print(f"Reinvested        : {result.treasury.last_reinvested:,.2f}")
        stats_after = _load_retrain_log_stats(config)
        if args.log_predictions or stats_after.joined_samples > 0:
            _print_retrain_log_summary(config, stats_after, before=stats_before)
        return 0

    session = runtime.run_session(
        mode="replay" if args.replay else "paper",
        n_bars=args.bars,
        live_bars=args.live_bars,
        retrain_every=args.retrain_every,
        model_version=args.model_version,
        log_predictions=args.log_predictions,
    )
    print("\n=== Runtime session complete ===")
    print(f"Model version     : {runtime.status().model_version}")
    print(f"Bars processed    : {session.bars_processed}")
    print(f"Trades (fills)    : {session.fills}")
    print(f"Final equity      : {session.final_equity:,.2f}")
    stats_after = _load_retrain_log_stats(config)
    if args.log_predictions or stats_after.joined_samples > 0:
        _print_retrain_log_summary(config, stats_after, before=stats_before)
    return 0


def cmd_train_policy(args: argparse.Namespace) -> int:
    """Train the PPO trading policy on out-of-sample historical replay."""
    config = _load(args)
    if args.updates is not None:
        config.rl.total_updates = args.updates
    if args.rollout_steps is not None:
        config.rl.rollout_steps = args.rollout_steps
    if getattr(args, "observation_mode", None) is not None:
        config.rl.observation_mode = args.observation_mode
    if getattr(args, "trunk_frozen", None) is not None:
        config.rl.trunk_frozen = args.trunk_frozen
    if getattr(args, "policy_loss_weight", None) is not None:
        config.rl.policy_loss_weight = args.policy_loss_weight
    config.rl.enabled = True

    from epoch_ai.data.downloader import HistoricalDownloader
    from epoch_ai.execution.policy.env import TradingReplayEnv
    from epoch_ai.execution.policy.observation import observation_dim
    from epoch_ai.execution.policy.ppo_policy import PPOPolicy
    from epoch_ai.learning.policy_promotion import (
        _load_multi_head_champion,
        train_challenger_policy,
    )

    market = HistoricalDownloader(config).load_or_download(
        config.primary_symbol,
        n_bars=args.bars,
    )
    start = config.walk_forward.initial_train_period
    if len(market) <= start + 10:
        logger.error(
            "Need more than %d bars for OOS policy training; got %d.",
            start + 10,
            len(market),
        )
        return 1

    oos = market.iloc[start:]
    champion_model = _load_multi_head_champion(config)
    if champion_model is not None:
        logger.info(
            "Policy training on champion (%s observation).",
            config.rl.observation_mode,
        )
    else:
        logger.warning(
            "No multi-head champion model found; training policy on price-only proxy env."
        )

    if champion_model is None:
        env = TradingReplayEnv.from_market(config, oos)
        policy = PPOPolicy(observation_dim(config), config.rl)
        stats = policy.train(env)
    else:
        policy, _work_model, stats = train_challenger_policy(config, oos, champion_model)

    policy.save(config.rl.policy_path)

    print("\n=== Policy training complete ===")
    print(f"OOS bars replayed : {len(oos):,}")
    print(f"Observation mode  : {config.rl.observation_mode}")
    print(f"PPO updates         : {stats.updates}")
    print(f"Mean rollout reward : {stats.mean_reward:.6f}")
    print(f"Final replay equity : {stats.final_equity:,.2f}")
    print(f"Saved policy        : {config.rl.policy_path}")
    return 0


def cmd_evaluate_holdout(args: argparse.Namespace) -> int:
    """Evaluate predictor and policy benchmarks on the final holdout slice."""
    from epoch_ai.learning.acceptance import evaluate_holdout

    config = _load(args)
    report = evaluate_holdout(config, n_bars=args.bars)
    if report.skipped:
        print(f"Holdout evaluation skipped: {report.reason}")
        return 1
    print("\n=== Holdout acceptance report ===")
    print(f"Holdout bars     : {report.holdout_bars:,}")
    if report.predictor_metrics:
        print("Predictor metrics:")
        for key, value in sorted(report.predictor_metrics.items()):
            if isinstance(value, float):
                print(f"  {key:<28}{value:.6f}")
    print("Policy baseline  :", report.policy_baseline)
    print("Policy buy&hold  :", report.policy_buy_hold)
    if report.policy_champion:
        print("Policy champion  :", report.policy_champion)
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Download or synthesize historical data and cache it as parquet."""
    config = _load(args)
    if args.full_history and args.bars is not None:
        logger.error("Use either --full-history or --bars, not both.")
        return 1
    n_bars = None if args.full_history else args.bars
    downloader = HistoricalDownloader(config)
    df = downloader.load_or_download(
        config.primary_symbol,
        n_bars=n_bars,
        force=args.force,
        full_history=args.full_history,
    )
    logger.info(
        "Data ready: %s | %d bars | %s -> %s",
        config.primary_symbol,
        len(df),
        df.index[0],
        df.index[-1],
    )
    print(df.tail())
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run the full progressive historical-learning backtest."""
    config = _load(args)
    if args.long_threshold is not None:
        config.risk.long_threshold = args.long_threshold
    if args.short_threshold is not None:
        config.risk.short_threshold = args.short_threshold
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloader = HistoricalDownloader(config)
    market = downloader.load_or_download(config.primary_symbol, n_bars=args.bars)
    features = FeaturePipeline(config).transform(market)

    store = PredictionStore(config.logging.db_path) if args.log_predictions else None
    tracker = MLflowTracker(config.tracking)

    backtester = Backtester(config)
    with tracker:
        result = backtester.run(
            market, features=features, store=store, register_models=args.register_models
        )
        tracker.log_params(
            {
                "symbol": config.primary_symbol,
                "timeframe": config.timeframe,
                "horizon": config.prediction.horizon,
                "initial_train_period": config.walk_forward.initial_train_period,
                "step_size": config.walk_forward.step_size,
            }
        )
        tracker.log_metrics(result.metrics)
        tracker.log_learning_metrics(
            result.learning.step_history,
            result.learning_improvement,
            result.learning_curve,
        )

    metrics_payload = {
        "strategy": result.metrics,
        "benchmark": result.benchmark_metrics,
        "learning_improvement": result.learning_improvement,
        "learning_curve": result.learning_curve,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    result.equity_curve.rename("equity").to_csv(out_dir / "equity_curve.csv")
    result.learning.step_history.to_csv(out_dir / "step_history.csv", index=False)
    result.learning.feature_importance.rename("gain").to_csv(out_dir / "feature_importance.csv")
    curve_path = out_dir / "learning_curve.json"
    curve_path.write_text(json.dumps(result.learning_curve, indent=2))

    if tracker.active:
        tracker.log_artifact(metrics_path)
        tracker.log_artifact(curve_path)
        tracker.log_artifact(out_dir / "step_history.csv")

    _print_report(config, result, store)
    if store is not None:
        store.close()
    return 0


def cmd_paper_trade(args: argparse.Namespace) -> int:
    """Simulate near-real-time paper trading over the most recent bars."""
    config = _load(args)
    if args.long_threshold is not None:
        config.risk.long_threshold = args.long_threshold
    if args.short_threshold is not None:
        config.risk.short_threshold = args.short_threshold

    downloader = HistoricalDownloader(config)
    market = downloader.load_or_download(config.primary_symbol, n_bars=args.bars)

    from epoch_ai.features.pipeline import build_target, forward_return

    features = FeaturePipeline(config).transform(market)
    y = build_target(market, config.prediction)
    fwd = forward_return(market, config.prediction.horizon)
    data = features.join(y).join(fwd).dropna(subset=["target", "forward_return"])

    live_bars = min(args.live_bars, len(data) - config.walk_forward.initial_train_period)
    if live_bars < 1:
        logger.error("Not enough data for paper trading. Increase --bars.")
        return 1

    split = len(data) - live_bars
    result = run_bar_loop(
        config,
        market,
        start_pos=split,
        retrain_every=args.retrain_every,
    )

    print("\n=== Paper-trading summary ===")
    print(f"Symbol            : {config.primary_symbol}")
    print(f"Bars simulated    : {result.bars_processed}")
    print(f"Trades (fills)    : {result.fills}")
    print(f"Inline retrains   : {result.retrain_count}")
    print(f"Starting capital  : {config.risk.initial_capital:,.2f}")
    print(f"Final equity      : {result.final_equity:,.2f}")
    print(
        f"Return            : {(result.final_equity / config.risk.initial_capital - 1) * 100:,.2f}%"
    )
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    """Run a live WebSocket loop or historical replay fallback."""
    config = _load(args)
    if args.replay:
        downloader = HistoricalDownloader(config)
        market = downloader.load_or_download(config.primary_symbol, n_bars=args.bars)
        features = FeaturePipeline(config).transform(market)
        from epoch_ai.features.pipeline import build_target, forward_return

        y = build_target(market, config.prediction)
        fwd = forward_return(market, config.prediction.horizon)
        data = features.join(y).join(fwd).dropna(subset=["target", "forward_return"])
        live_bars = min(args.live_bars, len(data) - config.walk_forward.initial_train_period)
        if live_bars < 1:
            logger.error("Not enough data for live replay. Increase --bars.")
            return 1
        split = len(data) - live_bars
        result = run_bar_loop(
            config,
            market,
            start_pos=split,
            retrain_every=args.retrain_every,
        )
        print(f"Replay complete: {result.bars_processed} bars, equity={result.final_equity:,.2f}")
        return 0

    import asyncio

    from epoch_ai.data.websocket import RealtimeDataHandler

    handler = RealtimeDataHandler(config)

    def on_candle(symbol: str, frame) -> None:
        logger.info("New candle %s | buffer=%d rows", symbol, len(frame))

    try:
        asyncio.run(handler.stream(on_candle=on_candle))
    except RuntimeError as exc:
        logger.error("%s — use --replay for offline simulation.", exc)
        return 1
    return 0


def cmd_retrain(args: argparse.Namespace) -> int:
    """Retrain from SQLite logs or cached historical data."""
    config = _load(args)
    stats = _load_retrain_log_stats(config)
    _print_retrain_log_summary(config, stats)
    if stats.joined_samples < args.min_new_samples:
        print(
            f"\nNote: {stats.joined_samples:,} joined sample(s) < "
            f"--min-new-samples {args.min_new_samples}; "
            "retrain will use cached parquet history instead."
        )
    code = run_scheduled_retrain(config, min_new_samples=args.min_new_samples)
    return code


def _print_auto_retrain(result) -> None:
    """Pretty-print a single auto-retrain cycle result."""
    print("=" * 64)
    print("  AUTO-RETRAIN (challenger vs champion)")
    print("-" * 64)
    print(f"  Challenger     : {result.challenger_label}")
    print(f"  Champion       : {result.champion_label or '(none)'}")
    print(f"  Metric         : {result.metric}")
    print(f"  Challenger val : {result.challenger_value:.6f}")
    print(f"  Champion val   : {result.champion_value:.6f}")
    print(f"  Train / eval   : {result.train_rows} / {result.eval_rows} rows")
    print(f"  Decision       : {'PROMOTED' if result.promoted else 'kept champion'}")
    print(f"  Reason         : {result.reason}")
    print("=" * 64)


def cmd_auto_retrain(args: argparse.Namespace) -> int:
    """Retrain a challenger and promote it only if it beats the champion on a holdout.

    With ``--minutes`` the cycle repeats back-to-back (or every ``--interval-minutes``)
    until the wall-clock budget elapses, so a long GPU run is a single command.
    """
    config = _load(args)
    service = TrainingService(config)

    if args.minutes is None:
        result = service.auto_retrain(n_bars=args.bars)
        if result.skipped:
            print(f"Auto-retrain skipped: {result.reason}")
            return 1
        _print_auto_retrain(result)
        if args.promote_policy and config.rl.enabled:
            from epoch_ai.learning.policy_promotion import auto_train_and_promote_policy

            policy_result = auto_train_and_promote_policy(config, n_bars=args.bars)
            print(
                f"Policy: {'PROMOTED' if policy_result.promoted else 'kept champion'} "
                f"({policy_result.reason})"
            )
        return 0

    import time

    deadline = time.monotonic() + args.minutes * 60.0
    cycle = 0
    promotions = 0
    while True:
        cycle += 1
        result = service.auto_retrain(n_bars=args.bars)
        promotions += int(result.promoted)
        status = (
            "skipped"
            if result.skipped
            else ("PROMOTED" if result.promoted else "kept champion")
        )
        print(
            f"[cycle {cycle}] {status}: challenger={result.challenger_label} "
            f"{result.metric}={result.challenger_value:.6f}"
        )
        if time.monotonic() >= deadline:
            break
        if args.interval_minutes > 0:
            time.sleep(min(args.interval_minutes * 60.0, max(0.0, deadline - time.monotonic())))
    print(f"Done: {cycle} cycle(s), {promotions} promotion(s) over ~{args.minutes:g} min.")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    """Run a YAML sweep of config overrides and write metrics per experiment."""
    config = _load(args)
    sweep_path = Path(args.sweep)
    if not sweep_path.exists():
        logger.error("Sweep file not found: %s", sweep_path)
        return 1

    sweep = yaml.safe_load(sweep_path.read_text(encoding="utf-8")) or {}
    experiments = sweep.get("experiments", [])
    if not experiments:
        logger.error("Sweep file has no experiments.")
        return 1

    base_raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    downloader = HistoricalDownloader(config)
    market = downloader.load_or_download(config.primary_symbol, n_bars=args.bars)

    for exp in experiments:
        name = exp.get("name", "unnamed")
        overrides = exp.get("overrides", {})
        merged = apply_overrides(base_raw, overrides)
        exp_config = AppConfig.model_validate(merged)
        if getattr(args, "symbol", None):
            exp_config.symbols = [args.symbol]
        if args.max_steps is not None:
            exp_config.walk_forward.max_steps = args.max_steps

        features = FeaturePipeline(exp_config).transform(market)
        result = Backtester(exp_config).run(market, features=features)
        exp_dir = out_root / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "overrides": overrides,
            "strategy": result.metrics,
            "learning_improvement": result.learning_improvement,
            "learning_curve": result.learning_curve,
        }
        (exp_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
        logger.info("Sweep %s Sharpe=%.3f", name, result.metrics.get("sharpe", 0.0))

    print(f"Wrote {len(experiments)} experiment(s) to {out_root}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Print the resolved configuration as JSON."""
    config = _load(args)
    print(config.model_dump_json(indent=2))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Emit a multi-horizon forecast for the latest bar."""
    from epoch_ai.services.forecast_api import build_live_payload
    from epoch_ai.services.runtime import RuntimeService

    config = _load(args)
    market = HistoricalDownloader(config).load_or_download(
        config.primary_symbol,
        n_bars=args.bars,
    )
    runtime = RuntimeService(config)
    runtime.load_model(args.model_version)
    result = runtime.predict_multi_horizon(market)
    payload = build_live_payload(result)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"as_of={result.as_of} close={result.last_close:.2f} model={result.model_version}")
        for h in result.horizons:
            flag = "ok" if h.reliable else "low"
            print(
                f"  {h.label:>4}  p_up={h.p_up:.3f}  q50_ret={h.exp_return:+.5f}  "
                f"p50={h.price_p50:.2f}  conf={h.confidence:.2f}  [{flag}]"
            )
        baseline = payload["baseline"]
        print(
            f"baseline signal={baseline['signal']:+d}  "
            f"weighted_p_up={baseline['weighted_p_up']:.3f}  "
            f"heads={baseline['n_heads_used']}"
        )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export an open-weights bundle with a model card."""
    from epoch_ai.export.model_card import export_bundle_with_card

    config = _load(args)
    path = export_bundle_with_card(
        config,
        dest=args.dest,
        label=args.model_version,
    )
    print(f"Exported bundle: {path}")
    print(f"Model card: {path / 'MODEL_CARD.md'}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the FastAPI HTTP server."""
    config = _load(args)
    try:
        import uvicorn
    except ImportError as exc:
        logger.error("uvicorn required: pip install -r requirements-optional.txt")
        raise SystemExit(1) from exc

    from epoch_ai.api.app import create_app

    app = create_app(config)
    host = args.host or config.api.host
    port = args.port or config.api.port
    logger.info("Serving epochAI API at http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    """Promote the best tune sweep experiment to a config file."""
    from epoch_ai.tuning.promote import promote_best

    result = promote_best(
        args.config,
        args.sweep_out,
        metric=args.metric,
        dest=args.out,
    )
    print("\n=== Promote complete ===")
    print(f"Experiment        : {result.experiment}")
    print(f"{result.metric:<18}: {result.metric_value:.4f}")
    if result.promoted_config_path:
        print(f"Config written    : {result.promoted_config_path}")
    return 0


def cmd_kill_switch(args: argparse.Namespace) -> int:
    """Halt or resume live trading via the global kill switch."""
    from epoch_ai.execution.kill_switch import KillSwitch

    config = _load(args)
    ks = KillSwitch(config.execution.kill_switch_path)
    if args.action == "halt":
        state = ks.halt(args.reason)
        print(f"HALTED: {state.reason}")
    elif args.action == "resume":
        state = ks.resume()
        print(f"RESUMED at {state.updated_at}")
    else:
        state = ks.read()
        print(json.dumps({"halted": state.halted, "reason": state.reason, "updated_at": state.updated_at}))
    return 0


def cmd_telegram(args: argparse.Namespace) -> int:
    """Start the optional Telegram bot."""
    from epoch_ai.bots.telegram_bot import run_telegram_bot

    config = _load(args)
    run_telegram_bot(config)
    return 0


def cmd_schedule_retrain(args: argparse.Namespace) -> int:
    """Run periodic retraining on a fixed interval."""
    from epoch_ai.learning.scheduler import run_retrain_scheduler

    config = _load(args)
    results = run_retrain_scheduler(
        config,
        interval_hours=args.interval_hours,
        min_new_samples=args.min_new_samples,
        max_cycles=args.max_cycles,
        promote=args.promote,
        promote_policy=args.promote_policy,
    )
    print(f"Completed {len(results)} retrain cycle(s).")
    for idx, result in enumerate(results, start=1):
        if hasattr(result, "challenger_label"):
            print(
                f"  cycle {idx}: challenger={result.challenger_label} "
                f"promoted={result.promoted} skipped={result.skipped} ({result.reason})"
            )
        elif hasattr(result, "champion_path"):
            print(
                f"  cycle {idx}: policy promoted={result.promoted} "
                f"metric={result.metric}={result.challenger_value:.6f} ({result.reason})"
            )
        else:
            print(
                f"  cycle {idx}: version={result.model_version} rows={result.train_rows} "
                f"skipped={result.skipped}"
            )
    return 0


# ------------------------------------------------------------------- reporting
def _print_report(config: AppConfig, result, store: PredictionStore | None) -> None:
    m = result.metrics
    b = result.benchmark_metrics
    imp = result.learning_improvement
    curve = result.learning_curve
    n_rebalances = int(m.get("n_rebalances", 0))
    print("\n" + "=" * 64)
    print(f"  PROGRESSIVE BACKTEST REPORT - {config.primary_symbol} {config.timeframe}")
    print("=" * 64)
    print(f"  Predictions made   : {len(result.learning.predictions):,}")
    print(f"  Walk-forward steps : {len(result.learning.step_history):,}")
    print(f"  Position rebalances: {n_rebalances:,}")
    print("-" * 64)
    print(f"  {'Metric':<20}{'Strategy':>18}{'Buy & Hold':>18}")
    print("-" * 64)
    rows = [
        ("Total return", "total_return", "%"),
        ("CAGR", "cagr", "%"),
        ("Sharpe", "sharpe", ""),
        ("Sortino", "sortino", ""),
        ("Calmar", "calmar", ""),
        ("Max drawdown", "max_drawdown", "%"),
        ("Profit factor", "profit_factor", ""),
        ("Win rate", "win_rate", "%"),
    ]
    for label, key, unit in rows:
        sv = m[key] * (100 if unit == "%" else 1)
        bv = b[key] * (100 if unit == "%" else 1)
        print(f"  {label:<20}{sv:>17.3f}{unit}{bv:>17.3f}{unit}")
    print("-" * 64)
    if imp:
        print("  Learning curve (out-of-sample, walk-forward steps):")
        print(
            f"    accuracy   first: {imp.get('first_half_accuracy', 0):.3f}"
            f"  second: {imp.get('second_half_accuracy', 0):.3f}"
            f"  delta: {imp.get('delta', 0):+.3f}"
        )
        if "first_half_logloss" in imp:
            print(
                f"    logloss    first: {imp['first_half_logloss']:.3f}"
                f"  second: {imp['second_half_logloss']:.3f}"
                f"  delta: {imp['logloss_delta']:+.3f}"
            )
        if "first_half_dir_accuracy" in imp:
            print(
                f"    dir_acc    first: {imp['first_half_dir_accuracy']:.3f}"
                f"  second: {imp['second_half_dir_accuracy']:.3f}"
                f"  delta: {imp['dir_accuracy_delta']:+.3f}"
            )
        if "first_half_label_rate" in imp:
            print(
                f"    up-label % first: {imp['first_half_label_rate']:.3f}"
                f"  second: {imp['second_half_label_rate']:.3f}"
                f"  delta: {imp['label_rate_delta']:+.3f}"
            )
        if "first_half_mean_prediction" in imp:
            print(
                f"    mean P(up) first: {imp['first_half_mean_prediction']:.3f}"
                f"  second: {imp['second_half_mean_prediction']:.3f}"
                f"  delta: {imp['mean_prediction_delta']:+.3f}"
            )
        if "train_rows_per_step" in imp:
            slope = imp["train_rows_per_step"]
            if config.walk_forward.expanding:
                window_label = "expanding window"
            else:
                window_label = (
                    f"rolling window ({config.walk_forward.initial_train_period} bars)"
                )
            sign = "+" if slope >= 0 else ""
            print(f"    train_rows {sign}{slope:.0f} rows/step ({window_label})")
        if "accuracy_train_rows_corr" in imp:
            print(f"    acc~train_rows corr: {imp['accuracy_train_rows_corr']:+.3f}")
    if curve.get("n_steps", 0) > 0:
        print(f"    mean OOS acc: {curve.get('mean_oos_accuracy', 0):.3f}")
        if "oos_accuracy_trend_slope" in curve:
            print(f"    acc trend   : {curve['oos_accuracy_trend_slope']:+.5f} / step")
        hints = degradation_hints(imp)
        if hints:
            print("  Likely drivers of degradation:")
            for hint in hints:
                print(f"    - {hint}")
    importance = result.learning.feature_importance
    if not importance.empty:
        metric = importance_metric_label(config.model.backend)
        print("-" * 64)
        print(f"  Top 10 features ({metric}):")
        for name, score in importance.head(10).items():
            print(f"    {name:<28}{format_importance_value(float(score)):>14}")
    if store is not None:
        counts = store.counts()
        print("-" * 64)
        print(f"  Logged predictions : {counts['predictions']:,}")
        print(f"  Logged outcomes    : {counts['outcomes']:,}")
    print("=" * 64 + "\n")


def _add_set_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config (dotted keys, YAML values). Repeatable.",
    )


def _add_progress_watch_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared flags for ``progress`` and ``checkpoint status``."""
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Stay open and refresh the display until Ctrl+C.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Refresh interval in seconds when using --watch (default: 2).",
    )


# ------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for all sub-commands."""
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--config", default="config/config.yaml", help="Path to YAML config.")
    parent.add_argument("--symbol", default=None, help="Override the primary symbol.")
    parent.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    _add_set_argument(parent)

    parser = argparse.ArgumentParser(prog="epoch-ai", description=__doc__, parents=[parent])
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser(
        "train",
        help="Train the AI (progressive learning + model registry).",
        parents=[parent],
    )
    p_train.add_argument("--bars", type=int, default=None, help="Cap history (defaults to live cache when sufficient).")
    p_train.add_argument("--max-steps", type=int, default=None, help="Cap walk-forward steps.")
    p_train.add_argument(
        "--full-history",
        action="store_true",
        help="Backfill multi-year history from exchange start (slow; ignores cached cap).",
    )
    p_train.add_argument(
        "--refresh-data",
        action="store_true",
        help="Re-fetch OHLCV from the exchange before training (default: cache-only).",
    )
    p_train.add_argument("--log-predictions", action="store_true", help="Persist to SQLite.")
    p_train.add_argument(
        "--no-register",
        action="store_true",
        help="Skip writing models to the registry.",
    )
    p_train.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any saved checkpoint and start at step 0 (does not delete it).",
    )
    p_train.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the walk-forward checkpoint and start training from step 0.",
    )
    p_train.set_defaults(func=cmd_train)

    p_run = sub.add_parser(
        "run",
        help="Run a trained model from the registry.",
        parents=[parent],
    )
    p_run.add_argument("--bars", type=int, default=None, help="Approx number of bars.")
    p_run.add_argument("--live-bars", type=int, default=500, help="Tail length to run.")
    p_run.add_argument("--model-version", default=None, help="Registry label (default: latest).")
    p_run.add_argument(
        "--retrain-every",
        type=int,
        default=0,
        help="Inline retrain every N bars (0 = frozen model).",
    )
    p_run.add_argument(
        "--live-feed",
        action="store_true",
        help="Simulate live data bar-by-bar (predict + trade each new candle).",
    )
    p_run.add_argument(
        "--live-stream",
        action="store_true",
        help="Stream live exchange candles via WebSocket (requires ccxt.pro).",
    )
    p_run.add_argument(
        "--log-predictions",
        action="store_true",
        help="Persist predictions/outcomes to SQLite for retrain (all run modes).",
    )
    p_run.add_argument(
        "--reserve-fraction",
        type=float,
        default=None,
        help="Fraction of session wins to set aside (not reinvested).",
    )
    p_run.add_argument(
        "--confirm-live",
        action="store_true",
        help="Enable real exchange orders (requires API keys; use with care).",
    )
    p_run.add_argument(
        "--replay",
        action="store_true",
        help="Historical replay session (batch mode, not live-feed).",
    )
    p_run.add_argument("--long-threshold", type=float, default=None)
    p_run.add_argument("--short-threshold", type=float, default=None)
    p_run.add_argument(
        "--policy",
        choices=["threshold", "baseline", "learned", "learned_with_baseline_fallback"],
        default=None,
        help="Trading policy backend (default: config trading.policy_backend).",
    )
    p_run.add_argument("--max-steps", type=int, default=None, help=argparse.SUPPRESS)
    p_run.set_defaults(func=cmd_run)

    p_train_policy = sub.add_parser(
        "train-policy",
        help="Train PPO trading policy on out-of-sample bar replay.",
        parents=[parent],
    )
    p_train_policy.add_argument(
        "--bars",
        type=int,
        default=None,
        help="Historical depth (OOS tail starts after initial_train_period).",
    )
    p_train_policy.add_argument(
        "--updates",
        type=int,
        default=None,
        help="Override rl.total_updates for a fast smoke run.",
    )
    p_train_policy.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="Override rl.rollout_steps.",
    )
    p_train_policy.add_argument(
        "--observation-mode",
        choices=("forecast", "embedding"),
        default=None,
        help="Policy observation: forecast summaries or TCN trunk embedding (ADR 0009).",
    )
    p_train_policy.add_argument(
        "--trunk-frozen",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="When embedding mode: freeze TCN trunk (Stage 1). Use --no-trunk-frozen for joint fine-tune.",
    )
    p_train_policy.add_argument(
        "--policy-loss-weight",
        type=float,
        default=None,
        help="Joint trunk fine-tune weight (Stage 2; requires --no-trunk-frozen and embedding mode).",
    )
    p_train_policy.set_defaults(func=cmd_train_policy)

    p_eval_holdout = sub.add_parser(
        "evaluate-holdout",
        help="Score predictor and policy on the untouched final holdout.",
        parents=[parent],
    )
    p_eval_holdout.add_argument("--bars", type=int, default=None, help="History depth cap.")
    p_eval_holdout.set_defaults(func=cmd_evaluate_holdout)

    p_predict = sub.add_parser(
        "predict",
        help="Multi-horizon forecast for the latest bar.",
        parents=[parent],
    )
    p_predict.add_argument("--bars", type=int, default=None, help="Warmup/history depth.")
    p_predict.add_argument("--model-version", default=None, help="Registry label (default: latest).")
    p_predict.add_argument("--json", action="store_true", help="Emit JSON payload.")
    p_predict.set_defaults(func=cmd_predict)

    p_dl = sub.add_parser("download", help="Download/synthesize and cache history.", parents=[parent])
    p_dl.add_argument("--bars", type=int, default=None, help="Approx number of bars.")
    p_dl.add_argument(
        "--full-history",
        action="store_true",
        help="Backfill multi-year history from exchange start (slow; ignores cached cap).",
    )
    p_dl.add_argument("--force", action="store_true", help="Ignore cache.")
    p_dl.set_defaults(func=cmd_download)

    p_bt = sub.add_parser("backtest", help="Run the progressive learning backtest.", parents=[parent])
    p_bt.add_argument("--bars", type=int, default=None, help="Approx number of bars.")
    p_bt.add_argument("--max-steps", type=int, default=None, help="Cap walk-forward steps.")
    p_bt.add_argument("--out", default="artifacts/backtests", help="Artifact output dir.")
    p_bt.add_argument("--log-predictions", action="store_true", help="Persist to SQLite store.")
    p_bt.add_argument("--register-models", action="store_true", help="Version each model.")
    p_bt.add_argument(
        "--long-threshold", type=float, default=None, help="Override P(up) long entry."
    )
    p_bt.add_argument(
        "--short-threshold", type=float, default=None, help="Override P(up) short entry."
    )
    p_bt.set_defaults(func=cmd_backtest)

    p_pt = sub.add_parser("paper-trade", help="Simulate near-real-time paper trading.", parents=[parent])
    p_pt.add_argument("--bars", type=int, default=None, help="Approx number of bars.")
    p_pt.add_argument("--live-bars", type=int, default=500, help="Held-out tail to trade.")
    p_pt.add_argument(
        "--long-threshold", type=float, default=None, help="Override P(up) long entry."
    )
    p_pt.add_argument(
        "--short-threshold", type=float, default=None, help="Override P(up) short entry."
    )
    p_pt.add_argument(
        "--retrain-every",
        type=int,
        default=0,
        help="Inline retrain every N bars (0 = never).",
    )
    p_pt.add_argument("--max-steps", type=int, default=None, help=argparse.SUPPRESS)
    p_pt.set_defaults(func=cmd_paper_trade)

    p_live = sub.add_parser("live", help="WebSocket stream or historical replay.", parents=[parent])
    p_live.add_argument("--bars", type=int, default=None, help="Bars for --replay mode.")
    p_live.add_argument("--live-bars", type=int, default=300, help="Replay tail length.")
    p_live.add_argument(
        "--replay",
        action="store_true",
        help="Replay historical tail instead of WebSocket stream.",
    )
    p_live.add_argument(
        "--retrain-every",
        type=int,
        default=0,
        help="Inline retrain every N bars during replay (0 = never).",
    )
    p_live.add_argument("--max-steps", type=int, default=None, help=argparse.SUPPRESS)
    p_live.set_defaults(func=cmd_live)

    p_rt = sub.add_parser("retrain", help="Retrain from logs or parquet history.", parents=[parent])
    p_rt.add_argument(
        "--min-new-samples",
        type=int,
        default=50,
        help="Minimum joined SQLite rows before using log-based retrain.",
    )
    p_rt.add_argument("--max-steps", type=int, default=None, help=argparse.SUPPRESS)
    p_rt.set_defaults(func=cmd_retrain)

    p_auto = sub.add_parser(
        "auto-retrain",
        help="Retrain a challenger and promote it only if it beats the champion.",
        parents=[parent],
    )
    p_auto.add_argument("--bars", type=int, default=None, help="Approx number of bars.")
    p_auto.add_argument(
        "--minutes",
        type=float,
        default=None,
        help="Loop the retrain/promote cycle for this many minutes (default: one cycle).",
    )
    p_auto.add_argument(
        "--interval-minutes",
        type=float,
        default=0.0,
        help="Sleep between cycles when looping (default 0 = back-to-back).",
    )
    p_auto.add_argument(
        "--promote-policy",
        action="store_true",
        help="After predictor promotion cycle, train/promote the PPO policy.",
    )
    p_auto.add_argument("--max-steps", type=int, default=None, help=argparse.SUPPRESS)
    p_auto.set_defaults(func=cmd_auto_retrain)

    p_tune = sub.add_parser("tune", help="Run a YAML config sweep.", parents=[parent])
    p_tune.add_argument(
        "--sweep",
        default="config/sweeps/example.yaml",
        help="YAML file listing experiments and overrides.",
    )
    p_tune.add_argument("--bars", type=int, default=None, help="Approx number of bars.")
    p_tune.add_argument("--max-steps", type=int, default=None, help="Cap walk-forward steps.")
    p_tune.add_argument("--out", default="artifacts/sweeps", help="Sweep output directory.")
    p_tune.set_defaults(func=cmd_tune)

    p_info = sub.add_parser("info", help="Print resolved configuration.", parents=[parent])
    p_info.add_argument("--max-steps", type=int, default=None, help=argparse.SUPPRESS)
    p_info.set_defaults(func=cmd_info)

    p_export = sub.add_parser("export", help="Export open-weights bundle + model card.", parents=[parent])
    p_export.add_argument("--dest", default="artifacts/exports", help="Output directory.")
    p_export.add_argument("--model-version", default=None, help="Registry label (default: latest).")
    p_export.set_defaults(func=cmd_export)

    p_serve = sub.add_parser("serve", help="Start FastAPI HTTP API.", parents=[parent])
    p_serve.add_argument("--host", default=None, help="Bind host (default: config.api.host).")
    p_serve.add_argument("--port", type=int, default=None, help="Bind port (default: config.api.port).")
    p_serve.set_defaults(func=cmd_serve)

    p_promote = sub.add_parser("promote", help="Promote best tune experiment.", parents=[parent])
    p_promote.add_argument(
        "--sweep-out",
        default="artifacts/sweeps",
        help="Directory written by tune command.",
    )
    p_promote.add_argument("--metric", default="sharpe", help="Metric to maximize.")
    p_promote.add_argument(
        "--out",
        default="config/promoted.yaml",
        help="Path for promoted config YAML.",
    )
    p_promote.set_defaults(func=cmd_promote)

    p_kill = sub.add_parser("kill-switch", help="Global trading halt control.", parents=[parent])
    p_kill.add_argument("action", choices=["status", "halt", "resume"], nargs="?", default="status")
    p_kill.add_argument("--reason", default="manual halt via CLI", help="Halt reason.")
    p_kill.set_defaults(func=cmd_kill_switch)

    p_tg = sub.add_parser("telegram", help="Start optional Telegram bot.", parents=[parent])
    p_tg.set_defaults(func=cmd_telegram)

    p_sched = sub.add_parser(
        "schedule-retrain",
        help="Periodic retrain scheduler loop.",
        parents=[parent],
    )
    p_sched.add_argument(
        "--interval-hours",
        type=float,
        default=None,
        help="Hours between retrains (default: adaptation.schedule_interval_hours).",
    )
    p_sched.add_argument(
        "--min-new-samples",
        type=int,
        default=50,
        help="Minimum joined SQLite rows for log-based retrain.",
    )
    p_sched.add_argument(
        "--max-cycles",
        type=int,
        default=1,
        help="Stop after N cycles (default 1 for smoke; omit loop with large value).",
    )
    p_sched.add_argument(
        "--promote",
        action="store_true",
        help="Use the challenger/champion gate each cycle (promote only if better).",
    )
    p_sched.add_argument(
        "--promote-policy",
        action="store_true",
        help="With --promote, also train/promote the PPO policy when rl.enabled.",
    )
    p_sched.set_defaults(func=cmd_schedule_retrain)

    p_ckpt = sub.add_parser(
        "checkpoint",
        help="Manage walk-forward training checkpoints.",
        parents=[parent],
    )
    p_ckpt_sub = p_ckpt.add_subparsers(dest="checkpoint_cmd", required=True)
    p_ckpt_seed = p_ckpt_sub.add_parser(
        "seed",
        help="Create a resume checkpoint after stopping an old train run.",
        parents=[parent],
    )
    p_ckpt_seed.add_argument(
        "--last-step",
        type=int,
        required=True,
        help="Last completed walk-forward step from logs (e.g. 75 for 'Step 75 | ...').",
    )
    p_ckpt_seed.add_argument(
        "--model-version",
        default=None,
        help="Registry label (default: v_{last_step+1}, e.g. v_76 after step 75).",
    )
    p_ckpt_seed.add_argument("--bars", type=int, default=None, help="Optional bar cap.")
    p_ckpt_seed.set_defaults(func=cmd_checkpoint_seed)
    p_ckpt_refresh = p_ckpt_sub.add_parser(
        "refresh",
        help="Refresh checkpoint fingerprint after resume-safe config changes.",
        parents=[parent],
    )
    p_ckpt_refresh.add_argument("--bars", type=int, default=None, help="Optional bar cap.")
    p_ckpt_refresh.set_defaults(func=cmd_checkpoint_refresh)
    p_ckpt_status = p_ckpt_sub.add_parser(
        "status",
        help="Show walk-forward progress from the checkpoint (alias for progress).",
        parents=[parent],
    )
    p_ckpt_status.add_argument(
        "--bars",
        type=int,
        default=None,
        help="Optional bar cap when recomputing resolved rows.",
    )
    p_ckpt_status.add_argument(
        "--refresh-rows",
        action="store_true",
        help="Recount resolved rows from cached data (slower, verifies row total).",
    )
    p_ckpt_status.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap total steps (same as train --max-steps).",
    )
    _add_progress_watch_arguments(p_ckpt_status)
    p_ckpt_status.set_defaults(func=cmd_progress)

    p_progress = sub.add_parser(
        "progress",
        help="Show walk-forward training position and steps remaining.",
        parents=[parent],
    )
    p_progress.add_argument(
        "--bars",
        type=int,
        default=None,
        help="Optional bar cap when recomputing resolved rows.",
    )
    p_progress.add_argument(
        "--refresh-rows",
        action="store_true",
        help="Recount resolved rows from cached data (slower, verifies row total).",
    )
    p_progress.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap total steps (same as train --max-steps).",
    )
    _add_progress_watch_arguments(p_progress)
    p_progress.set_defaults(func=cmd_progress)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
