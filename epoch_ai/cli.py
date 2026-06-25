"""Command-line orchestration for epoch_ai.

Sub-commands:

* ``train``        - train the AI (progressive walk-forward + model registry).
* ``run``          - run a trained model (paper/replay session from registry).
* ``download``     - fetch the longest possible history (or synthesize it offline).
* ``backtest``     - run the progressive historical-learning backtest.
* ``paper-trade``  - simulate near-real-time paper trading with periodic updates.
* ``live``         - WebSocket stream or historical replay live loop.
* ``retrain``      - periodic retrain from SQLite logs or parquet history.
* ``tune``         - run a YAML sweep over config overrides.
* ``info``         - print the resolved configuration.

Run ``python -m epoch_ai <command> --help`` for details.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from epoch_ai.backtesting.engine import Backtester
from epoch_ai.config.overrides import apply_overrides, parse_set_args
from epoch_ai.config.settings import AppConfig
from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.execution.live_loop import run_bar_loop, run_scheduled_retrain
from epoch_ai.features.pipeline import FeaturePipeline
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.services.runtime import RuntimeService
from epoch_ai.services.training import TrainingService
from epoch_ai.tracking.mlflow_tracker import MLflowTracker
from epoch_ai.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


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


# --------------------------------------------------------------------- commands
def cmd_train(args: argparse.Namespace) -> int:
    """Train the AI via progressive walk-forward learning and register the model."""
    config = _load(args)
    service = TrainingService(config)
    result = service.train(
        n_bars=args.bars,
        max_steps=args.max_steps,
        log_predictions=args.log_predictions,
        register=not args.no_register,
    )
    print("\n=== Training complete ===")
    print(f"Symbol            : {config.primary_symbol}")
    print(f"Model version     : {result.model_version or '(not registered)'}")
    print(f"Walk-forward steps: {result.walk_forward_steps}")
    print(f"Final train rows  : {result.train_rows:,}")
    if not result.feature_importance.empty:
        print("Top features:")
        for name, gain in result.feature_importance.head(5).items():
            print(f"  {name:<28}{gain:>10.1f}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run a trained model from the registry (paper/replay session)."""
    config = _load(args)
    if args.long_threshold is not None:
        config.risk.long_threshold = args.long_threshold
    if args.short_threshold is not None:
        config.risk.short_threshold = args.short_threshold

    runtime = RuntimeService(config)
    status = runtime.status()
    if status.models_available == 0:
        logger.error("No trained models in registry. Run `python -m epoch_ai train` first.")
        return 1

    result = runtime.run_session(
        mode="replay" if args.replay else "paper",
        n_bars=args.bars,
        live_bars=args.live_bars,
        retrain_every=args.retrain_every,
        model_version=args.model_version,
    )
    print("\n=== Runtime session complete ===")
    print(f"Model version     : {runtime.status().model_version}")
    print(f"Bars processed    : {result.bars_processed}")
    print(f"Trades (fills)    : {result.fills}")
    print(f"Final equity      : {result.final_equity:,.2f}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Download or synthesize historical data and cache it as parquet."""
    config = _load(args)
    downloader = HistoricalDownloader(config)
    df = downloader.load_or_download(config.primary_symbol, n_bars=args.bars, force=args.force)
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
    return run_scheduled_retrain(config, min_new_samples=args.min_new_samples)


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


# ------------------------------------------------------------------- reporting
def _print_report(config: AppConfig, result, store: PredictionStore | None) -> None:
    m = result.metrics
    b = result.benchmark_metrics
    imp = result.learning_improvement
    curve = result.learning_curve
    print("\n" + "=" * 64)
    print(f"  PROGRESSIVE BACKTEST REPORT - {config.primary_symbol} {config.timeframe}")
    print("=" * 64)
    print(f"  Predictions made   : {len(result.learning.predictions):,}")
    print(f"  Walk-forward steps : {len(result.learning.step_history):,}")
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
        print("  Learning curve (out-of-sample directional accuracy):")
        print(f"    first half : {imp['first_half_accuracy']:.3f}")
        print(f"    second half: {imp['second_half_accuracy']:.3f}")
        print(f"    delta      : {imp['delta']:+.3f}")
    if curve.get("n_steps", 0) > 0:
        print(f"    mean OOS   : {curve.get('mean_oos_accuracy', 0):.3f}")
        if "oos_accuracy_trend_slope" in curve:
            print(f"    trend slope: {curve['oos_accuracy_trend_slope']:+.5f}")
    if not result.learning.feature_importance.empty:
        print("-" * 64)
        print("  Top 10 features by gain:")
        for name, gain in result.learning.feature_importance.head(10).items():
            print(f"    {name:<28}{gain:>14.1f}")
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
    p_train.add_argument("--bars", type=int, default=None, help="Approx number of bars.")
    p_train.add_argument("--max-steps", type=int, default=None, help="Cap walk-forward steps.")
    p_train.add_argument("--log-predictions", action="store_true", help="Persist to SQLite.")
    p_train.add_argument(
        "--no-register",
        action="store_true",
        help="Skip writing models to the registry.",
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
    p_run.add_argument("--replay", action="store_true", help="Alias for paper replay mode.")
    p_run.add_argument("--long-threshold", type=float, default=None)
    p_run.add_argument("--short-threshold", type=float, default=None)
    p_run.add_argument("--max-steps", type=int, default=None, help=argparse.SUPPRESS)
    p_run.set_defaults(func=cmd_run)

    p_dl = sub.add_parser("download", help="Download/synthesize and cache history.", parents=[parent])
    p_dl.add_argument("--bars", type=int, default=None, help="Approx number of bars.")
    p_dl.add_argument("--force", action="store_true", help="Ignore cache.")
    p_dl.set_defaults(func=cmd_download)

    p_bt = sub.add_parser("backtest", help="Run the progressive learning backtest.", parents=[parent])
    p_bt.add_argument("--bars", type=int, default=None, help="Approx number of bars.")
    p_bt.add_argument("--max-steps", type=int, default=None, help="Cap walk-forward steps.")
    p_bt.add_argument("--out", default="artifacts/backtests", help="Artifact output dir.")
    p_bt.add_argument("--log-predictions", action="store_true", help="Persist to SQLite store.")
    p_bt.add_argument("--register-models", action="store_true", help="Version each model.")
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

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
