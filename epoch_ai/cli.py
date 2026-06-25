"""Command-line orchestration for epoch_ai.

Sub-commands:

* ``download``     - fetch the longest possible history (or synthesize it offline).
* ``backtest``     - run the progressive historical-learning backtest.
* ``paper-trade``  - simulate near-real-time paper trading with periodic updates.
* ``info``         - print the resolved configuration.

Run ``python -m epoch_ai <command> --help`` for details.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from epoch_ai.backtesting.engine import Backtester
from epoch_ai.config.settings import AppConfig, load_config
from epoch_ai.data.downloader import HistoricalDownloader
from epoch_ai.execution.paper_trader import PaperTrader
from epoch_ai.execution.risk import RiskManager
from epoch_ai.features.pipeline import FeaturePipeline
from epoch_ai.logging_system.store import PredictionStore
from epoch_ai.models.lightgbm_model import LightGBMModel
from epoch_ai.tracking.mlflow_tracker import MLflowTracker
from epoch_ai.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _load(args: argparse.Namespace) -> AppConfig:
    config = load_config(args.config)
    if getattr(args, "symbol", None):
        config.symbols = [args.symbol]
    if getattr(args, "max_steps", None) is not None:
        config.walk_forward.max_steps = args.max_steps
    return config


# --------------------------------------------------------------------- commands
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

    # Persist artifacts.
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "strategy": result.metrics,
                "benchmark": result.benchmark_metrics,
                "learning_improvement": result.learning_improvement,
            },
            indent=2,
        )
    )
    result.equity_curve.rename("equity").to_csv(out_dir / "equity_curve.csv")
    result.learning.step_history.to_csv(out_dir / "step_history.csv", index=False)
    result.learning.feature_importance.rename("gain").to_csv(out_dir / "feature_importance.csv")

    _print_report(config, result, store)
    if store is not None:
        store.close()
    return 0


def cmd_paper_trade(args: argparse.Namespace) -> int:
    """Simulate near-real-time paper trading over the most recent bars.

    Live exchange streaming is geo-blocked in many environments, so this trains on
    all-but-the-last ``--live-bars`` candles and then steps bar-by-bar through the
    held-out tail, predicting, applying risk rules and (paper) executing - exactly
    the live loop, driven by recorded data.
    """
    config = _load(args)
    if args.long_threshold is not None:
        config.risk.long_threshold = args.long_threshold
    if args.short_threshold is not None:
        config.risk.short_threshold = args.short_threshold

    downloader = HistoricalDownloader(config)
    market = downloader.load_or_download(config.primary_symbol, n_bars=args.bars)
    features = FeaturePipeline(config).transform(market)

    from epoch_ai.features.pipeline import build_target, forward_return

    y = build_target(market, config.prediction)
    fwd = forward_return(market, config.prediction.horizon)
    data = features.join(y).join(fwd).dropna(subset=["target", "forward_return"])
    feature_cols = list(features.columns)

    live_bars = min(args.live_bars, len(data) - config.walk_forward.initial_train_period)
    if live_bars < 1:
        logger.error("Not enough data for paper trading. Increase --bars.")
        return 1

    split = len(data) - live_bars
    model = LightGBMModel(config.model, task=config.prediction.task)
    model.fit(data[feature_cols].iloc[:split], data["target"].iloc[:split])

    risk_manager = RiskManager(config.risk, config.prediction)
    trader = PaperTrader(config.risk)
    close = market["close"]

    for pos in range(split, len(data)):
        ts = data.index[pos]
        price = float(close.loc[ts])
        raw_pred = float(model.predict(data[feature_cols].iloc[[pos]])[0])
        decision = risk_manager.decide(raw_pred)
        trader.rebalance(str(ts), price, decision)
        trader.mark_to_market(float(data["forward_return"].iloc[pos]) / config.prediction.horizon)

    print("\n=== Paper-trading summary ===")
    print(f"Symbol            : {config.primary_symbol}")
    print(f"Bars simulated    : {live_bars}")
    print(f"Trades (fills)    : {len(trader.fills)}")
    print(f"Starting capital  : {config.risk.initial_capital:,.2f}")
    print(f"Final equity      : {trader.equity:,.2f}")
    print(f"Return            : {(trader.equity / config.risk.initial_capital - 1) * 100:,.2f}%")
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


# ------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for all sub-commands."""
    parser = argparse.ArgumentParser(prog="epoch-ai", description=__doc__)
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config.")
    parser.add_argument("--symbol", default=None, help="Override the primary symbol.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Download/synthesize and cache history.")
    p_dl.add_argument("--bars", type=int, default=None, help="Approx number of bars.")
    p_dl.add_argument("--force", action="store_true", help="Ignore cache.")
    p_dl.set_defaults(func=cmd_download)

    p_bt = sub.add_parser("backtest", help="Run the progressive learning backtest.")
    p_bt.add_argument("--bars", type=int, default=None, help="Approx number of bars.")
    p_bt.add_argument("--max-steps", type=int, default=None, help="Cap walk-forward steps.")
    p_bt.add_argument("--out", default="artifacts/backtests", help="Artifact output dir.")
    p_bt.add_argument("--log-predictions", action="store_true", help="Persist to SQLite store.")
    p_bt.add_argument("--register-models", action="store_true", help="Version each model.")
    p_bt.set_defaults(func=cmd_backtest)

    p_pt = sub.add_parser("paper-trade", help="Simulate near-real-time paper trading.")
    p_pt.add_argument("--bars", type=int, default=None, help="Approx number of bars.")
    p_pt.add_argument("--live-bars", type=int, default=500, help="Held-out tail to trade.")
    p_pt.add_argument(
        "--long-threshold", type=float, default=None, help="Override P(up) long entry."
    )
    p_pt.add_argument(
        "--short-threshold", type=float, default=None, help="Override P(up) short entry."
    )
    p_pt.add_argument("--max-steps", type=int, default=None, help=argparse.SUPPRESS)
    p_pt.set_defaults(func=cmd_paper_trade)

    p_info = sub.add_parser("info", help="Print resolved configuration.")
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
