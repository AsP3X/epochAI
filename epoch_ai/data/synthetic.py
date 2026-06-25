"""Synthetic but *realistic* market-data generator.

Public crypto exchange APIs are frequently geo-blocked from cloud/CI environments.
To keep the entire progressive-learning pipeline runnable offline, this module
synthesises a multi-year OHLCV series with derivatives context (funding rate, open
interest, estimated liquidations) using a **regime-switching** price process.

The regimes (bull / bear / chop / high-volatility crash) give the model exposure to
diverse market conditions - exactly what the progressive learning engine needs to
discover relationships between context and forward price moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epoch_ai.utils.timeframe import timeframe_to_minutes, timeframe_to_pandas_freq

# Per-bar (annualised-ish) drift/vol per regime, plus expected dwell length in bars.
_REGIMES: dict[str, dict[str, float]] = {
    "bull": {"drift": 0.00018, "vol": 0.006, "dwell": 1500.0},
    "bear": {"drift": -0.00016, "vol": 0.008, "dwell": 1200.0},
    "chop": {"drift": 0.00000, "vol": 0.004, "dwell": 1800.0},
    "crash": {"drift": -0.00060, "vol": 0.020, "dwell": 250.0},
}
_REGIME_NAMES = list(_REGIMES.keys())


def generate_synthetic_ohlcv(
    *,
    timeframe: str,
    start: str,
    n_bars: int,
    seed: int = 7,
    start_price: float = 9_000.0,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV + derivatives dataset.

    Args:
        timeframe: Bar size (e.g. ``"15m"``).
        start: ISO start timestamp for the first (oldest) bar.
        n_bars: Number of bars to generate.
        seed: RNG seed for reproducibility.
        start_price: Price of the first bar.

    Returns:
        A DataFrame indexed by a UTC ``timestamp`` with columns:
        ``open, high, low, close, volume, funding_rate, open_interest,
        liquidations``.
    """
    if n_bars < 1:
        raise ValueError("n_bars must be >= 1")

    rng = np.random.default_rng(seed)
    minutes = timeframe_to_minutes(timeframe)
    bars_per_day = (24 * 60) / minutes
    index = pd.date_range(
        start=start, periods=n_bars, freq=timeframe_to_pandas_freq(timeframe), tz="UTC"
    )

    # --- Regime path (Markov-ish switching driven by dwell times) -----------
    regimes = np.empty(n_bars, dtype=object)
    current = rng.choice(_REGIME_NAMES)
    for i in range(n_bars):
        regimes[i] = current
        dwell = _REGIMES[current]["dwell"]
        if rng.random() < 1.0 / dwell:
            current = rng.choice(_REGIME_NAMES)

    drift = np.array([_REGIMES[r]["drift"] for r in regimes])
    vol = np.array([_REGIMES[r]["vol"] for r in regimes])

    # --- Log-return price process ------------------------------------------
    shocks = rng.standard_normal(n_bars)
    log_returns = drift + vol * shocks
    log_price = np.log(start_price) + np.cumsum(log_returns)
    close = np.exp(log_price)

    open_ = np.empty(n_bars)
    open_[0] = start_price
    open_[1:] = close[:-1]

    # Intrabar range scales with volatility.
    wick = np.abs(rng.standard_normal(n_bars)) * vol * close
    high = np.maximum(open_, close) + wick * 0.6
    low = np.minimum(open_, close) - wick * 0.6

    # --- Volume: base + volatility-driven spikes ---------------------------
    base_vol = 800.0
    intraday = 1.0 + 0.4 * np.sin(2 * np.pi * (np.arange(n_bars) % bars_per_day) / bars_per_day)
    volume = base_vol * intraday * (1.0 + 12.0 * np.abs(log_returns)) * (
        1.0 + 0.3 * rng.random(n_bars)
    )

    # --- Derivatives context -----------------------------------------------
    # Funding rate: mean-reverting around 0, biased by drift; clipped to +/-0.3%.
    funding = np.zeros(n_bars)
    for i in range(1, n_bars):
        funding[i] = 0.985 * funding[i - 1] + 0.05 * drift[i] + 0.00003 * rng.standard_normal()
    funding = np.clip(funding, -0.003, 0.003)

    # Open interest: random-walk in log space, grows over time, dips on crashes.
    oi_shock = rng.standard_normal(n_bars) * 0.01
    oi_trend = np.linspace(0.0, 1.2, n_bars)
    open_interest = 50_000.0 * np.exp(oi_trend + np.cumsum(oi_shock) * 0.1)
    open_interest *= np.where(regimes == "crash", 0.85, 1.0)

    # Estimated liquidations: spike on large adverse moves / high vol.
    liquidations = np.maximum(0.0, (np.abs(log_returns) - 0.01)) * open_interest * 2.0
    liquidations *= 1.0 + 0.5 * rng.random(n_bars)

    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": funding,
            "open_interest": open_interest,
            "liquidations": liquidations,
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame
