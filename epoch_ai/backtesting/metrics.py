"""Trading-performance metrics computed directly from a returns series.

Implemented natively (numpy/pandas) so metrics are always available even without
``vectorbt``. All ratios use a configurable per-period -> annual scaling factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_series(returns) -> pd.Series:
    if isinstance(returns, pd.Series):
        return returns.dropna().astype(float)
    return pd.Series(np.asarray(returns, dtype=float)).dropna()


def max_drawdown(equity: pd.Series) -> float:
    """Return the maximum drawdown (a negative number) of an equity curve."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def compute_metrics(
    returns,
    *,
    annualization: float = 1.0,
    periods_per_year: float | None = None,
) -> dict[str, float]:
    """Compute a standard suite of trading metrics from per-period returns.

    Args:
        returns: Per-period strategy returns (fractional, not percent).
        annualization: ``sqrt(periods_per_year)`` factor for Sharpe/Sortino scaling.
        periods_per_year: Used for CAGR/Calmar; defaults to ``annualization ** 2``.

    Returns:
        Dict with total return, CAGR, Sharpe, Sortino, Calmar, max drawdown, profit
        factor, win rate, volatility and trade count.
    """
    r = _to_series(returns)
    n = len(r)
    if n == 0:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "volatility": 0.0,
            "n_periods": 0,
        }

    if periods_per_year is None:
        periods_per_year = annualization ** 2

    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)

    mean = r.mean()
    std = r.std(ddof=1) if n > 1 else 0.0
    downside = r[r < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0

    sharpe = float(mean / std * annualization) if std > 0 else 0.0
    sortino = float(mean / downside_std * annualization) if downside_std > 0 else 0.0

    years = n / periods_per_year if periods_per_year > 0 else 0.0
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else 0.0

    mdd = max_drawdown(equity)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0

    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    profit_factor = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)
    win_rate = float((r > 0).mean())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": mdd,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "volatility": float(std * annualization),
        "n_periods": int(n),
    }
