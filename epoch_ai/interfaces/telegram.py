"""Telegram message formatting for forecasts and trade alerts."""

from __future__ import annotations

from typing import Any


def format_forecast_summary(payload: dict[str, Any]) -> str:
    """Render a multi-horizon live payload as a compact Telegram message."""
    if payload.get("error") == "no_trained_model":
        return "No trained models. Run `train` first."

    lines = [
        "epochAI forecast",
        f"as_of: {payload.get('as_of', '?')}",
        f"close: {payload.get('last_close', 0):.2f}",
        f"model: {payload.get('model_version', '?')}",
        "",
    ]
    for h in payload.get("horizons", []):
        flag = "ok" if h.get("reliable") else "low"
        lines.append(
            f"{h.get('label', '?'):>4}  p_up={h.get('p_up', 0):.3f}  "
            f"p50={h.get('price_p50', 0):.2f}  conf={h.get('confidence', 0):.2f}  [{flag}]"
        )
    baseline = payload.get("baseline") or {}
    if baseline:
        lines.extend(
            [
                "",
                f"baseline signal={baseline.get('signal', 0):+d}  "
                f"p_up={baseline.get('weighted_p_up', 0.5):.3f}  "
                f"heads={baseline.get('n_heads_used', 0)}",
            ]
        )
    return "\n".join(lines)


def format_trade_alert(
    *,
    symbol: str,
    timestamp: str,
    signal: int,
    price: float,
    equity: float,
    model_version: str,
) -> str:
    """Format a fill/trade event push notification."""
    side = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(signal, str(signal))
    return (
        f"Trade {side} {symbol}\n"
        f"time: {timestamp}\n"
        f"price: {price:.2f}\n"
        f"equity: {equity:.2f}\n"
        f"model: {model_version}"
    )


def format_safety_alert(*, event: str, symbol: str, timestamp: str, detail: str) -> str:
    """Format kill-switch or calibration safety events."""
    return f"SAFETY {event}\n{symbol} @ {timestamp}\n{detail}"
