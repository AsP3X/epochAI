"""Terminal progress helpers for long-running CLI operations."""

from __future__ import annotations

import os
import shutil
import sys
import time
from collections import deque

_VT_ENABLED = False


def format_bytes(n_bytes: int) -> str:
    """Return a compact human-readable byte size."""
    if n_bytes < 0:
        raise ValueError("n_bytes must be >= 0")
    value = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def estimate_parquet_bytes(n_bars: int) -> int:
    """Rough compressed-parquet size for OHLCV rows indexed by timestamp."""
    return max(n_bars, 0) * 36


def _terminal_columns() -> int:
    try:
        return max(40, shutil.get_terminal_size(fallback=(80, 24)).columns)
    except OSError:
        return 80


def _short_count(n: int) -> str:
    if n >= 1_000_000:
        if n % 1_000_000 == 0:
            return f"{n // 1_000_000}M"
        text = f"{n / 1_000_000:.2f}".rstrip("0").rstrip("." )
        return f"{text}M"
    if n >= 10_000:
        return f"{n // 1_000}k"
    if n >= 1_000:
        if n % 1_000 == 0:
            return f"{n // 1_000}k"
        text = f"{n / 1_000:.1f}".rstrip("0").rstrip("." )
        return f"{text}k"
    return str(n)


def _short_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _short_rate(rate: float) -> str:
    if rate <= 0:
        return "--/s"
    if rate >= 10_000:
        return f"{rate / 1_000:.1f}k/s"
    return f"{rate:,.0f}/s"


def _enable_vt_on_windows() -> None:
    """Best-effort enable ANSI clear-line sequences on Windows consoles."""
    global _VT_ENABLED
    if _VT_ENABLED or os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        enable_vt = 0x0004
        for handle_id in (-11, -12):  # stdout, stderr
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_uint()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | enable_vt)
                _VT_ENABLED = True
    except (AttributeError, ImportError, OSError):
        _VT_ENABLED = False


class DownloadProgressBar:
    """Single-line stderr progress bar for paginated historical downloads."""

    def __init__(
        self,
        *,
        total: int,
        desc: str,
        unit: str = "bars",
        enabled: bool | None = None,
    ) -> None:
        self.total = max(total, 1)
        self.desc = desc
        self.unit = unit
        self.current = 0
        self._stream = sys.stderr
        self.enabled = self._stream.isatty() if enabled is None else enabled
        self._last_render = 0.0
        self._closed = False
        self._rate_anchor = 0
        self._rate_start = time.monotonic()
        self._recent_rates: deque[float] = deque(maxlen=8)
        self._last_tick = self._rate_start
        self._last_count = 0
        if self.enabled:
            _enable_vt_on_windows()

    def __enter__(self) -> DownloadProgressBar:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def begin_rate_tracking(self) -> None:
        """Measure ETA from the current position (ignores cached baseline bars)."""
        self._rate_anchor = self.current
        now = time.monotonic()
        self._rate_start = now
        self._recent_rates.clear()
        self._last_tick = now
        self._last_count = self.current

    def set_total(self, total: int) -> None:
        """Adjust the target when fewer bars exist than requested."""
        self.total = max(total, 1)
        if self.current > self.total:
            self.current = self.total
        self._render(force=True)

    def advance_to(self, n: int, *, render: bool = True) -> None:
        """Move the bar to ``n`` downloaded units (monotonic, capped at total)."""
        new = min(max(n, 0), self.total)
        if new > self._last_count:
            now = time.monotonic()
            delta = new - self._last_count
            dt = now - self._last_tick
            if dt > 0:
                self._recent_rates.append(delta / dt)
            self._last_tick = now
            self._last_count = new
        self.current = new
        if self.enabled and render:
            self._render(force=False)

    def refresh(self) -> None:
        """Redraw the progress line immediately."""
        if self.enabled:
            self._render(force=True)

    def update(self, n: int = 1) -> None:
        self.advance_to(self.current + n)

    def _effective_rate(self) -> float:
        if self._recent_rates:
            return sum(self._recent_rates) / len(self._recent_rates)
        elapsed = time.monotonic() - self._rate_start
        gained = self.current - self._rate_anchor
        if elapsed >= 0.5 and gained > 0:
            return gained / elapsed
        return 0.0

    def _label(self) -> str:
        return self.desc.split()[-1] if " " in self.desc else self.desc

    def _build_line(self, cols: int) -> str:
        pct = min(1.0, self.current / self.total) * 100.0
        rate = self._effective_rate()
        remaining = (self.total - self.current) / rate if rate > 0 else 0.0
        cur_size = format_bytes(estimate_parquet_bytes(self.current))
        tot_size = format_bytes(estimate_parquet_bytes(self.total))
        eta = "done" if self.current >= self.total else _short_eta(remaining)

        prefix = f"{self._label()} "
        suffix = (
            f" {_short_count(self.current)}/{_short_count(self.total)} "
            f"{pct:4.1f}% {_short_rate(rate)} ETA {eta} {cur_size}/~{tot_size}"
        )

        # Agent: give the bar all leftover width so it grows with the terminal.
        bar_width = max(12, cols - len(prefix) - len(suffix) - 3)
        filled = int(bar_width * (pct / 100.0))
        if filled >= bar_width:
            bar = "=" * bar_width
        else:
            bar = "=" * filled + ">" + " " * (bar_width - filled - 1)

        return f"{prefix}[{bar}]{suffix}"

    def _render(self, *, force: bool) -> None:
        now = time.monotonic()
        if not force and self.current < self.total and now - self._last_render < 0.1:
            return
        self._last_render = now

        cols = _terminal_columns()
        line = self._build_line(cols)
        if len(line) > cols - 1:
            line = line[: cols - 4] + "..."

        # Agent: never exceed terminal width; clear row with ANSI when available.
        if _VT_ENABLED or os.environ.get("WT_SESSION") or os.environ.get("TERM"):
            self._stream.write("\r\x1b[2K" + line)
        else:
            self._stream.write("\r" + line.ljust(min(len(line), cols - 1)))
        self._stream.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.enabled:
            self._render(force=True)
            self._stream.write("\n")
            self._stream.flush()


def render_live_text(text: str, *, stream=None) -> None:
    """Redraw ``text`` in-place on a TTY (full screen clear each frame)."""
    if stream is None:
        stream = sys.stdout
    if not stream.isatty():
        stream.write(text)
        if not text.endswith("\n"):
            stream.write("\n")
        stream.write("\n")
        stream.flush()
        return

    _enable_vt_on_windows()
    # Agent: full erase every frame; partial clear (H+J) left stale lines on Windows.
    stream.write("\x1b[2J\x1b[H")
    stream.write(text.rstrip("\n"))
    stream.write("\x1b[J\n")
    stream.flush()


def build_fraction_bar(completed: int, total: int, width: int = 30) -> str:
    """Return an ASCII progress bar for ``completed`` of ``total``."""
    if total <= 0:
        return "[" + " " * width + "]"
    pct = min(1.0, max(0.0, completed / total))
    filled = int(width * pct)
    if filled >= width:
        inner = "=" * width
    elif filled <= 0:
        inner = ">" + " " * (width - 1)
    else:
        inner = "=" * filled + ">" + " " * (width - filled - 1)
    return f"[{inner}]"
