"""Tests for download progress helpers."""

from __future__ import annotations

from epoch_ai.utils.progress import (
    DownloadProgressBar,
    _short_count,
    estimate_parquet_bytes,
    format_bytes,
)


def test_format_bytes():
    assert format_bytes(512) == "512 B"
    assert format_bytes(2048) == "2.0 KB"
    assert format_bytes(5 * 1024 * 1024) == "5.0 MB"


def test_estimate_parquet_bytes_scales_with_bars():
    assert estimate_parquet_bytes(1000) == 36_000
    assert estimate_parquet_bytes(0) == 0


def test_short_count():
    assert _short_count(10_000_000) == "10M"
    assert _short_count(1_500_000) == "1.5M"
    assert _short_count(233_159) == "233k"
    assert _short_count(21_000) == "21k"


def test_download_progress_bar_disabled_does_not_write(capsys):
    with DownloadProgressBar(total=1000, desc="Test", enabled=False) as bar:
        bar.begin_rate_tracking()
        bar.advance_to(500)
        bar.advance_to(1000)
    captured = capsys.readouterr()
    assert captured.err == ""


def test_download_progress_rate_ignores_baseline(monkeypatch):
    clock = {"t": 0.0}

    def mono() -> float:
        return clock["t"]

    monkeypatch.setattr("epoch_ai.utils.progress.time.monotonic", mono)

    bar = DownloadProgressBar(total=1000, desc="Test", enabled=False)
    bar.advance_to(800)
    bar.begin_rate_tracking()
    clock["t"] = 1.0
    bar.advance_to(900)
    # Rate uses progress since begin_rate_tracking (100 bars/s), not since bar 0.
    assert bar._effective_rate() == 100.0
    bar._recent_rates.append(200.0)
    assert bar._effective_rate() == 150.0
