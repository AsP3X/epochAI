"""Tests for kill switch."""

from __future__ import annotations

from epoch_ai.execution.kill_switch import KillSwitch


def test_kill_switch_halt_resume(tmp_path):
    path = str(tmp_path / "kill.json")
    ks = KillSwitch(path)
    assert not ks.is_halted()
    ks.halt("test halt")
    assert ks.is_halted()
    assert ks.read().reason == "test halt"
    ks.resume()
    assert not ks.is_halted()
