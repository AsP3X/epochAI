"""Tests for audit log."""

from __future__ import annotations

import json

from epoch_ai.execution.audit_log import AuditLog


def test_audit_log_append(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path)
    log.append("prediction", {"symbol": "BTC/USDT", "signal": 1})
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["type"] == "prediction"
    assert record["symbol"] == "BTC/USDT"
