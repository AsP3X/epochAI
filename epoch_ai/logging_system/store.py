"""SQLite-backed store for prediction and outcome logs.

The store is the system's memory: every prediction (with its full feature vector) and
every realised outcome (with rich context) is persisted here. The progressive
learning engine and retraining jobs read back from it to assemble training datasets
that include historical context.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from epoch_ai.logging_system.schemas import OutcomeLog, PredictionLog
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    symbol        TEXT    NOT NULL,
    model_version TEXT    NOT NULL,
    horizon       INTEGER NOT NULL,
    prediction    REAL    NOT NULL,
    confidence    REAL    NOT NULL,
    signal        INTEGER NOT NULL,
    entry_price   REAL,
    features      TEXT    NOT NULL,
    created_at    TEXT    DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_pred_symbol_ts ON predictions(symbol, timestamp);

CREATE TABLE IF NOT EXISTS outcomes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id     INTEGER NOT NULL UNIQUE,
    resolve_timestamp TEXT    NOT NULL,
    forward_return    REAL    NOT NULL,
    realized_label    INTEGER NOT NULL,
    exit_price        REAL,
    context           TEXT    NOT NULL,
    created_at        TEXT    DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);
"""


class PredictionStore:
    """A small DAO over the predictions/outcomes SQLite database."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ----------------------------------------------------------------- writes
    def log_prediction(self, pred: PredictionLog) -> int:
        """Insert a prediction and return its new row id."""
        cur = self._conn.execute(
            """
            INSERT INTO predictions
                (timestamp, symbol, model_version, horizon, prediction,
                 confidence, signal, entry_price, features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pred.timestamp,
                pred.symbol,
                pred.model_version,
                pred.horizon,
                float(pred.prediction),
                float(pred.confidence),
                int(pred.signal),
                pred.entry_price,
                json.dumps(pred.features),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def log_outcome(self, outcome: OutcomeLog) -> None:
        """Insert (or replace) the outcome for a prediction."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO outcomes
                (prediction_id, resolve_timestamp, forward_return,
                 realized_label, exit_price, context)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                outcome.prediction_id,
                outcome.resolve_timestamp,
                float(outcome.forward_return),
                int(outcome.realized_label),
                outcome.exit_price,
                json.dumps(outcome.context, default=float),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------ reads
    def predictions_frame(self, symbol: str | None = None) -> pd.DataFrame:
        """Return all predictions (optionally filtered by symbol) as a DataFrame."""
        query = "SELECT * FROM predictions"
        params: tuple = ()
        if symbol is not None:
            query += " WHERE symbol = ?"
            params = (symbol,)
        return pd.read_sql_query(query, self._conn, params=params)

    def outcomes_frame(self) -> pd.DataFrame:
        """Return all outcomes as a DataFrame."""
        return pd.read_sql_query("SELECT * FROM outcomes", self._conn)

    def counts(self) -> dict[str, int]:
        """Return row counts for quick status checks."""
        preds = self._conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        outs = self._conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
        return {"predictions": int(preds), "outcomes": int(outs)}

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> PredictionStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
