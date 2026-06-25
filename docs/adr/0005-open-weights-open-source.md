# ADR 0005: Open weights and fully open source

## Status

Accepted

## Context

epochAI must be **open weights** (trained model files are publishable and usable by
anyone) and **fully open source** (code, config, training pipeline, and inference path
are transparent and self-hostable). This supports community audit, reproducibility,
and future interfaces (Telegram, website) without vendor lock-in.

## Decision

1. **Open weights** — every model version in `artifacts/models/v_*/` consists of plain
   files (`model.txt`, `metadata.json`) with no encryption or load-time license checks.
   Use `ModelRegistry.export_open_bundle()` to publish a shareable directory.

2. **Fully open source** — the entire train → run pipeline lives in this repository:
   features, walk-forward engine, registry, live trading engine, and services layer.
   Optional integrations (ccxt, mlflow, vectorbt) remain lazy-imported; core paths work
   without them.

3. **No license selected by the project automation** — the repository owner will choose
   a license separately if and when they decide. Agents and CI must **not** add or
   assign a `LICENSE` file or SPDX identifier unless the owner explicitly requests it.

## Consequences

- Third parties can train, export, and run models on their own hardware.
- We document openness in README and agent rules; we do not imply a specific legal
  license until the owner adds one.
- Proprietary extensions (if any) must live outside core modules or behind clearly
  optional plugins — never as the only path to train or infer.

## Non-goals

- Picking MIT, Apache-2.0, or any other license in this ADR or in tooling metadata.
- Encumbering weights with commercial-use clauses in code — legal terms are out of
  scope until the owner publishes a license.
