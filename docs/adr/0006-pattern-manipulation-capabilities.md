# ADR 0006: Pattern recognition and manipulation-risk capabilities

## Status

Accepted

## Context

epochAI predicts short-horizon direction on liquid perpetual markets using gradient-boosted
trees over modular causal feature groups. Users want **secondary** capabilities:

1. Classic chart-pattern geometry (head & shoulders, double tops, flags, etc.).
2. Rug-pull / manipulation awareness without making that the primary training objective.

Tree models do not natively “see” chart shapes; discretionary pattern labels are sparse,
subjective, and easy to leak future information if confirmation rules are sloppy. Rug pulls
on DEX memecoins require on-chain data that BTC/USDT history alone cannot provide.

This ADR aligns with ADR 0002 (prediction vs execution separation): the direction model
may consume soft risk proxies as features, but hard trade blocks belong in execution.

## Decision

### 1. Pattern geometry as continuous features (not a CNN)

Add an optional `PatternFeatures` group emitting **continuous scores** in `[0, 1]` or
`[-1, 1]` — swing distances, double-top similarity, triangle convergence, flag pole ratio,
breakout strength, candlestick context. LightGBM combines these with derivatives,
microstructure, and TA.

**Why not a separate pattern CNN?** Keeps the open-weights pipeline reproducible, causal,
and config-toggleable; avoids a heavy second model and labeled pattern dataset.

### 2. Causal pivot confirmation lag

Swing pivots are **confirmed** only after `pivot_confirm_bars` subsequent bars print.
Features at bar `t` never use OHLCV from bars after `t`. Trailing `pivot_confirm_bars`
rows are masked to neutral zero (pivots not yet confirmable at series end).

### 3. Manipulation proxies in features; hard gate in execution

| Concern | Layer | Mechanism |
|---------|-------|-----------|
| Wash volume, wick clusters, return skew | `ManipulationFeatures` | Soft scores fed to direction model |
| LP drain, holder concentration | Extended `OnChainFeatures` | When columns joined externally |
| Block / scale trades above threshold | `SafetyScorer` + `RiskManager` | Execution-only; default **off** |

True rug-pull classification requires labeled DEX events and a dedicated dataset — **out of
scope** for the core BTC trainer. OHLCV proxies catch manipulation-like regimes on majors;
on-chain columns activate when trading alts.

### 4. Defaults preserve backward compatibility

`features.patterns`, `features.manipulation`, and `safety.enabled` default to `false`.

## Consequences

**Positive**

- Pattern and manipulation signal available without changing the primary target.
- Safety gate is opt-in and isolated from training.
- All new columns follow existing `FeatureGroup` + config-driven patterns.

**Negative**

- Loop-based swing detection may be slower on very long histories (optimize if needed).
- BTC-only training will not learn DEX rug semantics without future data plumbing.

**Follow-up (not in this ADR)**

- DEX token downloader and labeled rug dataset.
- Multi-timeframe pattern context (1h/4h resampled).
- Multi-task direction + manipulation head.

## Non-goals

- DEX ingestion pipeline.
- Multi-task training heads.
- Changing primary symbol or prediction horizon.
- Pattern detection as the dominant feature family.
