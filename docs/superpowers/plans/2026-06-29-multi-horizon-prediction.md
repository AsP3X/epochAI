# Multi-Horizon Price Prediction + Learned Trading Policy — Implementation Plan

> **For agentic workers:** Implement task-by-task. Steps use checkbox (`- [ ]`) syntax.
> Respect `ml-causality.mdc`, `prediction-execution-separation.mdc`, `config-driven.mdc`,
> `regression-testing.mdc`, `definition-of-done.mdc`, and `open-weights.mdc` throughout.
> **This plan deliberately crosses the prediction/execution boundary** (a learned RL trading
> policy) — authorized by the repo owner on 2026-06-29 and recorded in ADR 0008.

**Goal:** Forecast the near-future of BTC price at multiple horizons —
**immediate/next-bar, 5m, 10m, 15m, 30m, 1hr** — emitting per horizon a calibrated
direction `P(up)`, expected return, and **quantile price band** (`p10/p50/p90`). One canonical
prediction artifact feeds two consumers:

1. **Candle-graph overlay** — live forward "cone of uncertainty" + historical
   predicted-vs-realized + per-horizon reliability.
2. **Automatic trading bot** — a **reinforcement-learning policy** that *learns from scratch*
   when to enter/exit and how large to size, using the predictor's outputs + features as
   observations. Paper-only for v1.

---

## Decisions locked

### Round 1 — prediction shape
| Decision | Choice |
|---|---|
| Base timeframe | **1-minute** candles |
| Horizons (1m candles) | `[1, 5, 10, 15, 30, 60]` → `1m, 5m, 10m, 15m, 30m, 1hr` (`1` = "immediate"/next bar) |
| Per-horizon output | Quantile band `p10/p50/p90` + calibrated `P(up)` + expected return |
| Model | Single **multi-head `evolved_nn`** (shared trunk, per-horizon heads) |
| Data source | **Real Binance 1m via CCXT, run locally** (not geo-blocked on owner's machine) |
| History depth | **Maximum available** (multi-year) |
| Hardware | **NVIDIA CUDA GPU** |
| Base strategy | **Replace** 15m entirely with the 1m multi-horizon model |
| Promotion metric | **Confidence-weighted average of per-horizon Brier/logloss** |
| Chart delivery | **Library functions returning JSON** (RuntimeService + forecast_api) + **store query** for history |

### Round 2 — training budget & risk realism
| Decision | Choice |
|---|---|
| Initial training | **Full progressive walk-forward over all available history** (heavy; accepted) |
| Ongoing training | Switch to **coarse scheduled auto-retrain** after the initial run (agent recommendation) |
| Initial train window | Largest feasible (use as much history as possible before first prediction) |
| Fees/slippage | **Realistic taker (~0.05%) + slippage** |
| Trade frequency | **Configurable**, default **selective** (high-confidence only) |
| Unreliable heads | **Auto-drop** heads below a coverage/Brier reliability floor |
| Adaptation | **Scheduled auto-retrain + challenger/champion promotion** |

### Round 3 — learned trading policy (RL)
| Decision | Choice |
|---|---|
| Approach | **Hybrid**: supervised multi-horizon predictor (chart + policy inputs) **+** separate RL policy that learns entry/exit/size |
| Reward | **Risk-adjusted return (Sharpe-like) net of fees** |
| Action space | **Direction (long/short/flat) + position size + exit timing** |
| Guardrails | **Hard caps** (max position, leverage cap, max-drawdown kill-switch); learning happens inside them |
| Observations | **Per-horizon p_up/return/bands + raw features + current position** |
| Algorithm | **PPO** (continuous sizing, stable) unless evidence says otherwise |
| Market / direction | **Perp long + short** (binanceusdm) |
| Execution target | **Paper / replay only** for v1 (real keys later) |

### Round 4 — concrete defaults (all config-adjustable)
| Knob | Value | Plain meaning |
|---|---|---|
| Leverage cap | **Configurable; default 1x (no leverage)** | Max borrowing the bot may ever use; default = trades only with cash it has. Safest. |
| Max-drawdown kill-switch | **20%** | Halt trading if equity falls 20% from peak. |
| Paper starting equity | **$10,000** | Pretend bankroll; sets PnL scale only. |
| Reliability floor | **Moderate** | Drop a horizon head from decisions if band coverage is off by >0.15 OR Brier ≥ no-skill baseline. |
| Coarse auto-retrain cadence | **Daily** | After the initial full run, retrain on each day's new bars. |
| Macro/on-chain at 1m | **Forward-fill** | Causally ffill the latest real daily value onto 1m bars; group disabled when no real feed exists. |

### Round 5 — scope & realism
| Decision | Choice |
|---|---|
| Feature set at 1m | **Full** expanded set (AI decides what matters) |
| Context coins | **ETH, SOL, BNB, DOGE** (all four, 1m data fetched for each) |
| Holding cost | **Include perp funding** (~8h recurring) in paper reward/PnL for realism |
| Live chart source | **Real-time WebSocket stream** (true live updating chart) |
| Frontend(s) | **Both website + Telegram** (thin output adapters over the same JSON engine) |
| Max hold time | **Configurable cap, default ~1 day** safety net; AI may exit earlier |

### Round 6 — evaluation, UX, feedback loop
| Decision | Choice |
|---|---|
| Success bar | **Beat BOTH buy-and-hold AND the baseline policy, net of all costs, on an untouched final holdout** |
| Chart content | Prediction cone **+ bot trade markers + current position + running equity/PnL** |
| Telegram alerts | **Yes** — push trade events (open/close + result) and safety events (kill-switch) |
| Delivery | **All at once** (complete system together) |
| State persistence | **Persist across restarts** (resume open position/equity/history) |
| Action logging → feedback loop | **Log every bot action** (observation, prediction, decision, fill, outcome) as structured data that **future retraining consumes to improve both predictor and policy** — live experience becomes better training data over time |

**Implies a fresh train** (multi-head output incompatible with current single-head checkpoint).

**Out of scope (v1):** real-money order routing/exchange keys, bundled web frontend (JSON/query
APIs only), multi-symbol (BTC/USDT only), GBM multi-horizon (NN only).

---

## The hard constraint (why 1m)

A horizon cannot be shorter than one base candle, so 1m/5m/10m require a **1m base**. Honesty
caveat: the short heads (1m/5m) will correctly hug `P(up)≈0.5` / `exp_return≈0`; usable edge
concentrates in 30m/1hr. Per-horizon calibration + coverage surface this; the auto-drop floor
removes unreliable heads from the policy's effective signal.

---

## Design principles (binding)

| Principle | Rule |
|---|---|
| **Causality** | Targets use `close.shift(-h)` for labels only; purge/embargo = `max(horizons)`; no future leak into any head. |
| **Authorized boundary crossing** | The RL policy couples prediction→execution. This is allowed only per ADR 0008; the supervised predictor remains independently trainable/usable and the policy is isolated in `execution/policy/`. Tests required in **both** layers. |
| **One artifact** | Chart and policy read the same `MultiHorizonPredictionResult`, produced once in `RuntimeService`. |
| **Honest uncertainty** | Quantile bands, per-horizon isotonic calibration, coverage tracked. |
| **Learning inside guardrails** | RL learns size/timing **within** hard caps + kill-switch; caps are config, not learned. |
| **Config-driven** | Horizons, quantiles, reward weights, caps, trade-frequency, retrain cadence all in Pydantic + YAML. |
| **Open weights** | Both the predictor bundle and the RL policy are plain publishable files; no encryption/DRM (`open-weights.mdc`). |

---

## Canonical contract

`MultiHorizonPredictionResult` (`epoch_ai/services/types.py`), per "now" bar `T`:

```jsonc
{
  "as_of": "2026-06-29T14:08:00Z", "last_close": 61234.5, "model_version": "v_0042",
  "horizons": [
    { "label": "1m", "horizon": 1, "target_time": "2026-06-29T14:09:00Z",
      "p_up": 0.51, "exp_return": 0.0002,
      "price_p50": 61247.0, "price_p10": 61190.0, "price_p90": 61305.0,
      "confidence": 0.42, "reliable": false }   // reliable = passes coverage/Brier floor
    // ... 5m, 10m, 15m, 30m, 1hr
  ]
}
```
- `target_time = T + h×1m`; `price_pX = last_close × exp(return_pX)`.
- `confidence` = calibration reliability × inverse band-width ∈ [0,1].
- `.to_json()` is the single source of truth for chart + policy observations.

---

## Trading architecture (hybrid)

```
features ─┬─► multi-head evolved_nn ─► MultiHorizonPredictionResult ─┬─► chart (cone + history)
          │                                                          │
          └──────────────────────────────────────────────► RL policy (obs = preds+features+pos)
                                                              │  PPO, reward = risk-adj net fees
                                                              ▼
                                          hard caps + kill-switch ─► RiskManager ─► paper fills
```
- A **baseline heuristic policy** (confidence-weighted ensemble agreement over all six
  horizons, auto-dropping unreliable heads) is implemented first as a comparison benchmark and
  a safe fallback; the **learned PPO policy** must beat it OOS before promotion.

---

## Acceptance criteria (end-to-end, on an untouched final holdout)

The last slice of history is reserved and seen by **neither** the predictor **nor** the policy
during training. The system is "working" only if, on that holdout, net of taker fees + slippage
+ funding:

1. **Predictor:** per-horizon calibration sane (reliable heads pass the coverage/Brier floor);
   confidence-weighted Brier beats a no-skill baseline.
2. **Policy:** risk-adjusted return **beats both** (a) buy-and-hold BTC and (b) the
   confidence-weighted baseline policy.
3. **Safety:** hard caps + drawdown kill-switch + max-hold never breached in evaluation.

## Baked-in design notes (no decision needed)

- **OOS-only policy training:** the RL policy consumes the predictor's **out-of-sample**
  forecasts (those made on data the predictor had not trained on at that step). No in-sample
  predictions leak into policy training — otherwise the learned behavior won't transfer live.
- **Live robustness:** WebSocket auto-reconnect/backfill, missing-candle handling, and a
  heartbeat/health log for the running paper bot.
- **State persistence:** open position, equity, and history saved to disk and resumed on restart.
- **Feedback loop:** every action (observation snapshot, per-horizon prediction, decision, fill,
  realized outcome) is logged in a structured, replayable form; the retrain job can consume this
  live-experience log to improve subsequent predictor + policy versions.
- **Disclaimer & seam:** keep/strengthen the research-only, paper-only, not-financial-advice
  notice; leave a clean (unimplemented) seam for real-money execution; public market data needs
  no API keys (secrets stay out of committed files per `config-driven.mdc`).
- **RL dependency:** add the RL/PPO library to `requirements-optional.txt`, lazy-imported like
  other heavy deps; core predict/chart paths must run without it.

## File map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `config/config.yaml` | `timeframe: "1m"`; `prediction.horizons/quantiles`; grown warmup; `trading.*`; `rl.*` |
| Modify | `epoch_ai/config/settings.py` | `PredictionConfig` (horizons/labels/quantiles + validators); `TradingConfig`; `RLConfig` |
| Modify | `epoch_ai/features/pipeline.py` | `build_target` → multi-horizon DataFrame |
| Modify | `epoch_ai/models/base.py` | `predict` → `(n_rows, n_heads)`; add `predict_structured` |
| Modify | `epoch_ai/models/nn_genome.py` | Multi-head output width |
| Modify | `epoch_ai/models/nn_trainer.py` | Pinball (quantile) + BCE (direction) multi-head loss; coverage hooks |
| Modify | `epoch_ai/models/evolved_nn_model.py` | Multi-head fit/predict; importance gated to final fit |
| Modify | `epoch_ai/models/calibration.py` | Per-horizon isotonic calibrators (list sidecar) |
| Modify | `epoch_ai/models/registry.py` | Persist `horizons/quantiles/n_outputs` |
| Modify | `epoch_ai/learning/checkpoint.py` | Fingerprint includes horizons/quantiles |
| Modify | `epoch_ai/learning/step_metrics.py` | Per-horizon Brier/AUC/logloss + pinball + coverage |
| Modify | `epoch_ai/learning/progressive.py` | Per-horizon OOS; purge at `max(horizons)` |
| Modify | `epoch_ai/learning/promotion.py` | Confidence-weighted Brier promotion gate |
| Modify | `epoch_ai/logging_system/store.py` | One row per (timestamp, horizon); store band + p_up |
| Modify | `epoch_ai/services/types.py` | `HorizonForecast` + `MultiHorizonPredictionResult` + `to_json` |
| Modify | `epoch_ai/services/runtime.py` | Assemble structured result; price-band math |
| Create | `epoch_ai/services/forecast_api.py` | Live JSON + historical (store) chart payloads |
| Create | `epoch_ai/execution/policy/baseline.py` | Confidence-weighted ensemble baseline policy |
| Create | `epoch_ai/execution/policy/env.py` | Trading environment (obs, action, reward, fees, caps) |
| Create | `epoch_ai/execution/policy/ppo_policy.py` | PPO agent: train / save / load / act |
| Create | `epoch_ai/execution/policy/guardrails.py` | Hard caps + max-drawdown kill-switch + max-hold |
| Create | `epoch_ai/execution/session_state.py` | Persist/resume open position, equity, history across restarts |
| Create | `epoch_ai/execution/action_log.py` | Structured action/outcome log (feedback-loop training data) |
| Create | `epoch_ai/interfaces/telegram.py` | Telegram adapter: chart-on-request + trade/safety push alerts |
| Create | `epoch_ai/interfaces/web.py` | Web/dashboard JSON adapter (predictions + trades + equity) |
| Modify | `epoch_ai/execution/risk.py` | Accept policy action; enforce caps |
| Modify | `epoch_ai/execution/live_engine.py`, `live_loop.py` | Feed structured prediction → policy → fills |
| Modify | `epoch_ai/cli.py` | `predict` (per-horizon table/JSON); `train-policy`; `run` uses policy |
| Create | `docs/adr/0008-multi-horizon-and-learned-policy.md` | ADR: multi-head + quantile + RL boundary crossing |
| Modify | `tests/*` | See per-phase Verify rows below |
| Create | `tests/test_forecast_api.py`, `tests/test_policy_env.py`, `tests/test_ppo_policy.py` | New coverage |
| Modify | `README.md`, `AGENTS.md`, `requirements*.txt` | Docs + add `torch`/PPO deps (lazy where possible) |

---

## Phase 0 — 1m data + training budget

- [ ] `config.yaml`: `timeframe: "1m"`; grow `walk_forward.initial_train_period` to the largest feasible window (`> max(horizons)=60` by a wide margin); document expected runtime.
- [ ] Confirm `HistoricalDownloader` fetches real Binance 1m via CCXT locally for BTC **and all four context coins (ETH, SOL, BNB, DOGE)**; verify pagination + caching for **maximum** history (multi-year parquet × 5 symbols — check disk budget).
- [ ] Keep the **full** expanded feature set active at 1m (let the model select); macro/on-chain forward-filled, disabled when no real feed.
- [ ] Verify synthetic 1m generation (tests/CI fallback).
- [ ] **Initial run = full progressive walk-forward over all history** (accept long runtime on CUDA). Provide `--max-steps`/`--bars` only for smokes, not the real initial train.
- [ ] After initial train, switch ongoing training to **coarse scheduled auto-retrain** (larger `step_size`, `retrain_frequency`).

**Verify:** `python -m epoch_ai info`; `download` smoke; document real-data + runtime expectations in README/AGENTS.

## Phase 1 — Multi-horizon target

- [ ] `PredictionConfig`: `horizons=[1,5,10,15,30,60]`, `horizon_labels`, `quantiles=[0.1,0.5,0.9]`; keep scalar `horizon` (primary) for back-compat. Validators (quantiles sorted in (0,1) incl. 0.5; `initial_train_period > max(horizons)`; per-horizon purge/embargo).
- [ ] `build_target` → DataFrame: per `h`, `ret_h = log(close.shift(-h)/close)` + `up_h`. Last `h` rows NaN per column.

**Verify:** `pytest tests/test_features.py tests/test_config.py` — shapes, NaN tails, no leakage.

## Phase 2 — Multi-head `evolved_nn`

- [ ] Output width = `len(horizons) × (len(quantiles)+1)`: per horizon, quantile outputs + 1 direction logit.
- [ ] `nn_trainer`: loss = mean over heads of `pinball(quantiles) + BCE(direction)`; enforce monotone quantiles (sort / soft penalty); preserve mixed precision / compile / warm-start.
- [ ] `BaseModel.predict` → `(n_rows, n_heads)`; add `predict_structured`. GBM stays single-head.
- [ ] Registry/checkpoint persist `horizons/quantiles/n_outputs`.

**Verify:** `pytest tests/test_model.py -m slow` — multi-head shape + save/load (`atol=1e-5`).

## Phase 3 — Calibration, metrics, promotion

- [ ] Per-horizon isotonic calibrators (list sidecar travels with bundle); quantile coverage check (p10–p90 ≈ 0.8).
- [ ] `step_metrics` per-horizon Brier/AUC/logloss + pinball + coverage; `progressive` per-horizon OOS, purge at `max(horizons)`.
- [ ] `promotion.py`: champion gate = **confidence-weighted average per-horizon Brier**.

**Verify:** `pytest tests/test_progressive.py tests/test_calibration.py tests/test_promotion.py`.

## Phase 4 — Artifact, chart, baseline policy

- [ ] `services/types.py`: `HorizonForecast` + `MultiHorizonPredictionResult` (+ `to_json`, `reliable` flag from floor).
- [ ] `RuntimeService.predict_market`: `predict_structured` → per-horizon calibration → `price_pX = last_close·exp(return_pX)` → absolute `target_time` + `confidence`.
- [ ] `logging_system/store.py`: one row per (timestamp, horizon) using existing `horizon` column; persist p_up + band; resolve outcomes per horizon.
- [ ] `services/forecast_api.py`: **live** payload (forward cone, fed by the **real-time WebSocket** stream via `RuntimeService.run_live_stream`) + **historical** payload (predicted-vs-realized + reliability) from the store, **plus the bot's trade markers, current position, and running equity/PnL**. JSON only; thin **website (`interfaces/web.py`) + Telegram (`interfaces/telegram.py`)** adapters render it / push alerts (no charting code in the engine).
- [ ] `execution/policy/baseline.py`: confidence-weighted ensemble over all six horizons, auto-dropping heads below the reliability floor; configurable trade-frequency (default selective). This is the benchmark the RL policy must beat.
- [ ] `cli.py`: `predict` command (table + `--json`).

**Verify:** `pytest tests/test_services.py tests/test_forecast_api.py`; `python -m epoch_ai predict --json` smoke.

## Phase 5 — Learned RL trading policy (boundary crossing)

- [ ] `RLConfig` + `TradingConfig` in `settings.py` + `config.yaml`: reward weights, hard caps (max position fraction, **leverage cap default 1x**, max-drawdown kill-switch 20%), action bounds, **taker fee/slippage + perp funding**, **max-hold cap (~1 day)**, **$10k paper equity**, trade-frequency default (selective), decision-horizon set, **moderate reliability floor**.
- [ ] `execution/policy/guardrails.py`: enforce hard caps + drawdown kill-switch on any action (learned or baseline).
- [ ] `execution/policy/env.py`: trading environment.
  - **Observation:** per-horizon p_up/return/bands (reliable only) + raw feature row + current position/PnL.
  - **Action:** direction (long/short/flat) + continuous size (∈ caps) + exit.
  - **Reward:** risk-adjusted return (Sharpe-like) net of **taker fees + slippage + perp funding (held positions)**; drawdown penalty.
  - **Max-hold cap** (configurable, default ~1 day): force-close positions exceeding it; AI may exit earlier.
  - Causal replay over historical 1m bars; respects walk-forward (policy trains only on data the predictor was allowed to see).
- [ ] `execution/policy/ppo_policy.py`: PPO (torch, CUDA) — train / save / load (open-weights file) / `act(obs)`.
- [ ] Integrate into `RiskManager` / `live_engine` / `live_loop`: prediction → policy action → guardrails → paper fill.
- [ ] `execution/session_state.py`: persist/resume open position, equity, history across restarts (auto-reconnect + missing-candle handling + heartbeat on the live path).
- [ ] `execution/action_log.py`: log every (observation, per-horizon prediction, decision, fill, realized outcome) in replayable form for the feedback loop.
- [ ] `interfaces/telegram.py`: push trade events (open/close + result) and safety events (kill-switch); chart-on-request.
- [ ] `cli.py`: `train-policy` (train PPO on **out-of-sample** replay) and `run` uses learned policy with baseline fallback.

**Verify:** `pytest tests/test_policy_env.py tests/test_ppo_policy.py tests/test_risk.py tests/test_paper_trader.py`. Confirm: caps never breached; policy evaluated OOS vs baseline; paper-only (no live keys).

## Phase 6 — Adaptation

- [ ] Scheduled **auto-retrain + challenger/champion promotion** for the **predictor** (conf-weighted Brier gate) and the **policy** (OOS risk-adjusted return gate vs current champion + baseline + buy-and-hold).
- [ ] Coarse cadence config (post-initial): larger `step_size`/`retrain_frequency`; daily default.
- [ ] **Feedback loop:** retrain job consumes the `action_log` live-experience data so each new version learns from real bot behavior/outcomes.
- [ ] Reserve and **never train on** a final holdout slice; run the Acceptance-criteria evaluation on it.

**Verify:** `pytest tests/test_retrain_job.py tests/test_promotion.py`; auto-retrain smoke.

## Phase 7 — Docs & definition-of-done

- [ ] `docs/adr/0008-multi-horizon-and-learned-policy.md`: multi-head + quantile + **RL boundary-crossing rationale** and isolation.
- [ ] `README.md` / `AGENTS.md`: multi-horizon `predict`, chart use case, learned-policy bot, 1m base + runtime, fresh-train + paper-only notes, new deps; **strengthen the research-only / paper-only / not-financial-advice disclaimer** and document the real-money seam as intentionally unimplemented.
- [ ] Full gate: `.venv/Scripts/ruff.exe check .` clean; `.venv/Scripts/python.exe -m pytest` green (report counts); `python -m epoch_ai info`; predict + policy smokes. Document blocked items.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Max 1m history × evolution × full progressive = days of compute | Initial full run accepted; use CUDA; coarse auto-retrain afterward; smokes use `--bars`/`--max-steps`. |
| RL instability / overfitting on noisy 1m | Hybrid (predictor stays standalone); baseline benchmark must be beaten OOS; hard caps + kill-switch; risk-adjusted reward; walk-forward-respecting replay. |
| Boundary creep | RL isolated in `execution/policy/`; predictor independently usable; ADR 0008; tests both layers. |
| Short heads near-random | Calibration + coverage + auto-drop reliability floor. |
| Quantile crossing | Sort outputs / monotonicity penalty. |
| Guardrail bypass | All actions pass through `guardrails.py`; tests assert caps never breached. |
| Checkpoint incompatibility | Fresh train; fingerprint includes horizons/quantiles. |
| Feedback loop amplifies its own bias | Retrain on action-log uses *realized outcomes* (not the bot's hopes); keep market-outcome labels primary; gate promotions on the untouched holdout. |
| All-at-once delivery = long lead time before anything usable | Build predictor+chart paths first internally so they're verifiable early even if delivered together; phase verification matrix gates each layer. |

## Verification matrix

| Phase | Commands |
|---|---|
| 0 | `info`; `download` smoke |
| 1 | `pytest tests/test_features.py tests/test_config.py` |
| 2 | `pytest tests/test_model.py -m slow` |
| 3 | `pytest tests/test_progressive.py tests/test_calibration.py tests/test_promotion.py` |
| 4 | `pytest tests/test_services.py tests/test_forecast_api.py`; `predict --json` |
| 5 | `pytest tests/test_policy_env.py tests/test_ppo_policy.py tests/test_risk.py tests/test_paper_trader.py` |
| 6 | `pytest tests/test_retrain_job.py tests/test_promotion.py` |
| 7 | `ruff check .`; full `pytest`; `info`; predict + policy smokes |
