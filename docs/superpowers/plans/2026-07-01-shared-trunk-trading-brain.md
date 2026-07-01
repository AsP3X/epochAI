# Shared-Trunk Trading Brain — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Respect `ml-causality.mdc`, `prediction-execution-separation.mdc` (this plan deliberately crosses it — see ADR task), `config-driven.mdc`, `regression-testing.mdc`, `definition-of-done.mdc`, `open-weights.mdc`, `git-commits.mdc` (no commit/push without explicit permission).

**Goal:** Build one GPU-trainable "trading brain" that shares a single deep representation between a supervised multi-horizon price predictor and a reinforcement-learned trading policy, so the AI both forecasts price *and* discovers its own entry/exit/sizing strategies from simulated PnL.

> **Implementation status (2026-07-01):** Phases 1–11 code, tests, config, and docs are
> complete on `feature/shared-trunk-trading-brain`. **Pending GPU validation only:**
> Milestone Gate A (forecast-mode policy baseline) and Task 10 holdout gates (frozen
> embedding vs forecast; joint fine-tune vs Gate A without Brier/AUC regression).

**Architecture:** A shared **trunk** (the existing dilated causal TCN, scaled up) produces a per-bar embedding. Two consumers read that embedding: (1) supervised **prediction heads** (per-horizon direction logit + return quantiles) trained on actual future price — the dense, low-noise signal that keeps forecasts accurate and powers the chart; (2) a **policy/value head** trained by PPO on a **multi-bar** simulated-trading reward — the signal that teaches strategy. Both gradients flow into the trunk, so learning to trade reshapes the representation while direct price supervision keeps it honest. We reach this end-state through a staged on-ramp that de-risks the RL work.

**Tech Stack:** Python 3.12, PyTorch (CUDA), existing `MultiHeadModel`/`TCNModel`, `TradingReplayEnv`, PPO, Pydantic v2 config + YAML, pytest.

---

## Decision: why the shared-trunk brain (and not pure staged-hybrid or pure end-to-end)

The user asked to look into the future of each option and choose the most powerful *trainable* version.

| Option | Future ceiling | Trainability on GPU + our data | Verdict |
|---|---|---|---|
| **A — staged hybrid** (deep predictor → separate small RL policy on the predictor's *outputs*) | Medium. The policy only ever sees a 3-numbers-per-horizon summary of what the predictor chose to expose; anything the predictor discards for direction accuracy is lost to the trader forever. | Excellent. Dense supervised signal; small RL problem. | On-ramp, not end-state. |
| **B — pure end-to-end RL on PnL** (one net, single trading reward) | High in theory, but noisy PnL reward + non-stationary market makes it 10–100× data-hungry, unstable, un-auditable, and it loses the chart-forecast product. | Poor sample efficiency; hard to validate; wastes the working predictor. | Rejected as a starting point. |
| **A.5 — shared-trunk multi-task brain** (one trunk, supervised heads + policy head) | **Highest practical ceiling.** The policy head reads the *full learned embedding*, not a lossy summary, so no information bottleneck. The supervised heads act as a dense auxiliary task that stabilizes the noisy RL signal — the standard fix for exactly B's inefficiency. Trunk embedding is extensible to future heads (regime, volatility, execution quality, multi-asset). | **Best FLOP efficiency:** one expensive trunk shared by cheap heads instead of two big nets; scales cleanly with GPU (deeper/wider trunk, longer lookback, bigger batch). | **Chosen end-state.** |

A.5 is the disciplined realization of the user's "unified brain that predicts and trades, and the trading reinforces the model" instinct: the shared trunk *is* the reinforcement loop, but anchored by clean price supervision so it strengthens rather than degrades prediction.

**Strategy:** Build A first (Phases 1–5) to get a real, measurable trading learner on top of today's predictor, then merge into the shared-trunk brain (Phases 6–8). Nothing in A is wasted — it becomes the policy-training harness and the benchmark A.5 must beat.

---

## Target architecture

```
 All data (OHLCV, funding, OI, cross-asset ETH/SOL/BNB/DOGE, sentiment, on-chain)
                                   |
                     FeaturePipeline (causal features)
                                   |
                         +---------v----------+
                         |   SHARED TRUNK     |  dilated causal TCN (scaled up)
                         |  embedding e_t     |  e_t = trunk(window_{t-L+1..t})
                         +----+----------+----+
                              |          |
             supervised heads |          | policy/value head (PPO)
        per-horizon p_up +    |          | action: target weight in [-1,1]
        return quantiles      |          | reward: multi-bar risk-adj PnL net of costs
             (future price)   |          | (learns entry/exit/size/abstain)
                              |          |
                    chart + accuracy    trading decisions (paper now, live later)
```

- **Trunk:** `epoch_ai/models/tcn_model.py` `_TCNNet.network` already emits `last = h[:, :, -1]` (size = last channel count). We expose it as an embedding and scale capacity via config.
- **Supervised heads:** existing `MultiHeadSpec` linear head (unchanged semantics).
- **Policy/value head:** new module reading the trunk embedding + portfolio state.
- **Reward:** new multi-bar reward in `TradingReplayEnv`.

---

## File structure map

**Create:**
- `docs/adr/0009-shared-trunk-trading-brain.md` — records the deliberate prediction/execution merge, reward definition, promotion metric.
- `epoch_ai/execution/policy/trunk_policy.py` — policy+value head on a trunk embedding (A.5).
- `tests/test_reward_multibar.py` — multi-bar reward semantics.
- `tests/test_trunk_policy.py` — trunk-embedding policy head (A.5).

**Modify:**
- `epoch_ai/config/settings.py` — extend `RLConfig` (reward horizon, capacity, observation mode), `ModelConfig.tcn` (trunk capacity), add trunk-policy toggles.
- `config/config.yaml` — mirror every new field with documented defaults.
- `epoch_ai/execution/policy/env.py` — multi-bar reward; expose embedding-fed observation hook.
- `epoch_ai/execution/policy/observation.py` — multi-timescale observation (single + multi-bar heads) + optional embedding passthrough.
- `epoch_ai/execution/policy/ppo_policy.py` — larger configurable net; accept embedding observations.
- `epoch_ai/learning/policy_promotion.py` — train/eval PPO on **real** forecasts via `from_forecasts`; absolute-metric gate.
- `epoch_ai/cli.py` — `cmd_train_policy` uses `from_forecasts` + champion model; surface new flags.
- `epoch_ai/models/tcn_model.py` — `embed(x)` method returning trunk embedding; optional policy head wiring (A.5).
- `epoch_ai/learning/acceptance.py` — already scores real forecasts; extend policy line to score learned champion on the same env.

**Check (read before editing):**
- `epoch_ai/models/multi_head.py` (`MultiHeadSpec`, losses), `epoch_ai/models/factory.py` (`build_model`), `epoch_ai/models/registry.py` (champion pointer), `epoch_ai/services/types.py` (`build_horizon_forecast`, `horizon_confidence`).

---

## Phase 0 — ADR + alignment (no runtime code)

### Task 0: Author ADR 0009

**Files:**
- Create: `docs/adr/0009-shared-trunk-trading-brain.md`

- [ ] **Step 1: Write the ADR** capturing: (a) the learned policy becomes the primary trader with baseline/threshold demoted to fallback/benchmark; (b) reward = multi-bar risk-adjusted return net of fees + funding; (c) promotion metric = absolute OOS risk-adjusted return (baseline/buy-and-hold are report lines, not gates); (d) the shared trunk deliberately merges prediction and execution representations — supersedes ADR 0008's strict separation for the learned path, authorized by the repo owner; (e) paper-only for v1; open-weights (trunk + heads are plain files).

- [ ] **Step 2: Cross-link** ADR 0008 and `prediction-execution-separation.mdc` to point at 0009 for the learned path.

- [ ] **Step 3 (permission-gated): Commit** only if the user explicitly asks.

---

## Phase 1 — Train the policy on the REAL predictor (highest leverage, A)

Today `train-policy` and `auto_train_and_promote_policy` build the env with the price-only proxy (`from_market`). `TradingReplayEnv.from_forecasts` already exists (causal). Wire the real champion model into policy training.

### Task 1: Policy training consumes champion `predict_structured`

**Files:**
- Modify: `epoch_ai/learning/policy_promotion.py`
- Modify: `epoch_ai/cli.py` (`cmd_train_policy`)
- Test: `tests/test_ppo_policy.py`

- [ ] **Step 1: Write the failing test** — a champion multi-head stub yields structured forecasts; assert the training env is built via `from_forecasts` (env `model_version == "replay-real"`), not the proxy.

```python
def test_policy_trains_on_real_forecasts(tmp_path, small_config):
    # Build a tiny multi-head model, register it, run one policy train cycle,
    # and assert the env used real forecasts (structured_forecasts is not None).
    ...
    assert env.structured_forecasts is not None
    assert env.current_forecast().model_version == "replay-real"
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_ppo_policy.py::test_policy_trains_on_real_forecasts -v` → FAIL (still uses `from_market`).

- [ ] **Step 3: Implement** — in `policy_promotion.auto_train_and_promote_policy` and `cli.cmd_train_policy`, load the champion via `ModelRegistry`, compute `predict_structured` on the feature-aligned holdout/train close series, and build train/eval envs with `TradingReplayEnv.from_forecasts(config, close, structured, horizons)`. Fall back to `from_market` only when no multi-head champion exists.

- [ ] **Step 4: Run tests** — `pytest tests/test_ppo_policy.py -v` → PASS.

- [ ] **Step 5 (permission-gated): Commit.**

---

## Phase 2 — Multi-bar, cost-aware reward (fixes RL noise)

Bar-to-bar reward is noise-dominated (empirically: OOS AUC h1≈0.54 vs h12≈0.58). Reward decisions over a hold/decision horizon so the learning signal carries real movement.

### Task 2: Config for reward shaping

**Files:**
- Modify: `epoch_ai/config/settings.py` (`RLConfig`)
- Modify: `config/config.yaml` (`rl:`)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test** — assert new `RLConfig` fields exist with defaults and validation.

```python
def test_rl_reward_config_defaults():
    cfg = AppConfig().rl
    assert cfg.reward_horizon >= 1
    assert cfg.turnover_penalty >= 0.0
    assert cfg.reward_mode in {"per_bar", "multi_bar"}
```

- [ ] **Step 2: Run** `pytest tests/test_config.py::test_rl_reward_config_defaults -v` → FAIL.

- [ ] **Step 3: Implement** — add to `RLConfig`:

```python
reward_mode: Literal["per_bar", "multi_bar"] = Field(
    default="multi_bar",
    description="per_bar: single-bar PnL (noisy). multi_bar: PnL over reward_horizon.",
)
reward_horizon: int = Field(
    default=12, ge=1,
    description="Bars over which a decision's PnL is accumulated for the reward (multi_bar).",
)
turnover_penalty: float = Field(
    default=0.0, ge=0.0,
    description="Penalty per unit of |weight change| to discourage churn.",
)
```

Mirror in `config/config.yaml` under `rl:` with comments.

- [ ] **Step 4: Run** `pytest tests/test_config.py -v` and `python -m epoch_ai info` → PASS / resolves.

- [ ] **Step 5 (permission-gated): Commit.**

### Task 3: Multi-bar reward in the env

**Files:**
- Modify: `epoch_ai/execution/policy/env.py` (`step`)
- Test: `tests/test_reward_multibar.py`

- [ ] **Step 1: Write the failing test** — on a deterministic rising close series, a held long position's `multi_bar` reward over `reward_horizon` equals the accumulated risk-adjusted return minus turnover, and is less noisy than summed per-bar rewards (assert equality to a hand-computed value; assert turnover penalty reduces reward when weight flips every bar).

- [ ] **Step 2: Run** `pytest tests/test_reward_multibar.py -v` → FAIL.

- [ ] **Step 3: Implement** — in `TradingReplayEnv.step`, when `reward_mode == "multi_bar"`, accumulate equity change over the next `reward_horizon` bars for the reward attributed to the current action (respecting `done`), subtract `turnover_penalty * |weight - prev_weight|`, keep the existing `drawdown_penalty` term. Preserve causality: reward uses only bars at/after the action, never future info leaking into the *observation*. Keep `per_bar` as the legacy path.

- [ ] **Step 4: Run** `pytest tests/test_reward_multibar.py tests/test_policy_env.py -v` → PASS.

- [ ] **Step 5 (permission-gated): Commit.**

---

## Phase 3 — Multi-timescale observation + policy capacity

### Task 4: Multi-timescale observation

**Files:**
- Modify: `epoch_ai/execution/policy/observation.py`
- Test: `tests/test_policy_env.py`

- [ ] **Step 1: Write the failing test** — with `decision_horizons` including a fast head (1) and multi-bar heads (6, 12), assert the observation contains reliable per-horizon (p_up, exp_return, confidence) for each, plus the 4 portfolio scalars, and abstains (0.5, 0, 0) for unreliable heads.

- [ ] **Step 2: Run** → FAIL if defaults don't include the fast+multi mix.

- [ ] **Step 3: Implement** — keep `build_observation` shape logic; ensure `decision_horizons` default (config) spans single + multi-bar (e.g. `[1, 6, 12]`). No hidden-layer change here — this is representation *input*, not depth.

- [ ] **Step 4: Run** `pytest tests/test_policy_env.py -v` → PASS.

- [ ] **Step 5 (permission-gated): Commit.**

### Task 5: Configurable, larger PPO net

**Files:**
- Modify: `epoch_ai/config/settings.py` (`RLConfig.hidden_sizes` docs; GPU-aware caps), `config/config.yaml`
- Modify: `epoch_ai/execution/policy/ppo_policy.py`
- Test: `tests/test_ppo_policy.py`

- [ ] **Step 1: Write the failing test** — construct `PPOPolicy` with `hidden_sizes=[256,256,128]` and assert the body has the expected layer count; assert it trains one update on CPU without error.

- [ ] **Step 2: Run** → PASS/FAIL depending; if PPO already generic over `hidden_sizes`, the test mainly guards regressions and asserts larger default. Adjust default `hidden_sizes` + `total_updates`/`rollout_steps` to a serious (config-capped) budget; document weak-GPU downshift.

- [ ] **Step 3: Implement** any needed generalization (device/AMP, batch over rollouts) so bigger nets train on CUDA.

- [ ] **Step 4: Run** `pytest tests/test_ppo_policy.py -v` → PASS.

- [ ] **Step 5 (permission-gated): Commit.**

---

## Phase 4 — Make the learned policy the default trader

### Task 6: Default backend + learned-path abstention

**Files:**
- Modify: `config/config.yaml` (`trading.policy_backend`), `epoch_ai/execution/policy/executor.py`
- Test: `tests/test_policy_env.py` (or new `tests/test_executor_backend.py`)

- [ ] **Step 1: Write the failing test** — with a saved policy artifact present, `decide_trading_action` uses the learned path; with none, it falls back to baseline. On the learned path, the reliability floor / threshold dead band does **not** pre-filter (the policy learns when to abstain); guardrails (leverage cap, drawdown kill, max-hold) still clamp.

- [ ] **Step 2: Run** → FAIL (default is `baseline`; learned path still gated).

- [ ] **Step 3: Implement** — set default `trading.policy_backend: learned_with_baseline_fallback`; on the learned branch skip the reliability/threshold pre-filter but keep `apply_guardrails`.

- [ ] **Step 4: Run** targeted tests + `python -m epoch_ai info` → PASS.

- [ ] **Step 5 (permission-gated): Commit.**

---

## Phase 5 — Promote on absolute trading quality

### Task 7: Absolute-metric promotion gate

**Files:**
- Modify: `epoch_ai/learning/policy_promotion.py` (`decide_policy_promotion` usage), `config/config.yaml` (`rl.promotion`)
- Modify: `epoch_ai/learning/acceptance.py` (score learned champion via `from_forecasts`)
- Test: `tests/test_promotion.py` (or `tests/test_policy_promotion.py`)

- [ ] **Step 1: Write the failing test** — promotion requires challenger absolute `risk_adjusted_return` improvement over champion (baseline/buy-hold recorded but not required to gate); a challenger that beats baseline but not the champion is NOT promoted.

- [ ] **Step 2: Run** → FAIL if current gate couples to baseline.

- [ ] **Step 3: Implement** — make `require_beat_baseline`/`require_beat_buy_hold` default to report-only; keep champion-improvement gate. `evaluate_holdout` already builds real-forecast envs; extend its `policy_champion` line to score the learned policy on the same `from_forecasts` env.

- [ ] **Step 4: Run** `pytest tests/test_promotion.py -v` → PASS.

- [ ] **Step 5 (permission-gated): Commit.**

### Milestone gate A (must pass before Phase 6)
On the user's GPU box with a real registered champion:
- `python -m epoch_ai train --bars <large>` produces a champion (predictor unchanged).
- `python -m epoch_ai train-policy` trains PPO on **real** forecasts; `evaluate-holdout` shows the learned policy trading (non-flat) with a reported absolute risk-adjusted return.
- Record: does the learned policy beat baseline and buy-and-hold on the untouched holdout? This number is the bar A.5 must exceed.

---

## Phase 6 — Expose the trunk embedding (A.5 foundation)

### Task 8: `TCNModel.embed`

**Files:**
- Modify: `epoch_ai/models/tcn_model.py`
- Test: `tests/test_model.py` (or `tests/test_tcn.py` if present; else add)

- [ ] **Step 1: Write the failing test** — a trained/tiny TCN returns `embed(x)` of shape `(n_rows, trunk_dim)` where `trunk_dim == channels[-1]`; embeddings are deterministic and causal (row `i` unchanged by future rows).

- [ ] **Step 2: Run** → FAIL (`embed` undefined).

- [ ] **Step 3: Implement** — refactor `_TCNNet.forward` to expose the pre-head embedding (`last`), and add `TCNModel.embed(x)` that runs the trunk over causal windows (reuse `_forward_frame` windowing) and returns `last` without the linear head. Keep `predict`/`predict_structured` behavior identical (they still apply the head).

- [ ] **Step 4: Run** `pytest tests/test_model.py -v` → PASS.

- [ ] **Step 5 (permission-gated): Commit.**

---

## Phase 7 — Trunk-fed policy head (A.5 core, research-gated)

This is where the policy reads the **full embedding** instead of the 3-numbers-per-horizon summary. Deterministic scaffolding is TDD; the training quality is a research gate.

### Task 9: `trunk_policy.py` policy/value head

**Files:**
- Create: `epoch_ai/execution/policy/trunk_policy.py`
- Modify: `epoch_ai/execution/policy/observation.py` (embedding passthrough mode)
- Modify: `epoch_ai/config/settings.py` (`rl.observation_mode: {"forecast","embedding"}`), `config/config.yaml`
- Test: `tests/test_trunk_policy.py`

- [ ] **Step 1: Write the failing test** — build a policy head over `obs = concat(trunk_embedding, portfolio_scalars)`; assert action in `[-1, 1]`; assert one PPO update runs; assert `observation_mode="embedding"` produces the embedding-sized observation.

- [ ] **Step 2: Run** → FAIL (module absent).

- [ ] **Step 3: Implement** — a small actor-critic head (config-sized) consuming `trunk_dim + 4` inputs. Env observation builder gains an `embedding` mode that calls `model.embed` on the current causal window and concatenates portfolio state. Reuse PPO update math from `ppo_policy.py` (extract shared helper if cleaner).

- [ ] **Step 4: Run** `pytest tests/test_trunk_policy.py -v` → PASS.

- [ ] **Step 5 (permission-gated): Commit.**

### Task 10: Joint / staged trunk fine-tuning (research gate)

- [ ] **Step 1:** Start with a **frozen trunk** (predictor trained supervised; policy head learns on the frozen embedding). Verify it matches-or-beats Phase-5 forecast-observation policy on the holdout.
- [ ] **Step 2:** Enable **joint fine-tuning**: unfreeze the trunk, add the PPO policy-gradient term to the supervised multi-head loss with a config weight (`rl.policy_loss_weight`), keeping the supervised heads active so the dense signal anchors the representation. Train on GPU.
- [ ] **Step 3: Gate** — accept A.5 only if, on the untouched holdout (`evaluate-holdout`), the joint brain's absolute risk-adjusted return exceeds Milestone Gate A's number **without** degrading predictor Brier/AUC beyond a configured tolerance. If it degrades prediction, lower `policy_loss_weight` or keep the trunk frozen. Document results.

---

## Phase 8 — Capacity scaling for the powerful GPU

### Task 11: Trunk capacity config

**Files:**
- Modify: `config/config.yaml` (`model.tcn`), `epoch_ai/config/settings.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test** — deeper/wider TCN config validates (e.g. `channels: [128,128,256,256,512]`, `lookback: 192`) and `batch_size`/CUDA caps resolve.
- [ ] **Step 2: Run** → PASS/FAIL; add validation if missing.
- [ ] **Step 3: Implement** — document a GPU preset (deeper trunk, longer lookback, larger batch, mixed precision on) and a weak-GPU downshift, all via YAML/`--set`. No hard-coded constants.
- [ ] **Step 4:** `python -m epoch_ai info` resolves the preset.
- [ ] **Step 5 (permission-gated): Commit.**

> Capacity note: more depth/width/lookback improves the **trunk's** ability to draw connections from the many data points — this is where "more hidden layers" pays off (representation), distinct from the policy head size (strategy). Scale the trunk with data volume; keep regularization (dropout, weight decay, recency weighting, walk-forward) to avoid overfitting the noisier long horizons.

---

## Verification matrix (run per change type; evidence before "done")

| Changed | Commands |
|---|---|
| Config / YAML | `.venv/bin/python -m pytest tests/test_config.py`; `python -m epoch_ai info` |
| Env / reward | `pytest tests/test_reward_multibar.py tests/test_policy_env.py` |
| PPO / trunk policy | `pytest tests/test_ppo_policy.py tests/test_trunk_policy.py` |
| Promotion / acceptance | `pytest tests/test_promotion.py`; real-model `evaluate-holdout` on GPU box |
| Model (embed) | `pytest tests/test_model.py` |
| Any Python | `.venv/bin/ruff check .` |
| Cross-cutting | full `.venv/bin/python -m pytest` + backtest smoke `python -m epoch_ai backtest --bars 8000 --max-steps 12` |

## Definition of done (per `definition-of-done.mdc`)
1. Every phase implemented, tested, or explicitly blocked with reason.
2. `ruff` clean; `pytest` green (report counts).
3. Walk-forward causality preserved (reward uses only at/after-action bars; embeddings causal; embargo respected).
4. New tunables in Pydantic **and** `config/config.yaml`.
5. ADR 0009 recorded; README/AGENTS updated for new default backend and commands.
6. `evaluate-holdout` shows the learned/joint policy trading with an absolute risk-adjusted metric; A.5 gated on beating Milestone Gate A without degrading prediction.
7. No destructive artifact actions; no git writes without explicit permission.

## Risks & mitigations
- **RL instability / overfit noisy reward** → multi-bar reward, frozen-trunk first, supervised auxiliary anchor, absolute-metric gate, walk-forward holdout.
- **Prediction regresses when trunk fine-tunes on PnL** → `policy_loss_weight` tuning; fall back to frozen trunk; Brier/AUC tolerance gate.
- **Overlapping multi-bar labels inflate confidence** → embargo (already `= max_horizon`), report wider error bars on long horizons.
- **GPU OOM at higher capacity** → config caps (`cuda_batch_cap`, worker caps), mixed precision, downshift preset.
- **Boundary crossing** → ADR 0009 + tests in both prediction and policy layers.

## Execution handoff
Two execution options:
1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks.
2. **Inline Execution** — execute tasks in this session with checkpoints.
