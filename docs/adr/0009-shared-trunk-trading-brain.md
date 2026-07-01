# ADR 0009: Shared-Trunk Trading Brain

## Status

Accepted (2026-07-01) — implementation in progress via
`docs/superpowers/plans/2026-07-01-shared-trunk-trading-brain.md`. Supersedes the strict
prediction/execution isolation of ADR 0008 **for the learned trading path only**; the
supervised predictor remains independently trainable and the boundary still holds for the
`threshold`/`baseline` backends.

## Context

ADR 0008 established a multi-head predictor and a separate PPO policy that consumed the
predictor's compressed outputs (per-horizon P(up) + quantiles). The product goal is a
single "trading brain" that both forecasts price and learns its own entry/exit/size
strategy from simulated trading, trainable on a powerful GPU over all collected data.

Two limits of the ADR-0008 design block the most powerful version:

1. **Information bottleneck.** The policy only ever saw a 3-numbers-per-horizon summary of
   the forecast, so anything the predictor learned but did not expose was lost to the
   trader.
2. **Reward noise.** A single-bar PnL reward is noise-dominated, making direct RL
   data-hungry and unstable.

The repository owner evaluated three options (staged hybrid, pure end-to-end RL,
shared-trunk multi-task) and chose the **shared-trunk multi-task brain** as the highest
practical ceiling that is still GPU-trainable and auditable.

## Decision

- **Shared trunk.** The existing dilated causal TCN trunk (`epoch_ai/models/tcn_model.py`)
  emits a per-bar embedding. It feeds (a) the supervised multi-horizon heads (direction
  logit + return quantiles) and (b) a new **policy/value head**. Direct price supervision
  is retained as a dense auxiliary task that anchors the representation against RL noise.
- **Policy reads the embedding.** The learned policy consumes the full trunk embedding
  (plus portfolio state), not the compressed forecast summary — removing the bottleneck.
- **Multi-bar, cost-aware reward.** The replay env rewards a decision held over
  `rl.reward_horizon` bars, net of fees/funding and a turnover penalty
  (`rl.reward_mode="multi_bar"`), lowering reward variance while preserving causality.
- **Absolute-metric promotion.** Policies are promoted on an **absolute** risk-adjusted
  return floor (`promotion.min_absolute_metric`) plus champion improvement; baseline and
  buy-and-hold are **report-only** benchmarks, not gates.
- **Learned policy is the default trader** (`trading.policy_backend =
  learned_with_baseline_fallback`), with the baseline ensemble as fallback when no policy
  artifact exists. Hard guardrails (leverage cap, drawdown kill, max-hold) always clamp.
- **Staged training.** Phase A trains the policy on the frozen predictor's real forecasts.
  Phase A.5 unfreezes the trunk and adds the policy-gradient term to the supervised loss
  (`rl.policy_loss_weight`), gated so trunk fine-tuning may not degrade predictor
  Brier/AUC beyond a configured tolerance.
- **Paper-only.** Real-money order routing remains intentionally unimplemented.

## Boundary crossing (explicit)

ADR 0008 kept training labels out of `execution/policy/` and PnL out of `models/`. This
ADR **deliberately merges the representations**: the trunk is shaped by both supervised
price labels and the RL trading reward. The repository owner authorized this coupling for
the learned path. Mitigations that keep it honest:

| Risk | Mitigation |
| --- | --- |
| RL degrades prediction | Supervised heads stay active; `policy_loss_weight` tunable; frozen-trunk fallback; Brier/AUC tolerance gate |
| Look-ahead leakage | Observation strictly causal; reward uses only at/after-decision bars; walk-forward embargo preserved |
| Overfitting noisy PnL | Multi-bar reward, absolute-metric holdout gate, recency weighting |
| Un-auditable model | Trunk embedding + heads are plain open-weights files; chart forecasts retained |

The `threshold` and `baseline` backends remain fully within the ADR-0008 separation.

## Consequences

- Adds `TCNModel.embed` and `execution/policy/trunk_policy.py`; the policy observation
  gains an `embedding` mode.
- Promotion/acceptance metrics are now honest per-bar (Sharpe + max-drawdown rebuilt from
  the per-bar equity curve), so absolute numbers changed meaning vs pre-multi-bar reports.
- Existing PPO policies trained on the forecast-summary observation are incompatible with
  the embedding observation; retrain required.
- Open weights preserved: trunk, supervised heads, and policy/value head are plain files;
  no encryption or license gate is added.
- Real 1m microstructure data remains required for short-horizon edge; synthetic is
  plumbing-only.
