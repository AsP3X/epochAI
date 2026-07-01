"""Proximal policy optimization agent (open-weights PyTorch sidecar)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from epoch_ai.config.settings import RLConfig
from epoch_ai.execution.policy.env import TradingReplayEnv
from epoch_ai.utils.logging import get_logger

logger = get_logger(__name__)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Learned policy requires PyTorch. pip install torch"
        ) from exc
    return torch, nn


def _resolve_device(device_pref: str):
    torch, _ = _require_torch()
    if device_pref == "cpu":
        return torch.device("cpu")
    if device_pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(slots=True)
class TrainStats:
    """Summary metrics from a PPO training run."""

    updates: int
    mean_reward: float
    final_equity: float


class PPOPolicy:
    """Small actor-critic PPO for continuous target-weight actions."""

    def __init__(
        self,
        obs_dim: int,
        config: RLConfig,
        *,
        device=None,
    ) -> None:
        torch, nn = _require_torch()
        self.config = config
        self.obs_dim = obs_dim
        self.device = device or _resolve_device(config.device)

        layers: list = []
        last = obs_dim
        for width in config.hidden_sizes:
            layers.extend([nn.Linear(last, width), nn.Tanh()])
            last = width
        self.body = nn.Sequential(*layers).to(self.device)
        self.actor_mean = nn.Linear(last, 1).to(self.device)
        self.actor_logstd = nn.Parameter(torch.zeros(1, device=self.device))
        self.critic = nn.Linear(last, 1).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.body.parameters())
            + list(self.actor_mean.parameters())
            + [self.actor_logstd]
            + list(self.critic.parameters()),
            lr=config.learning_rate,
        )

    def _features(self, obs):
        torch, _ = _require_torch()
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.body(x)

    def act(self, obs: np.ndarray, *, deterministic: bool = False) -> float:
        """Sample or mean target weight in ``[-1, 1]`` (guardrails applied downstream)."""
        obs_arr = np.asarray(obs, dtype=np.float32).ravel()
        if obs_arr.shape[0] != self.obs_dim:
            raise ValueError(
                f"Observation length {obs_arr.shape[0]} != policy obs_dim {self.obs_dim}. "
                "The policy artifact may have been trained in a different observation mode "
                "(forecast vs embedding) or trunk width."
            )
        torch, _ = _require_torch()
        with torch.inference_mode():
            h = self._features(obs_arr)
            mean = torch.tanh(self.actor_mean(h)).squeeze(-1)
            if deterministic:
                action = mean
            else:
                std = self.actor_logstd.exp().expand_as(mean)
                action = torch.clamp(mean + std * torch.randn_like(mean), -1.0, 1.0)
            return float(action.cpu().numpy().reshape(-1)[0])

    def train(
        self,
        env: TradingReplayEnv,
        *,
        on_update: Callable[[int, TradingReplayEnv], None] | None = None,
    ) -> TrainStats:
        """Run PPO updates on a replay environment.

        Args:
            on_update: Optional callback invoked after each PPO update with
                ``(update_index, env)``. Used by joint trunk fine-tuning to refresh
                embeddings or run a supervised auxiliary step on the shared TCN.
        """
        torch, _ = _require_torch()
        cfg = self.config
        rewards: list[float] = []

        for update in range(cfg.total_updates):
            obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            obs = env.reset()
            for _ in range(cfg.rollout_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                h = self.body(obs_t.unsqueeze(0))
                mean = torch.tanh(self.actor_mean(h)).squeeze(-1)
                std = self.actor_logstd.exp()
                dist = torch.distributions.Normal(mean, std)
                raw = dist.sample()
                action = torch.clamp(raw, -1.0, 1.0)
                logp = dist.log_prob(raw).sum()
                value = self.critic(h).squeeze(-1)

                next_obs, reward, done, _ = env.step(float(action.cpu().numpy()))
                obs_buf.append(obs)
                act_buf.append(float(action.cpu().numpy()))
                logp_buf.append(float(logp.detach().cpu().numpy()))
                val_buf.append(float(value.detach().cpu().numpy()))
                rew_buf.append(reward)
                done_buf.append(float(done))
                rewards.append(reward)
                obs = next_obs if not done else env.reset()

            self._ppo_update(obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf)
            if on_update is not None:
                on_update(update, env)
            if (update + 1) % max(1, cfg.total_updates // 5) == 0:
                logger.info(
                    "PPO update %d/%d mean_rollout_reward=%.6f equity=%.2f",
                    update + 1,
                    cfg.total_updates,
                    float(np.mean(rew_buf)),
                    env.portfolio.equity,
                )

        return TrainStats(
            updates=cfg.total_updates,
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            final_equity=float(env.portfolio.equity),
        )

    def _ppo_update(self, obs, actions, old_logp, values, rewards, dones) -> None:
        torch, _ = _require_torch()
        cfg = self.config
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        returns = torch.zeros_like(rewards_t)
        adv = torch.zeros_like(rewards_t)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones_t[t]
            delta = rewards_t[t] + cfg.gamma * next_value * mask - values_t[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * mask * gae
            adv[t] = gae
            next_value = values_t[t]
            returns[t] = adv[t] + values_t[t]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        act_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        old_logp_t = torch.tensor(old_logp, dtype=torch.float32, device=self.device)
        ret_t = returns.detach()
        adv_t = adv.detach()

        for _ in range(cfg.train_epochs):
            h = self.body(obs_t)
            mean = torch.tanh(self.actor_mean(h)).squeeze(-1)
            std = self.actor_logstd.exp().expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            logp = dist.log_prob(act_t)
            ratio = torch.exp(logp - old_logp_t)
            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_t
            policy_loss = -torch.min(s1, s2).mean()
            value_loss = ((self.critic(h).squeeze(-1) - ret_t) ** 2).mean()
            loss = policy_loss + 0.5 * value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save(self, path: str | Path) -> None:
        """Persist open-weights policy sidecar."""
        torch, _ = _require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "obs_dim": self.obs_dim,
            "observation_mode": self.config.observation_mode,
            "config": self.config.model_dump(),
            "state_dict": {
                "body": self.body.state_dict(),
                "actor_mean": self.actor_mean.state_dict(),
                "actor_logstd": self.actor_logstd.detach().cpu(),
                "critic": self.critic.state_dict(),
            },
        }
        torch.save(payload, path)
        meta = path.with_suffix(".json")
        meta.write_text(
            json.dumps(
                {
                    "obs_dim": self.obs_dim,
                    "observation_mode": self.config.observation_mode,
                    "open_weights": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: RLConfig | None = None,
        *,
        expected_obs_dim: int | None = None,
    ) -> PPOPolicy:
        """Load a saved policy."""
        torch, _ = _require_torch()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = config or RLConfig.model_validate(payload.get("config") or {})
        obs_dim = int(payload["obs_dim"])
        if expected_obs_dim is not None and obs_dim != expected_obs_dim:
            saved_mode = payload.get("observation_mode", "unknown")
            raise ValueError(
                f"Policy at {path} has obs_dim={obs_dim} (saved observation_mode="
                f"{saved_mode!r}) but the runtime expects {expected_obs_dim} for "
                f"rl.observation_mode={cfg.observation_mode!r}. Retrain or fix config."
            )
        policy = cls(obs_dim, cfg)
        state = payload["state_dict"]
        policy.body.load_state_dict(state["body"])
        policy.actor_mean.load_state_dict(state["actor_mean"])
        policy.actor_logstd.data = state["actor_logstd"].to(policy.device)
        policy.critic.load_state_dict(state["critic"])
        return policy
