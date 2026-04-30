"""PPO-style RLHF.

Reference: Stiennon et al. "Learning to Summarize with Human Feedback" (2020),
Ouyang et al. "InstructGPT" (2022). The standard RLHF pipeline:

  1. Sample completions from the current policy.
  2. Score each with a reward model (callable here for flexibility).
  3. Add a per-token KL penalty against a frozen reference policy.
  4. Estimate per-token advantages with GAE on top of a value head.
  5. Update the policy with a PPO clipped surrogate (multiple inner epochs
     over the same rollouts, with old log-probs frozen).

This module provides:
  * ``ValueHead`` — a tiny linear head on top of the policy's last hidden
    state. Constructed alongside the policy at trainer creation time.
  * ``PPOConfig`` — hyperparameters.
  * ``PPOTrainer`` — rollout + multi-epoch PPO update. Inherits from
    ``Trainer``.
  * ``compute_ppo_loss`` and ``compute_gae`` — pure functions, unit-tested.

The reward function is a user-provided callable
``reward_fn(prompt: str, response: str) -> float`` — for full RLHF,
substitute a reward model (binary classifier trained on preference data).
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..trainer import Trainer, TrainerConfig, _NullCtx
from .grpo import _sample_completion, _selected_logprobs_full, _per_token_kl


RewardFn = Callable[[str, str], float]


# --------------------------------------------------------------------------- #
# Value head
# --------------------------------------------------------------------------- #


class ValueHead(nn.Module):
    """Scalar value head. Maps last-hidden-state → V(s).

    The head is shared across all ranks/positions; its only parameter is a
    tiny ``Linear(hidden, 1)``. It is *not* attached to the policy module
    so that ``policy.save_pretrained`` keeps producing a valid LM checkpoint.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states).squeeze(-1)  # [B, T]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class PPOConfig:
    rollouts_per_step: int = 4         # how many prompts to roll out per outer step
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0
    clip_eps: float = 0.2              # PPO clip
    value_clip_eps: float = 0.2        # value-loss clip (set <=0 to disable)
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 1.0                 # discount; 1.0 is standard for token-level RLHF
    gae_lambda: float = 0.95
    kl_coef: float = 0.05              # per-token KL penalty added to reward
    inner_epochs: int = 2              # PPO update epochs per rollout batch
    pad_token_id: int = 0


# --------------------------------------------------------------------------- #
# GAE
# --------------------------------------------------------------------------- #


def compute_gae(
    rewards: torch.Tensor,           # [B, T]
    values: torch.Tensor,            # [B, T+1] (includes bootstrap value at T)
    response_mask: torch.Tensor,     # [B, T]
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation. Returns (advantages, returns).

    ``advantages[b, t]`` and ``returns[b, t]`` are zero outside the response
    region (per ``response_mask``).
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        next_v = values[:, t + 1]
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae
    returns = advantages + values[:, :-1]
    advantages = advantages * response_mask
    returns = returns * response_mask
    return advantages, returns


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


def compute_ppo_loss(
    log_pi: torch.Tensor,           # [B, T]
    log_pi_old: torch.Tensor,       # [B, T] (frozen at rollout time)
    entropy: torch.Tensor,          # [B, T]
    values: torch.Tensor,           # [B, T] (current value-head output)
    old_values: torch.Tensor,       # [B, T] (frozen at rollout time)
    advantages: torch.Tensor,       # [B, T]
    returns: torch.Tensor,          # [B, T]
    response_mask: torch.Tensor,    # [B, T]
    clip_eps: float = 0.2,
    value_clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """PPO loss = -policy_surrogate + vf_coef * value_loss - entropy_coef * entropy."""
    ratio = torch.exp(log_pi - log_pi_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    pg_loss = -torch.minimum(surr1, surr2)

    if value_clip_eps > 0:
        v_clipped = old_values + torch.clamp(values - old_values, -value_clip_eps, value_clip_eps)
        vf1 = (values - returns) ** 2
        vf2 = (v_clipped - returns) ** 2
        v_loss = 0.5 * torch.maximum(vf1, vf2)
    else:
        v_loss = 0.5 * (values - returns) ** 2

    masked = (pg_loss + vf_coef * v_loss - entropy_coef * entropy) * response_mask
    n = response_mask.sum().clamp(min=1.0)
    loss = masked.sum() / n

    metrics = {
        "ppo/ratio_mean": (ratio * response_mask).sum().detach() / n,
        "ppo/pg_loss": (pg_loss * response_mask).sum().detach() / n,
        "ppo/v_loss": (v_loss * response_mask).sum().detach() / n,
        "ppo/entropy": (entropy * response_mask).sum().detach() / n,
    }
    return loss, metrics


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


class PPOTrainer(Trainer):
    """PPO-RLHF trainer.

    Each call to ``compute_loss(batch)`` runs:
      1. Sampling rollout (frozen policy snapshot).
      2. Reward + KL-shaped per-token rewards.
      3. GAE → advantages, returns.
      4. ``inner_epochs`` of PPO loss minimization on the rollout buffer.

    Returns the *mean* of the inner-epoch losses so the outer scheduler /
    logger sees a single scalar.
    """

    def __init__(
        self,
        model: nn.Module,
        prompt_dataloader: DataLoader,
        tokenizer,
        reward_fn: RewardFn,
        config: Optional[TrainerConfig] = None,
        ppo_config: Optional[PPOConfig] = None,
        ref_model: Optional[nn.Module] = None,
        value_head: Optional[ValueHead] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(model, prompt_dataloader, config, eval_dataloader=None, device=device)
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.ppo_config = ppo_config or PPOConfig()
        if self.ppo_config.pad_token_id == 0 and tokenizer.pad_token_id is not None:
            self.ppo_config.pad_token_id = tokenizer.pad_token_id

        underlying = self._unwrapped_model()
        hidden_size = getattr(getattr(underlying, "config", None), "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden_size from model.config.")

        if value_head is None:
            value_head = ValueHead(hidden_size)
        value_head.to(self.device)
        self.value_head = value_head
        # Add value-head params to the optimizer the base class built. The
        # scheduler's LambdaLR snapshots the param-group count at build time,
        # so we rebuild it here with the now-extended optimizer.
        self.optimizer.add_param_group({"params": list(self.value_head.parameters())})
        from ..optimization import build_scheduler
        self.scheduler = build_scheduler(self.optimizer, self.config.scheduler)

        if ref_model is None:
            ref_model = copy.deepcopy(underlying)
        ref_model.to(self.device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        self.ref_model = ref_model

    # ------------------------------------------------------------------ #
    # Forward helpers
    # ------------------------------------------------------------------ #

    def _policy_forward(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the policy and return (log_probs_of_target, entropy, value_per_token).

        All shapes are [B, T-1] (shifted to next-token prediction).
        """
        out = self.model(input_ids=sequence, output_hidden_states=True)
        logits = out.logits[:, :-1, :]
        targets = sequence[:, 1:]
        log_probs_full = F.log_softmax(logits, dim=-1)
        log_probs = log_probs_full.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        # Per-position entropy of the predicted distribution.
        probs = log_probs_full.exp()
        entropy = -(probs * log_probs_full).sum(dim=-1)
        # Value head reads the hidden states of the *prefix*, aligned to predictions.
        last_hidden = out.hidden_states[-1][:, :-1, :]
        values = self.value_head(last_hidden)
        return log_probs, entropy, values

    @torch.no_grad()
    def _ref_logprobs(self, sequence: torch.Tensor) -> torch.Tensor:
        return _selected_logprobs_full(self.ref_model, sequence, response_start=0)

    # ------------------------------------------------------------------ #
    # One rollout-and-update step
    # ------------------------------------------------------------------ #

    def _build_rollout(self, batch: Any) -> Dict[str, torch.Tensor]:
        cfg = self.ppo_config
        prompts: List[str] = batch["prompts"] if isinstance(batch, dict) else list(batch)
        device = self.device
        eos = self.tokenizer.eos_token_id
        pad = cfg.pad_token_id

        # ------------------------------------------------------------------ #
        # 1) Rollout: sample completions and compute scalar rewards.
        # ------------------------------------------------------------------ #
        underlying = self._unwrapped_model()
        underlying.eval()

        seqs: List[torch.Tensor] = []
        prompt_lens: List[int] = []
        scalar_rewards: List[float] = []
        for prompt in prompts:
            ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            P = ids.size(1)
            for _ in range(cfg.rollouts_per_step):
                seq, _samp_lp = _sample_completion(
                    underlying, ids,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    eos_token_id=eos,
                    pad_token_id=pad,
                )
                response = self.tokenizer.decode(seq[0, P:], skip_special_tokens=True)
                scalar_rewards.append(float(self.reward_fn(prompt, response)))
                seqs.append(seq[0])
                prompt_lens.append(P)

        # Pad to a single batch.
        max_len = max(s.size(0) for s in seqs)
        B = len(seqs)
        seq_pad = torch.full((B, max_len), pad, dtype=torch.long, device=device)
        response_mask = torch.zeros((B, max_len - 1), device=device)
        for b, (s, p_len) in enumerate(zip(seqs, prompt_lens)):
            T = s.size(0)
            seq_pad[b, :T] = s
            start = max(p_len - 1, 0)
            end = T - 1
            response_mask[b, start:end] = 1.0

        # ------------------------------------------------------------------ #
        # 2) Frozen old log-probs, ref log-probs, old values.
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            old_log_pi, _old_entropy, old_values = self._policy_forward(seq_pad)
            ref_log_pi = self._ref_logprobs(seq_pad)

        # Per-token KL-shaped reward: 0 everywhere except on the LAST response
        # token of each sample, which receives the scalar reward; per-token
        # KL is subtracted at every response position.
        rewards = torch.zeros((B, max_len - 1), device=device)
        kl = (old_log_pi - ref_log_pi)
        rewards = rewards - cfg.kl_coef * kl  # KL penalty everywhere
        for b, (s, p_len) in enumerate(zip(seqs, prompt_lens)):
            last = s.size(0) - 2  # last response position in shifted axis
            if last >= 0:
                rewards[b, last] = rewards[b, last] + scalar_rewards[b]
        rewards = rewards * response_mask

        # ------------------------------------------------------------------ #
        # 3) GAE on the bootstrapped value estimates.
        # ------------------------------------------------------------------ #
        # Bootstrap value: append a 0 at T (terminal).
        values_for_gae = torch.cat([old_values, torch.zeros((B, 1), device=device)], dim=-1)
        advantages, returns = compute_gae(
            rewards, values_for_gae, response_mask,
            gamma=cfg.gamma, lam=cfg.gae_lambda,
        )
        # Normalize advantages over the response tokens for stability.
        adv_mean = (advantages * response_mask).sum() / response_mask.sum().clamp(min=1)
        adv_var = (((advantages - adv_mean) * response_mask) ** 2).sum() / response_mask.sum().clamp(min=1)
        advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8)
        advantages = advantages * response_mask

        return {
            "seq_pad": seq_pad,
            "old_log_pi": old_log_pi,
            "old_values": old_values,
            "advantages": advantages,
            "returns": returns,
            "response_mask": response_mask,
        }

    def _compute_loss_from_rollout(self, rollout: Dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.ppo_config
        log_pi, entropy, values = self._policy_forward(rollout["seq_pad"])
        loss, _m = compute_ppo_loss(
            log_pi=log_pi,
            log_pi_old=rollout["old_log_pi"],
            entropy=entropy,
            values=values,
            old_values=rollout["old_values"],
            advantages=rollout["advantages"],
            returns=rollout["returns"],
            response_mask=rollout["response_mask"],
            clip_eps=cfg.clip_eps,
            value_clip_eps=cfg.value_clip_eps,
            vf_coef=cfg.vf_coef,
            entropy_coef=cfg.entropy_coef,
        )
        return loss

    def compute_loss(self, batch: Any) -> torch.Tensor:
        rollout = self._build_rollout(batch)
        losses = [self._compute_loss_from_rollout(rollout) for _ in range(self.ppo_config.inner_epochs)]
        return torch.stack(losses).mean()

    def train(self, on_log: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        cfg = self.config
        if self.is_main_process:
            os.makedirs(cfg.output_dir, exist_ok=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        self.model.train()

        accum = cfg.gradient_accumulation_steps
        running_loss = 0.0
        t_start = time.time()
        microstep = 0
        stop_training = False

        for batch in self._iter_batches():
            rollout = self._build_rollout(batch)
            self._unwrapped_model().train()

            for _ in range(self.ppo_config.inner_epochs):
                ctx = torch.amp.autocast("cuda", dtype=self.amp_dtype) if self.use_amp else _NullCtx()
                with ctx:
                    loss = self._compute_loss_from_rollout(rollout) / accum
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss += loss.item() * accum
                microstep += 1

                if microstep % accum == 0:
                    self._optimizer_step(accum)

                    if cfg.log_every and self.global_step % cfg.log_every == 0:
                        avg = running_loss / (cfg.log_every * accum)
                        rec = {
                            "step": self.global_step,
                            "loss": avg,
                            "lr": self.scheduler.get_last_lr()[0],
                            "elapsed_s": time.time() - t_start,
                        }
                        if self.is_main_process:
                            if on_log:
                                on_log(rec)
                            else:
                                print(f"[step {rec['step']}] loss={rec['loss']:.4f} lr={rec['lr']:.2e}")
                        running_loss = 0.0

                    if cfg.save_every and self.global_step % cfg.save_every == 0:
                        self.save_checkpoint(os.path.join(cfg.output_dir, f"step_{self.global_step}"))

                    if cfg.max_steps > 0 and self.global_step >= cfg.max_steps:
                        stop_training = True
                        break

            if stop_training:
                break

        if microstep % accum != 0 and (cfg.max_steps <= 0 or self.global_step < cfg.max_steps):
            self._optimizer_step(microstep % accum)

        self.save_checkpoint(os.path.join(cfg.output_dir, "final"))
        return {"global_step": self.global_step}
