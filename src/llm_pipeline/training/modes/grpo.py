"""Group Relative Policy Optimization (GRPO).

Reference: Shao et al. "DeepSeekMath: Pushing the Limits of Mathematical
Reasoning in Open Language Models" (2024) — introduces GRPO.

GRPO is a critic-free policy-gradient method: for each prompt we sample a
group of ``G`` completions from the current policy, score them with a
reward function, and use the *group-relative* advantages

    A_i = (r_i - mean(r)) / (std(r) + eps)

as the policy-gradient signal. There is no value model; the group itself
provides the baseline. The objective is the PPO-clipped surrogate plus a
KL penalty against a frozen reference policy:

    L = -(1/G) Σ_i (1/|y_i|) Σ_t [
              min(ρ_t · A_i, clip(ρ_t, 1-ε, 1+ε) · A_i)  -  β · KL_t
        ]

where ρ_t = exp(log π(a_t|s_t) - log π_old(a_t|s_t)) and KL_t is the
unbiased per-token estimator KL = exp(log_ref - log_pi) - (log_ref - log_pi) - 1.

This module provides:
  * ``GRPOConfig`` — hyperparameters.
  * ``GRPOTrainer`` — sampling + loss + step. Inherits from ``Trainer``.
  * ``compute_grpo_loss`` — the loss alone, useful for unit tests.
  * Reward functions: pluggable callables of the form
    ``reward_fn(prompt: str, response: str) -> float``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..trainer import Trainer, TrainerConfig
from .sft import IGNORE_INDEX


RewardFn = Callable[[str, str], float]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class GRPOConfig:
    group_size: int = 4              # G — completions per prompt
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0                   # 0 disables top-k
    clip_eps: float = 0.2            # PPO clip epsilon
    beta_kl: float = 0.04            # KL coefficient (DeepSeek default 0.04)
    advantage_eps: float = 1e-4
    pad_token_id: int = 0


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


def _per_token_kl(log_pi: torch.Tensor, log_ref: torch.Tensor) -> torch.Tensor:
    """Unbiased per-token KL estimator: exp(d) - d - 1, where d = log_ref - log_pi."""
    d = log_ref - log_pi
    return torch.exp(d) - d - 1.0


def compute_grpo_loss(
    log_pi: torch.Tensor,         # [B, T] log-prob of sampled token under current policy
    log_pi_old: torch.Tensor,     # [B, T] log-prob under sampling-time policy
    log_ref: torch.Tensor,        # [B, T] log-prob under frozen reference
    advantages: torch.Tensor,     # [B] one advantage per response
    response_mask: torch.Tensor,  # [B, T] 1 for response tokens, 0 for prompt/pad
    clip_eps: float = 0.2,
    beta_kl: float = 0.04,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """GRPO loss + a few diagnostics.

    All tensors come in pre-aligned: position ``[b, t]`` is the log-prob of
    the sampled token at step ``t`` of response ``b``.
    """
    ratio = torch.exp(log_pi - log_pi_old)
    adv_b = advantages.unsqueeze(-1)  # [B, 1]
    surr1 = ratio * adv_b
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
    pg = torch.minimum(surr1, surr2)

    kl = _per_token_kl(log_pi, log_ref)

    per_token = -(pg - beta_kl * kl)  # negate: minimize
    masked = per_token * response_mask
    # Average within each response, then mean across the batch.
    lengths = response_mask.sum(dim=-1).clamp(min=1.0)
    per_response = masked.sum(dim=-1) / lengths
    loss = per_response.mean()

    metrics = {
        "grpo/ratio_mean": ratio.detach().mean(),
        "grpo/kl_mean": (kl * response_mask).sum().detach() / response_mask.sum().clamp(min=1),
        "grpo/advantage_mean": advantages.detach().mean(),
        "grpo/advantage_std": advantages.detach().std(unbiased=False),
    }
    return loss, metrics


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #


@torch.no_grad()
def _sample_completion(
    policy: nn.Module,
    input_ids: torch.Tensor,         # [1, P]
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    eos_token_id: Optional[int],
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedy/temperature sampling from ``policy``.

    Returns (full_sequence_ids, sampling_logprobs) where:
      * full_sequence_ids: [1, P + L] where L <= max_new_tokens.
      * sampling_logprobs: [1, L] log-prob of each sampled token under the
        sampling-time policy.
    """
    device = input_ids.device
    cur = input_ids
    sampled_logps: List[torch.Tensor] = []

    for _ in range(max_new_tokens):
        out = policy(input_ids=cur)
        logits = out.logits[:, -1, :] / max(temperature, 1e-6)
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        next_tok = torch.multinomial(probs, num_samples=1)  # [1, 1]
        sampled_logps.append(log_probs.gather(-1, next_tok).squeeze(-1))
        cur = torch.cat([cur, next_tok], dim=-1)
        if eos_token_id is not None and next_tok.item() == eos_token_id:
            break

    if not sampled_logps:
        return cur, torch.empty(1, 0, device=device)
    sampling_logps = torch.stack(sampled_logps, dim=-1)  # [1, L]
    return cur, sampling_logps


def _selected_logprobs_full(
    model: nn.Module,
    sequence: torch.Tensor,    # [B, T_total]
    response_start: int,
) -> torch.Tensor:
    """Compute log-probs of the response tokens under ``model`` from a full forward.

    Returns a [B, T_total - 1] tensor where position ``t`` is the log-prob
    of token ``t+1`` given prefix ``[:t+1]``. Caller masks to response only.
    """
    out = model(input_ids=sequence)
    logits = out.logits[:, :-1, :]                  # predict next token
    log_probs = F.log_softmax(logits, dim=-1)       # [B, T-1, V]
    targets = sequence[:, 1:]                       # [B, T-1]
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


class GRPOTrainer(Trainer):
    """GRPO trainer with on-policy sampling and group-relative advantages.

    Constructor takes a tokenizer (for sampling), a reward function, and a
    dataloader yielding raw prompt strings. Each "step" actually does:

      1. For each prompt in the batch, sample ``group_size`` completions.
      2. Compute reward for each completion.
      3. Normalize rewards within each prompt's group → advantages.
      4. Compute log-probs of each sampled token under policy + ref.
      5. Take a single optimizer step on the GRPO loss.
    """

    def __init__(
        self,
        model: nn.Module,
        prompt_dataloader: DataLoader,
        tokenizer,
        reward_fn: RewardFn,
        config: Optional[TrainerConfig] = None,
        grpo_config: Optional[GRPOConfig] = None,
        ref_model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        # The Trainer base class wants a dataloader of model batches, but our
        # rollouts produce them on the fly. We pass a degenerate one so that
        # base-class bookkeeping works (steps_per_epoch, etc.) but our
        # ``compute_loss`` ignores the batch and runs its own sampling.
        super().__init__(model, prompt_dataloader, config, eval_dataloader=None, device=device)
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.grpo_config = grpo_config or GRPOConfig()
        if self.grpo_config.pad_token_id == 0 and tokenizer.pad_token_id is not None:
            self.grpo_config.pad_token_id = tokenizer.pad_token_id

        if ref_model is None:
            ref_model = copy.deepcopy(self._unwrapped_model())
        ref_model.to(self.device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        self.ref_model = ref_model

    # ------------------------------------------------------------------ #
    # One rollout-and-loss step.
    # ------------------------------------------------------------------ #

    def compute_loss(self, batch: Any) -> torch.Tensor:
        """``batch`` is a list of prompt strings (the dataloader yields them)."""
        cfg = self.grpo_config
        prompts: List[str] = batch["prompts"] if isinstance(batch, dict) else list(batch)
        eos = self.tokenizer.eos_token_id
        pad = cfg.pad_token_id
        device = self.device

        # 1) Sample G rollouts per prompt and compute rewards.
        all_seqs: List[torch.Tensor] = []      # each [P_i + L_i]
        all_prompt_lens: List[int] = []
        all_sample_logps: List[torch.Tensor] = []
        all_advantages: List[float] = []
        all_response_lens: List[int] = []

        policy_for_sampling = self._unwrapped_model()
        policy_for_sampling.eval()

        for prompt in prompts:
            ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            P = ids.size(1)
            group_seqs: List[torch.Tensor] = []
            group_logps: List[torch.Tensor] = []
            group_rewards: List[float] = []
            for _ in range(cfg.group_size):
                seq, samp_lp = _sample_completion(
                    policy_for_sampling, ids,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    eos_token_id=eos,
                    pad_token_id=pad,
                )
                response = self.tokenizer.decode(seq[0, P:], skip_special_tokens=True)
                group_rewards.append(float(self.reward_fn(prompt, response)))
                group_seqs.append(seq[0])
                group_logps.append(samp_lp[0])

            r = torch.tensor(group_rewards, dtype=torch.float32, device=device)
            adv = (r - r.mean()) / (r.std(unbiased=False) + cfg.advantage_eps)
            for i in range(cfg.group_size):
                all_seqs.append(group_seqs[i])
                all_prompt_lens.append(P)
                all_sample_logps.append(group_logps[i])
                all_advantages.append(adv[i].item())
                all_response_lens.append(group_seqs[i].size(0) - P)

        policy_for_sampling.train()

        # 2) Pad everything into a single batch.
        max_len = max(s.size(0) for s in all_seqs)
        B = len(all_seqs)
        seq_pad = torch.full((B, max_len), pad, dtype=torch.long, device=device)
        response_mask = torch.zeros((B, max_len - 1), device=device)
        old_logps = torch.zeros((B, max_len - 1), device=device)
        for b, (s, p_len, samp_lp, r_len) in enumerate(
            zip(all_seqs, all_prompt_lens, all_sample_logps, all_response_lens)
        ):
            T = s.size(0)
            seq_pad[b, :T] = s
            # Position t in shifted log-probs corresponds to predicting token t+1.
            # Response tokens occupy indices [p_len, T) of the sequence; their
            # log-probs sit at indices [p_len-1, T-1) of the shifted tensor.
            start = max(p_len - 1, 0)
            end = T - 1
            response_mask[b, start:end] = 1.0
            # Old log-probs from sampling: align to the same shifted axis.
            old_logps[b, start:end] = samp_lp[: end - start]
        advantages = torch.tensor(all_advantages, dtype=torch.float32, device=device)

        # 3) Forward through current policy & frozen ref to get aligned log-probs.
        log_pi = _selected_logprobs_full(self.model, seq_pad, response_start=0)
        with torch.no_grad():
            log_ref = _selected_logprobs_full(self.ref_model, seq_pad, response_start=0)

        # 4) Loss.
        loss, _metrics = compute_grpo_loss(
            log_pi=log_pi,
            log_pi_old=old_logps,
            log_ref=log_ref,
            advantages=advantages,
            response_mask=response_mask,
            clip_eps=cfg.clip_eps,
            beta_kl=cfg.beta_kl,
        )
        return loss
