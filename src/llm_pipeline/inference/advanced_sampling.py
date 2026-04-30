"""Beam search + contrastive search decoding.

Two non-greedy / non-temperature-sampling search strategies that the
existing ``Sampler`` doesn't cover:

- **Beam search** (``beam_search_decode``): keep the top-K hypotheses by
  cumulative log-prob, expand each by the top-K continuations every step.
  Standard for translation and summarisation; deterministic; tends to
  produce repetitive output on open-ended generation.

- **Contrastive search** (``contrastive_search_decode``): Su et al.
  "A Contrastive Framework for Neural Text Generation" (NeurIPS 2022).
  At each step, pick the next token from the top-K candidates by
  maximising ``α · model_confidence(x_t) − (1 − α) · max_similarity(x_t, history)``.
  Penalises tokens whose hidden representation is too close to anything
  in the prefix, which sharply reduces repetition on open-ended outputs.

Both helpers operate on the **logits/sampling** level only — they take a
``model_forward`` callable that returns ``(logits, hidden)`` and a
tokenizer-free input_ids tensor. No KV cache plumbing, no engine surgery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Beam search
# --------------------------------------------------------------------------- #


@dataclass
class BeamHypothesis:
    """One beam under consideration."""

    tokens: torch.LongTensor       # [T] — full sequence including prompt
    score: float                  # cumulative log-prob (higher is better)
    finished: bool = False


def beam_search_decode(
    model_forward: Callable[[torch.Tensor], torch.Tensor],
    prompt_ids: torch.LongTensor,
    max_new_tokens: int,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> Tuple[torch.LongTensor, float]:
    """Vanilla beam search.

    ``model_forward(seq)`` must take a ``[1, T]`` tensor and return
    ``[1, T, V]`` logits — same contract as ``speculative_decode``.

    Returns ``(best_sequence, score)`` where ``best_sequence`` is the
    full token list (prompt + generated) and ``score`` is the
    length-normalised cumulative log-prob.
    """
    if prompt_ids.dim() != 2 or prompt_ids.shape[0] != 1:
        raise ValueError(f"prompt_ids must be shape [1, T]; got {tuple(prompt_ids.shape)}")
    if num_beams < 1:
        raise ValueError(f"num_beams must be ≥ 1; got {num_beams}")

    device = prompt_ids.device
    base_seq = prompt_ids[0]

    # Seed with the single starting beam.
    beams: List[BeamHypothesis] = [BeamHypothesis(tokens=base_seq, score=0.0, finished=False)]

    for _ in range(max_new_tokens):
        candidates: List[BeamHypothesis] = []
        for beam in beams:
            if beam.finished:
                candidates.append(beam)
                continue
            logits = model_forward(beam.tokens.unsqueeze(0))[0, -1, :]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            top_logp, top_tok = log_probs.topk(num_beams)
            for lp, tok in zip(top_logp.tolist(), top_tok.tolist()):
                new_tokens = torch.cat([beam.tokens, torch.tensor([tok], device=device)])
                finished = eos_token_id is not None and tok == eos_token_id
                candidates.append(BeamHypothesis(
                    tokens=new_tokens, score=beam.score + lp, finished=finished,
                ))

        # Length-normalised pick of top-K candidates.
        def _norm_score(b: BeamHypothesis) -> float:
            n_new = max(b.tokens.shape[0] - base_seq.shape[0], 1)
            return b.score / (n_new ** length_penalty)

        candidates.sort(key=_norm_score, reverse=True)
        beams = candidates[:num_beams]

        if all(b.finished for b in beams):
            break

    best = max(beams, key=lambda b: b.score / max(b.tokens.shape[0] - base_seq.shape[0], 1))
    n_new = max(best.tokens.shape[0] - base_seq.shape[0], 1)
    return best.tokens.unsqueeze(0), best.score / (n_new ** length_penalty)


# --------------------------------------------------------------------------- #
# Contrastive search
# --------------------------------------------------------------------------- #


def contrastive_search_decode(
    model_forward_with_hidden: Callable[
        [torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
    ],
    prompt_ids: torch.LongTensor,
    max_new_tokens: int,
    top_k: int = 4,
    alpha: float = 0.6,
    eos_token_id: Optional[int] = None,
) -> torch.LongTensor:
    """Contrastive search (Su et al., NeurIPS 2022).

    At each step:

    1. Take the model's top-K next-token candidates by softmax probability.
    2. For each candidate, compute the max cosine similarity between its
       resulting hidden representation and every previous step's hidden.
    3. Pick the candidate maximising ``α · model_conf − (1 − α) · max_sim``.

    ``α=1.0`` collapses to plain top-K argmax; ``α=0.0`` chooses purely on
    "novelty". The paper recommends ``α=0.6, top_k=4`` as a default.

    ``model_forward_with_hidden(seq)`` must return ``(logits, hidden)``
    where ``hidden`` is ``[B, T, H]`` (the pre-lm-head hidden states).
    """
    if prompt_ids.dim() != 2 or prompt_ids.shape[0] != 1:
        raise ValueError(f"prompt_ids must be shape [1, T]; got {tuple(prompt_ids.shape)}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
    if top_k < 1:
        raise ValueError(f"top_k must be ≥ 1; got {top_k}")

    seq = prompt_ids.clone()
    prompt_len = seq.shape[1]

    # First forward to seed the prefix hidden bank.
    logits, hidden = model_forward_with_hidden(seq)
    prefix_hidden = hidden[0]                    # [T, H]

    for _ in range(max_new_tokens):
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits.float(), dim=-1)
        top_probs, top_toks = probs.topk(top_k)

        # Compute per-candidate hidden by extending the sequence one step at a time.
        # This is the algorithm's main cost; production implementations cache the
        # KV state and only run the embedding + first layers per candidate.
        scores = []
        cand_hiddens = []
        for tok, p in zip(top_toks.tolist(), top_probs.tolist()):
            cand = torch.cat([seq, torch.tensor([[tok]], device=seq.device)], dim=1)
            _, cand_hid_seq = model_forward_with_hidden(cand)
            cand_h = cand_hid_seq[0, -1, :]      # [H]
            # Max cosine similarity of the new hidden vs every prefix hidden.
            ph_norm = F.normalize(prefix_hidden, dim=-1)
            ch_norm = F.normalize(cand_h.unsqueeze(0), dim=-1)
            sim = (ph_norm @ ch_norm.T).max().item()
            score = alpha * p - (1.0 - alpha) * sim
            scores.append(score)
            cand_hiddens.append(cand_h)

        best_idx = max(range(top_k), key=lambda i: scores[i])
        chosen = top_toks[best_idx].unsqueeze(0).unsqueeze(0)       # [1, 1]
        seq = torch.cat([seq, chosen], dim=1)
        prefix_hidden = torch.cat([prefix_hidden, cand_hiddens[best_idx].unsqueeze(0)], dim=0)

        if eos_token_id is not None and chosen.item() == eos_token_id:
            break

        # Refresh logits from the chosen path for the next iteration.
        logits, hidden = model_forward_with_hidden(seq)

    return seq[:, prompt_len:]
