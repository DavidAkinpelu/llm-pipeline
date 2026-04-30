"""Unit tests for the Muon optimizer wiring."""

import torch
import torch.nn as nn

from llm_pipeline.training.optimization import (
    OptimizerConfig,
    SingleDeviceMuonWithAuxAdam,
    build_optimizer,
    split_muon_param_groups,
)
from llm_pipeline.training.optimization.optimizers.muon import (
    zeropower_via_newtonschulz5,
)


class _TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.fc1 = nn.Linear(hidden, hidden, bias=True)
        self.norm = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, x):
        h = self.embed(x)
        h = self.fc2(torch.relu(self.fc1(h)))
        h = self.norm(h)
        return self.lm_head(h)


def test_split_routes_embeddings_and_head_to_adam():
    model = _TinyLM()
    muon, adam = split_muon_param_groups(model)
    muon_ids = {id(p) for p in muon}
    adam_ids = {id(p) for p in adam}

    # embed weight + lm_head weight + fc1 bias + layernorm weight/bias all go to AdamW
    assert id(model.embed.weight) in adam_ids
    assert id(model.lm_head.weight) in adam_ids
    assert id(model.fc1.bias) in adam_ids
    assert id(model.norm.weight) in adam_ids
    assert id(model.norm.bias) in adam_ids
    # hidden 2D weights go to Muon
    assert id(model.fc1.weight) in muon_ids
    assert id(model.fc2.weight) in muon_ids
    # no overlap
    assert muon_ids.isdisjoint(adam_ids)


def test_build_optimizer_muon_returns_hybrid():
    model = _TinyLM()
    cfg = OptimizerConfig(name="muon", lr=0.02, aux_adam_lr=3e-4, weight_decay=0.0)
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, SingleDeviceMuonWithAuxAdam)
    flags = sorted(g["use_muon"] for g in opt.param_groups)
    assert flags == [False, True]


def test_muon_step_decreases_loss_on_synthetic_problem():
    torch.manual_seed(0)
    model = _TinyLM()
    cfg = OptimizerConfig(name="muon", lr=0.02, aux_adam_lr=1e-3, weight_decay=0.0)
    opt = build_optimizer(model, cfg)

    x = torch.randint(0, 16, (4, 6))
    y = torch.randint(0, 16, (4, 6))

    def loss_fn():
        logits = model(x)
        return nn.functional.cross_entropy(logits.reshape(-1, 16), y.reshape(-1))

    initial = loss_fn().item()
    for _ in range(20):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    final = loss_fn().item()
    assert final < initial, f"loss did not decrease: {initial:.4f} -> {final:.4f}"


def test_newton_schulz_approximate_orthogonalization():
    torch.manual_seed(0)
    G = torch.randn(8, 16)
    X = zeropower_via_newtonschulz5(G, steps=5).float()
    # Singular values should land near 1 (the iteration intentionally produces
    # S' ~ Uniform(0.5, 1.5), not exact UV^T — so we just check the spread).
    s = torch.linalg.svdvals(X)
    assert s.min() > 0.4
    assert s.max() < 1.6
