"""Unit tests for the base training loop."""

import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from llm_pipeline.training.trainer import Trainer, TrainerConfig


class _TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 8, hidden_size: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.proj(self.embed(input_ids))


def _make_batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3]], dtype=torch.long),
    }


def test_trainer_flushes_partial_gradient_accumulation_window():
    """The trailing microbatches at epoch end should still produce an optimizer step."""
    batch = _make_batch()
    dataloader = DataLoader([batch, batch, batch], batch_size=None)
    trainer = Trainer(
        _TinyLM(),
        dataloader,
        TrainerConfig(
            output_dir=tempfile.mkdtemp(),
            num_epochs=1,
            gradient_accumulation_steps=2,
            log_every=0,
            save_every=0,
        ),
    )

    result = trainer.train()

    assert result["global_step"] == 2
    assert trainer.global_step == 2


def test_trainer_evaluate_supports_non_tensor_batch_fields():
    """Eval should mirror training's device move behavior and preserve metadata fields."""
    train_batch = _make_batch()
    eval_batch = {**_make_batch(), "meta": "keep-me"}
    trainer = Trainer(
        _TinyLM(),
        DataLoader([train_batch], batch_size=None),
        TrainerConfig(output_dir=tempfile.mkdtemp(), log_every=0, save_every=0),
        eval_dataloader=DataLoader([eval_batch], batch_size=None),
    )

    metrics = trainer.evaluate()

    assert "eval_loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["eval_loss"]))
