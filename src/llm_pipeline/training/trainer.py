"""Trainer for supervised fine-tuning with LoRA-aware integration.

A deliberately small training loop that handles the cases this repo cares about:
  * full-precision or AMP (fp16/bf16) forward via torch.amp.autocast
  * gradient accumulation
  * gradient clipping
  * cosine/linear/constant LR schedule
  * periodic eval and checkpointing
  * LoRA-only checkpointing when the model is a LoRAModelWrapper

It is intentionally not a competitor to HF Trainer — it's a clean reference
loop you can read top-to-bottom.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from .optimization import OptimizerConfig, SchedulerConfig, build_optimizer, build_scheduler


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _is_distributed() else 1


def _get_local_rank() -> int:
    """Returns LOCAL_RANK from torchrun, falling back to global rank, then 0."""
    return int(os.environ.get("LOCAL_RANK", _get_rank()))


@dataclass
class TrainerConfig:
    output_dir: str = "./checkpoints"
    num_epochs: int = 1
    max_steps: int = -1  # if >0, overrides num_epochs
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Mixed precision
    precision: str = "bf16"  # one of: fp32, fp16, bf16

    # Distribution strategy when ``torch.distributed`` is initialised:
    #   - "ddp":  DistributedDataParallel (replicate model, all-reduce grads).
    #   - "fsdp": FullyShardedDataParallel (shard params + grads + optimizer state).
    # Single-GPU runs ignore this. FSDP is what you want for >7B params on a
    # single node where DDP would OOM. Validated locally on single-GPU only —
    # multi-GPU validation is the responsibility of the cloud session.
    distribution: str = "ddp"

    # Optimization
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Logging / eval / checkpointing intervals (in optimizer steps)
    log_every: int = 10
    eval_every: int = 0  # 0 disables periodic eval
    save_every: int = 0  # 0 disables periodic checkpointing

    # Misc
    gradient_checkpointing: bool = False
    seed: int = 42


_DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class Trainer:
    """SFT-style trainer.

    The model is expected to return either a tensor of logits or an object/dict
    with a `loss` attribute/key when `labels` are passed (HF convention). If no
    loss is produced by the model, the trainer falls back to flat
    cross-entropy on `logits` vs `labels` (next-token shifting handled here).
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        config: Optional[TrainerConfig] = None,
        eval_dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or TrainerConfig()
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Distributed bookkeeping. The user is responsible for calling
        # dist.init_process_group() before constructing the Trainer (typically
        # via `torchrun` setting LOCAL_RANK / WORLD_SIZE).
        self.rank = _get_rank()
        self.world_size = _get_world_size()
        self.is_main_process = self.rank == 0

        if device is not None:
            self.device = device
        elif _is_distributed() and torch.cuda.is_available():
            local_rank = _get_local_rank()
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.model.to(self.device)

        torch.manual_seed(self.config.seed + self.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed + self.rank)

        if self.config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        # Wrap for distributed training if a process group is active.
        if _is_distributed():
            self._wrap_distributed()

        # Total optimizer steps for scheduler (best-effort estimate)
        steps_per_epoch = max(1, len(train_dataloader) // self.config.gradient_accumulation_steps)
        if self.config.max_steps > 0:
            total_steps = self.config.max_steps
        else:
            total_steps = steps_per_epoch * self.config.num_epochs
        self.config.scheduler.num_training_steps = max(total_steps, self.config.scheduler.num_training_steps or 0)

        # Build optimizer/scheduler from the (possibly DDP-wrapped) model — DDP
        # forwards .parameters() through to the underlying module.
        self.optimizer = build_optimizer(self.model, self.config.optimizer)
        self.scheduler = build_scheduler(self.optimizer, self.config.scheduler)

        self.amp_dtype = _DTYPE_MAP[self.config.precision]
        self.use_amp = self.amp_dtype != torch.float32 and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if (self.use_amp and self.amp_dtype == torch.float16) else None

        self.global_step = 0

    def _wrap_distributed(self) -> None:
        """Apply DDP or FSDP per ``self.config.distribution``.

        DDP: replicates the model across ranks, all-reduces gradients on
        backward. The traditional choice; works on small-to-medium models.

        FSDP: shards parameters, gradients, and optimizer state across
        ranks. The choice for >7B models that don't fit in a single GPU
        replicated. We use the modern ``FullyShardedDataParallel`` with an
        auto-wrap policy that wraps every transformer block (any module
        whose class name ends in ``Layer`` or ``Block``). Mixed precision
        is configured to match ``self.amp_dtype`` when set later.

        **Not validated locally** — needs ≥2 GPU host. Single-GPU code
        path is the same DDP regression already in tests.
        """
        strategy = (self.config.distribution or "ddp").lower()
        if strategy == "ddp":
            from torch.nn.parallel import DistributedDataParallel as DDP
            ddp_kwargs: Dict[str, Any] = {}
            if self.device.type == "cuda":
                ddp_kwargs["device_ids"] = [self.device.index]
                ddp_kwargs["output_device"] = self.device.index
            self.model = DDP(self.model, find_unused_parameters=False, **ddp_kwargs)
            return

        if strategy == "fsdp":
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import MixedPrecision
                from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            except ImportError as e:
                raise ImportError(
                    "FSDP requires a PyTorch build that includes torch.distributed.fsdp. "
                    "All recent PyTorch CUDA builds include it."
                ) from e

            # Mixed-precision config: keep grads + computations in the AMP
            # dtype, but reduce in FP32 for numerical stability.
            mp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
                self.config.precision
            ]
            mp_policy = MixedPrecision(
                param_dtype=mp_dtype,
                reduce_dtype=torch.float32,
                buffer_dtype=mp_dtype,
            )

            # Auto-wrap every transformer block. This identifies them by class
            # name ending in 'Layer' or 'Block' — covers Qwen3DecoderLayer,
            # LlamaDecoderLayer, MistralDecoderLayer, GPTBlock, etc.
            transformer_layer_classes = self._discover_transformer_layer_classes()
            if transformer_layer_classes:
                policy = lambda module, recurse, nonwrapped_numel: transformer_auto_wrap_policy(
                    module, recurse, nonwrapped_numel,
                    transformer_layer_cls=transformer_layer_classes,
                )
            else:
                policy = None

            self.model = FSDP(
                self.model,
                auto_wrap_policy=policy,
                mixed_precision=mp_policy,
                device_id=self.device.index if self.device.type == "cuda" else None,
                use_orig_params=True,
            )
            return

        raise ValueError(f"Unknown distribution strategy: {strategy!r}. Expected 'ddp' or 'fsdp'.")

    def _discover_transformer_layer_classes(self) -> set:
        """Walk the model and return the set of class types whose name
        ends in 'Layer' or 'Block'. Heuristic but covers Qwen3 / Llama /
        Mistral / Gemma / GPT-2-style architectures."""
        classes: set = set()
        for module in self.model.modules():
            name = type(module).__name__
            if name.endswith("DecoderLayer") or name.endswith("EncoderLayer") or name.endswith("Block"):
                classes.add(type(module))
        return classes

    def _unwrapped_model(self) -> nn.Module:
        """Strip DDP/FSDP wrapper if present so callers can hit underlying methods."""
        m = self.model
        # FSDP exposes the inner module via ``.module`` (same as DDP).
        return m.module if hasattr(m, "module") else m

    def _move_batch_to_device(self, batch):
        """Move tensor values in a dict batch to ``self.device`` non-tensors are passed through.

        Handy for trainers (GRPO, RLHF) whose batches contain strings or
        other non-tensor payloads.
        """
        if isinstance(batch, dict):
            return {
                k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        return batch  # list / other — pass through

    def _optimizer_parameters(self):
        """Yield each unique parameter tracked by the optimizer."""
        seen = set()
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if id(param) in seen:
                    continue
                seen.add(id(param))
                yield param

    def _optimizer_step(self, window_microsteps: int) -> None:
        """Apply one optimizer step, correcting partial accumulation windows."""
        accum = self.config.gradient_accumulation_steps
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        if 0 < window_microsteps < accum:
            scale = accum / window_microsteps
            for param in self._optimizer_parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)

        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(list(self._optimizer_parameters()), self.config.max_grad_norm)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    # ------------------------------------------------------------------ #
    # Forward / loss
    # ------------------------------------------------------------------ #

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"})
        # HF-style: outputs has .loss when labels are passed; if not, compute shifted CE.
        labels = batch.get("labels")
        if hasattr(outputs, "loss") and outputs.loss is not None and labels is None:
            return outputs.loss

        # Otherwise compute cross-entropy from logits.
        logits = getattr(outputs, "logits", outputs)
        if labels is None:
            raise ValueError("compute_loss requires labels in the batch when model has no loss.")
        # Shift so that token n predicts token n+1.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    # ------------------------------------------------------------------ #
    # Train loop
    # ------------------------------------------------------------------ #

    def _iter_batches(self):
        """Yield batches indefinitely when max_steps is set, else for num_epochs."""
        cfg = self.config
        if cfg.max_steps > 0:
            # Loop the dataloader until max_steps is reached.
            while True:
                for batch in self.train_dataloader:
                    yield batch
        else:
            for _ in range(cfg.num_epochs):
                for batch in self.train_dataloader:
                    yield batch

    def train(self, on_log: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        cfg = self.config
        if self.is_main_process:
            os.makedirs(cfg.output_dir, exist_ok=True)
        if _is_distributed():
            dist.barrier()
        self.model.train()

        accum = cfg.gradient_accumulation_steps
        running_loss = 0.0
        t_start = time.time()
        microstep = 0

        for batch in self._iter_batches():
            batch = self._move_batch_to_device(batch)

            ctx = torch.amp.autocast("cuda", dtype=self.amp_dtype) if self.use_amp else _NullCtx()
            with ctx:
                loss = self.compute_loss(batch) / accum
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

                if cfg.eval_every and self.eval_dataloader and self.global_step % cfg.eval_every == 0:
                    self.evaluate()

                if cfg.save_every and self.global_step % cfg.save_every == 0:
                    self.save_checkpoint(os.path.join(cfg.output_dir, f"step_{self.global_step}"))

                if cfg.max_steps > 0 and self.global_step >= cfg.max_steps:
                    break

        if microstep % accum != 0 and (cfg.max_steps <= 0 or self.global_step < cfg.max_steps):
            self._optimizer_step(microstep % accum)

        self.save_checkpoint(os.path.join(cfg.output_dir, "final"))
        return {"global_step": self.global_step}

    # ------------------------------------------------------------------ #
    # Eval / checkpoint
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataloader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        n = 0
        for batch in self.eval_dataloader:
            batch = self._move_batch_to_device(batch)
            ctx = torch.amp.autocast("cuda", dtype=self.amp_dtype) if self.use_amp else _NullCtx()
            with ctx:
                loss = self.compute_loss(batch)
            total_loss += loss.item()
            n += 1
        self.model.train()
        avg = total_loss / max(1, n)
        print(f"[eval @ step {self.global_step}] loss={avg:.4f}")
        return {"eval_loss": avg}

    def save_checkpoint(self, path: str) -> None:
        """Save a checkpoint.

        DDP / single-GPU: rank 0 writes the full state dict; others wait at
        a barrier. LoRA-wrapped models save only the adapter weights.

        FSDP: gathers the full sharded state to rank 0 first via
        ``FullStateDictConfig(rank0_only=True, offload_to_cpu=True)`` then
        writes it. The gather is collective so every rank participates.

        For very-large FSDP runs you'd typically use
        ``ShardedStateDictConfig`` instead and write per-rank shards —
        not implemented here, but the hook-up point is the
        ``state_dict_type`` context below.
        """
        is_fsdp = type(self.model).__name__ == "FullyShardedDataParallel"

        if is_fsdp:
            # Collect the full state on rank 0.
            from torch.distributed.fsdp import (
                FullStateDictConfig,
                FullyShardedDataParallel as FSDP,
                StateDictType,
            )
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
                state = self.model.state_dict()
            if not self.is_main_process:
                if _is_distributed():
                    dist.barrier()
                return
            os.makedirs(path, exist_ok=True)
            torch.save(state, os.path.join(path, "model.pt"))
            torch.save(
                {
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "global_step": self.global_step,
                },
                os.path.join(path, "trainer_state.pt"),
            )
            print(f"Saved FSDP checkpoint (full state, rank-0 only) to {path}")
            if _is_distributed():
                dist.barrier()
            return

        # DDP / single-GPU path.
        if not self.is_main_process:
            if _is_distributed():
                dist.barrier()
            return
        os.makedirs(path, exist_ok=True)
        underlying = self._unwrapped_model()
        if hasattr(underlying, "save_lora_weights"):
            underlying.save_lora_weights(os.path.join(path, "lora.pt"))
        else:
            torch.save(underlying.state_dict(), os.path.join(path, "model.pt"))
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
            },
            os.path.join(path, "trainer_state.pt"),
        )
        print(f"Saved checkpoint to {path}")
        if _is_distributed():
            dist.barrier()


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *args): return False
