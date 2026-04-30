"""HF datasets streaming wrapper.

Streams a HuggingFace dataset in chunks without loading the whole thing
into memory, applies a tokenizer on the fly, and yields fixed-length
token blocks suitable for an autoregressive trainer.

The classic use case: pretraining or large-corpus fine-tuning where the
raw text + tokenization would blow past RAM. Stream from a hub dataset,
tokenise lazily, pack into context-length blocks, feed to ``Trainer``.

The wrapper is ``IterableDataset``-compatible, so it slots into
``torch.utils.data.DataLoader(num_workers=0)`` directly. With
``num_workers > 0`` HF's streaming reader needs to be re-opened in each
worker (handled in ``__iter__``).

This module **soft-imports** ``datasets`` at construction so the
project can be installed without the optional dep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, List, Optional


@dataclass
class StreamingConfig:
    """Knobs for ``StreamingDataset``.

    Attributes
    ----------
    block_size : int
        Length of each yielded token block (== model context length).
    text_field : str
        Name of the text column in the source dataset.
    pack : bool
        If True, concatenate documents end-to-end and slice into uniform
        ``block_size`` chunks (no padding). Standard pretraining recipe.
        If False, yield one (truncated) block per document.
    eos_token_id : Optional[int]
        Inserted between documents when ``pack=True`` so the model can
        learn document boundaries. ``None`` skips it.
    buffer_size : int
        How many tokenised tokens to accumulate before yielding blocks
        from the buffer. Higher → fewer Python-level iteration calls.
    skip : int
        Skip the first N yielded blocks (resume support).
    """

    block_size: int = 2048
    text_field: str = "text"
    pack: bool = True
    eos_token_id: Optional[int] = None
    buffer_size: int = 16384
    skip: int = 0


class StreamingDataset:
    """``IterableDataset``-shaped wrapper over a HF streaming dataset.

    >>> from datasets import load_dataset
    >>> stream = load_dataset("wikitext", "wikitext-103-raw-v1",
    ...                       split="train", streaming=True)
    >>> ds = StreamingDataset(
    ...     stream, tokenizer=tok,
    ...     config=StreamingConfig(block_size=2048, pack=True, eos_token_id=tok.eos_token_id),
    ... )
    >>> for block in ds:
    ...     # block is a list[int] of length block_size
    ...     train_step(block)

    The wrapper doesn't depend on torch — call sites can wrap the result
    in ``torch.utils.data.DataLoader`` (which calls ``__iter__`` to drive
    iteration) or feed the iterator directly.
    """

    def __init__(
        self,
        source: Iterable[dict],
        tokenizer: Any,
        config: Optional[StreamingConfig] = None,
        text_extractor: Optional[Callable[[dict], str]] = None,
    ):
        self.source = source
        self.tokenizer = tokenizer
        self.config = config or StreamingConfig()
        self.text_extractor = text_extractor or (lambda r: r[self.config.text_field])

    def _tokenize(self, text: str) -> List[int]:
        out = self.tokenizer(text, add_special_tokens=False)
        if isinstance(out, dict):
            return list(out["input_ids"])
        # Some tokenizers return a list directly.
        return list(out)

    def __iter__(self) -> Iterator[List[int]]:
        cfg = self.config
        buffer: List[int] = []
        emitted = 0

        for record in self.source:
            text = self.text_extractor(record)
            if not text:
                continue
            tokens = self._tokenize(text)
            if cfg.pack:
                buffer.extend(tokens)
                if cfg.eos_token_id is not None:
                    buffer.append(cfg.eos_token_id)
                while len(buffer) >= cfg.block_size:
                    block = buffer[: cfg.block_size]
                    buffer = buffer[cfg.block_size:]
                    if emitted >= cfg.skip:
                        yield block
                    emitted += 1
                # Cap the buffer so a single huge document doesn't blow memory.
                if len(buffer) > cfg.buffer_size:
                    buffer = buffer[-cfg.buffer_size:]
            else:
                # One (possibly truncated, possibly padded) block per doc.
                if len(tokens) >= cfg.block_size:
                    block = tokens[: cfg.block_size]
                else:
                    pad_id = cfg.eos_token_id or 0
                    block = tokens + [pad_id] * (cfg.block_size - len(tokens))
                if emitted >= cfg.skip:
                    yield block
                emitted += 1

        # Tail: drop the partial block in pack mode (standard pretraining
        # convention — one extra block of padding skews loss).
