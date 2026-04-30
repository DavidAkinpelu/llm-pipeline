"""Memory-mapped token dataset for >RAM corpora.

Pretraining tokenized corpora typically run to hundreds of GB. Storing
them as a single ``np.memmap`` of int32 token IDs keeps memory usage
flat (the OS pages chunks in on demand) and gives random-access cheap
enough that you can cleanly reshuffle epoch-to-epoch.

Standard layout
---------------

A ``MemmapTokenDataset`` operates on a single contiguous binary file of
token IDs. Build it once offline by calling ``MemmapTokenDataset.build(
output_path, token_iterator, dtype=np.int32)``; query it at training
time with random-access slicing.

The ``__getitem__`` interface returns ``block_size``-length windows
into the token stream, drawn from random offsets — i.e. the standard
"random crops from a packed token sequence" pretraining recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np


class MemmapTokenDataset:
    """Random-access dataset over a binary token-id file.

    >>> # Offline: build the memmap file from a token iterator.
    >>> MemmapTokenDataset.build(
    ...     output_path="train.bin",
    ...     token_iterator=iter_token_ids,            # any iterable[list[int]]
    ...     dtype=np.int32,
    ... )
    >>> # At training time: random-access reads.
    >>> ds = MemmapTokenDataset("train.bin", block_size=2048, dtype=np.int32)
    >>> sample = ds[0]                                # numpy array, shape (block_size,)
    >>> len(ds)                                        # number of disjoint blocks
    """

    def __init__(
        self,
        path: str | Path,
        block_size: int,
        dtype: np.dtype = np.int32,
    ):
        self.path = Path(path)
        self.block_size = block_size
        self.dtype = np.dtype(dtype)
        if not self.path.exists():
            raise FileNotFoundError(f"memmap file not found: {self.path}")
        # Read-only memmap so the OS can page in / page out without copy-on-write.
        self.mm = np.memmap(self.path, dtype=self.dtype, mode="r")
        self.n_tokens = self.mm.shape[0]
        if self.n_tokens < block_size:
            raise ValueError(
                f"corpus has {self.n_tokens} tokens but block_size={block_size}"
            )

    def __len__(self) -> int:
        # Number of disjoint blocks — for ``random_offset`` access there are
        # ``n_tokens - block_size`` valid windows; we expose the disjoint
        # count to drive standard ``__getitem__`` epoch sweeps.
        return self.n_tokens // self.block_size

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        start = idx * self.block_size
        return np.array(self.mm[start:start + self.block_size])

    def random_window(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample a random ``block_size`` window from any offset (overlapping
        windows allowed). Standard pretraining sampler.
        """
        rng = rng or np.random.default_rng()
        start = int(rng.integers(0, self.n_tokens - self.block_size + 1))
        return np.array(self.mm[start:start + self.block_size])

    @classmethod
    def build(
        cls,
        output_path: str | Path,
        token_iterator: Iterable[List[int]],
        dtype: np.dtype = np.int32,
        chunk_size: int = 1 << 20,
    ) -> int:
        """Stream-tokens-to-disk: write ``token_iterator`` into a flat binary
        file at ``output_path``. Returns total tokens written.

        ``chunk_size`` tokens per write batch keeps memory bounded.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np_dtype = np.dtype(dtype)
        buffer: List[int] = []
        n_written = 0
        with output_path.open("wb") as f:
            for doc_tokens in token_iterator:
                buffer.extend(doc_tokens)
                while len(buffer) >= chunk_size:
                    arr = np.asarray(buffer[:chunk_size], dtype=np_dtype)
                    f.write(arr.tobytes())
                    n_written += chunk_size
                    buffer = buffer[chunk_size:]
            if buffer:
                arr = np.asarray(buffer, dtype=np_dtype)
                f.write(arr.tobytes())
                n_written += len(buffer)
        return n_written
