"""MinHash + LSH document deduplication.

Near-duplicate detection is a standard pretraining-data prep step:
exact-string dedup misses paraphrases / templated boilerplate. MinHash
signatures + locality-sensitive hashing (LSH) bucket by approximate
Jaccard similarity in O(n) total time, where exact pairwise comparison
would be O(n²).

Algorithm
---------

1. Shingle each document into a set of N-grams (default ``ngram=5``
   words).
2. Compute a length-K MinHash signature: K independent hash permutations,
   each producing the min hash value seen across all shingles. Two
   documents with Jaccard similarity ``s`` agree on each MinHash slot
   with probability ``s`` — so the fraction of matching slots
   approximates ``s`` for free.
3. **LSH banding**: split the K signatures into B bands of R slots each.
   Two docs collide in at least one band with probability
   ``1 − (1 − s^R)^B`` — tunable threshold. Collisions go through a
   final exact-Jaccard verification.

Reference: Leskovec, Rajaraman, Ullman, "Mining of Massive Datasets",
Chapter 3.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple

import numpy as np


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _shingles(text: str, ngram: int) -> Set[Tuple[str, ...]]:
    """Word-level N-grams as a set."""
    words = _WORD_RE.findall(text.lower())
    return {tuple(words[i:i + ngram]) for i in range(len(words) - ngram + 1)}


def _hash_shingle(shingle: Tuple[str, ...], seed: int) -> int:
    """Stable 64-bit hash for an N-gram + per-permutation seed."""
    h = hashlib.blake2b(
        ("\x1f".join(shingle)).encode("utf-8"),
        digest_size=8,
        person=seed.to_bytes(8, "little", signed=False),
    )
    return int.from_bytes(h.digest(), "little")


def minhash_signature(text: str, num_perm: int = 128, ngram: int = 5) -> np.ndarray:
    """Compute a length-``num_perm`` MinHash signature for ``text``."""
    shingles = _shingles(text, ngram)
    if not shingles:
        return np.full(num_perm, 0, dtype=np.uint64)
    sig = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    for sh in shingles:
        for perm in range(num_perm):
            h = _hash_shingle(sh, perm)
            if h < sig[perm]:
                sig[perm] = h
    return sig


def jaccard(a: Set, b: Set) -> float:
    """Standard Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass
class MinHashDeduplicator:
    """LSH-bucketed near-duplicate detector.

    Build incrementally with ``add(doc_id, text)``; query the canonical
    set with ``unique_ids()`` once everything is loaded. The first
    inserted document of any near-duplicate cluster is retained; later
    insertions that collide are dropped.

    Threshold (the Jaccard similarity above which docs are "duplicates")
    is implicit in the LSH (B, R) split: ``threshold ≈ (1/B)^(1/R)``.
    Defaults of ``num_perm=128, num_bands=16`` give ``threshold ≈ 0.5``.
    """

    num_perm: int = 128
    num_bands: int = 16
    ngram: int = 5
    similarity_threshold: float = 0.85         # for the exact verification step

    def __post_init__(self):
        if self.num_perm % self.num_bands != 0:
            raise ValueError(
                f"num_perm ({self.num_perm}) must be divisible by "
                f"num_bands ({self.num_bands})"
            )
        self.rows_per_band = self.num_perm // self.num_bands
        self._buckets: List[dict] = [{} for _ in range(self.num_bands)]
        self._sigs: dict[str, np.ndarray] = {}
        self._shingles: dict[str, Set] = {}
        self._kept: List[str] = []
        self._duplicates: dict[str, str] = {}    # dropped_id → first_canonical_id

    def add(self, doc_id: str, text: str) -> bool:
        """Return True iff ``doc_id`` is unique (kept), False if dropped as a near-dup."""
        sig = minhash_signature(text, self.num_perm, self.ngram)
        shingles = _shingles(text, self.ngram)

        # Bucket by band.
        band_keys: List[bytes] = []
        for b in range(self.num_bands):
            sl = sig[b * self.rows_per_band:(b + 1) * self.rows_per_band]
            band_keys.append(sl.tobytes())

        # Check existing buckets for collisions; verify with exact Jaccard.
        for band_idx, key in enumerate(band_keys):
            for cand_id in self._buckets[band_idx].get(key, ()):
                if jaccard(shingles, self._shingles[cand_id]) >= self.similarity_threshold:
                    self._duplicates[doc_id] = cand_id
                    return False

        # Not a duplicate — register.
        for band_idx, key in enumerate(band_keys):
            self._buckets[band_idx].setdefault(key, []).append(doc_id)
        self._sigs[doc_id] = sig
        self._shingles[doc_id] = shingles
        self._kept.append(doc_id)
        return True

    def unique_ids(self) -> List[str]:
        """IDs of documents kept after dedup, in insertion order."""
        return list(self._kept)

    def duplicate_pairs(self) -> dict[str, str]:
        """Mapping ``dropped_doc_id → first_canonical_doc_id``."""
        return dict(self._duplicates)
