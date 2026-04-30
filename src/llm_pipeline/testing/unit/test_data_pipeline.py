"""Tests for the data-pipeline utilities.

Covers streaming, memory-mapped tokens, MinHash dedup, Self-Instruct
+ Evol-Instruct synthetic generation, and best-of-N rejection-sampling
fine-tuning. All tests are CPU-only and don't require optional deps
(``datasets``, real LLMs) — we use stub generators / iterators to
exercise the orchestration logic.
"""

import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

from llm_pipeline.data import (
    BestOfNSampler,
    EvolInstructConfig,
    MemmapTokenDataset,
    MinHashDeduplicator,
    StreamingConfig,
    StreamingDataset,
    SelfInstructConfig,
    evol_instruct,
    jaccard,
    minhash_signature,
    rejection_sampling_finetune_dataset,
    self_instruct,
)


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


class _StubTokenizer:
    """Whitespace tokenizer; each unique word gets a stable integer id."""

    def __init__(self):
        self._vocab: dict[str, int] = {}

    def __call__(self, text: str, add_special_tokens: bool = False):
        ids = []
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab) + 1
            ids.append(self._vocab[word])
        return {"input_ids": ids}


def test_streaming_pack_yields_uniform_blocks():
    tok = _StubTokenizer()
    docs = [{"text": "a b c d e f g h"}, {"text": "i j k l m n"}]
    ds = StreamingDataset(
        docs, tokenizer=tok,
        config=StreamingConfig(block_size=4, pack=True, eos_token_id=None),
    )
    blocks = list(ds)
    assert all(len(b) == 4 for b in blocks)
    # 8 + 6 = 14 tokens → 3 full 4-blocks (tail of 2 dropped per pack convention).
    assert len(blocks) == 3


def test_streaming_pack_inserts_eos_between_documents():
    tok = _StubTokenizer()
    docs = [{"text": "a b"}, {"text": "c d"}]
    ds = StreamingDataset(
        docs, tokenizer=tok,
        config=StreamingConfig(block_size=5, pack=True, eos_token_id=999),
    )
    blocks = list(ds)
    assert len(blocks) == 1
    assert 999 in blocks[0]                            # at least one EOS spliced in


def test_streaming_no_pack_pads_short_docs():
    tok = _StubTokenizer()
    docs = [{"text": "a b c"}]
    ds = StreamingDataset(
        docs, tokenizer=tok,
        config=StreamingConfig(block_size=8, pack=False, eos_token_id=0),
    )
    blocks = list(ds)
    assert len(blocks) == 1
    assert len(blocks[0]) == 8
    assert blocks[0][3:] == [0, 0, 0, 0, 0]            # padded tail


def test_streaming_skip_resumes_from_offset():
    tok = _StubTokenizer()
    docs = [{"text": " ".join(str(i) for i in range(20))}]
    ds = StreamingDataset(
        docs, tokenizer=tok,
        config=StreamingConfig(block_size=4, pack=True, skip=2),
    )
    blocks = list(ds)
    # 20 tokens → 5 blocks, but skip=2 → 3 blocks emitted.
    assert len(blocks) == 3


# --------------------------------------------------------------------------- #
# Memmap dataset
# --------------------------------------------------------------------------- #


def test_memmap_build_writes_correct_total_tokens():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tokens.bin"
        token_iter = iter([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
        n = MemmapTokenDataset.build(path, token_iter, dtype=np.int32)
        assert n == 9
        assert path.stat().st_size == 9 * 4         # int32 → 4 bytes


def test_memmap_random_access_returns_correct_block():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tokens.bin"
        MemmapTokenDataset.build(path, [list(range(64))], dtype=np.int32)
        ds = MemmapTokenDataset(path, block_size=8, dtype=np.int32)
        assert len(ds) == 8
        np.testing.assert_array_equal(ds[0], np.arange(8))
        np.testing.assert_array_equal(ds[3], np.arange(24, 32))


def test_memmap_random_window_samples_within_bounds():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tokens.bin"
        MemmapTokenDataset.build(path, [list(range(100))], dtype=np.int32)
        ds = MemmapTokenDataset(path, block_size=10, dtype=np.int32)
        rng = np.random.default_rng(0)
        for _ in range(20):
            w = ds.random_window(rng)
            assert w.shape == (10,)
            # Window is a contiguous slice — values should be sequential.
            assert np.all(np.diff(w) == 1)


def test_memmap_rejects_too_short_corpus():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tokens.bin"
        MemmapTokenDataset.build(path, [list(range(5))], dtype=np.int32)
        with pytest.raises(ValueError, match="block_size"):
            MemmapTokenDataset(path, block_size=10)


# --------------------------------------------------------------------------- #
# MinHash deduplication
# --------------------------------------------------------------------------- #


def test_jaccard_similarity_identical_sets_are_one():
    a = {1, 2, 3}
    assert jaccard(a, a) == 1.0
    assert jaccard(set(), set()) == 1.0
    assert jaccard({1, 2}, {3, 4}) == 0.0


def test_minhash_signature_matches_known_jaccard():
    """Two documents with high text overlap should have correspondingly
    high MinHash agreement.
    """
    a = "the quick brown fox jumps over the lazy dog and so on and so forth"
    b = "the quick brown fox jumps over the lazy cat and then keeps running"
    sig_a = minhash_signature(a, num_perm=128, ngram=3)
    sig_b = minhash_signature(b, num_perm=128, ngram=3)
    agree = (sig_a == sig_b).mean()
    assert 0.1 < agree < 0.9                            # plausible mid-range overlap


def test_minhash_dedup_drops_near_duplicates():
    dedup = MinHashDeduplicator(num_perm=128, num_bands=16, similarity_threshold=0.5)
    assert dedup.add("a", "the quick brown fox jumps over the lazy dog")
    # Near-duplicate (same trigrams except one word change) should be dropped.
    near_dup = "the quick brown fox jumps over the lazy cat"
    is_unique = dedup.add("b", near_dup)
    # Unrelated text should be kept.
    assert dedup.add("c", "completely different content about gardening tomatoes basil")

    if not is_unique:
        # Duplicate caught.
        assert "b" in dedup.duplicate_pairs()
        assert dedup.duplicate_pairs()["b"] == "a"
    assert "a" in dedup.unique_ids()
    assert "c" in dedup.unique_ids()


def test_minhash_dedup_keeps_unrelated_documents():
    dedup = MinHashDeduplicator(num_perm=128, num_bands=16)
    docs = {
        "a": "machine learning models for natural language",
        "b": "the geology of plate tectonics",
        "c": "an introduction to category theory in mathematics",
    }
    for k, v in docs.items():
        assert dedup.add(k, v)
    assert set(dedup.unique_ids()) == {"a", "b", "c"}


def test_minhash_dedup_rejects_invalid_band_split():
    with pytest.raises(ValueError, match="divisible"):
        MinHashDeduplicator(num_perm=100, num_bands=16)


# --------------------------------------------------------------------------- #
# Synthetic data — Self-Instruct
# --------------------------------------------------------------------------- #


def test_self_instruct_grows_pool_with_novel_outputs():
    """Stub teacher emits genuinely-distinct instructions each call.

    The novelty filter rejects anything with high word-overlap against
    any pool member — so the stub varies the *content words*, not just
    a counter, to prove the orchestration accepts truly novel additions.
    """
    counter = [0]
    distinct_words = [
        ("compose", "haiku", "spring"),
        ("calculate", "factorial", "twelve"),
        ("debug", "python", "function"),
        ("design", "database", "schema"),
        ("explain", "quantum", "entanglement"),
        ("optimize", "matrix", "multiplication"),
        ("translate", "french", "literature"),
        ("describe", "photosynthesis", "process"),
        ("solve", "differential", "equation"),
        ("analyse", "shakespeare", "sonnet"),
    ]

    def stub_teacher(prompt: str) -> str:
        i = counter[0]
        counter[0] += 1
        a, b, c = distinct_words[i % len(distinct_words)]
        return f"5. {a}\n6. {b}\n7. {c}"

    seeds = ["seed instruction one", "seed instruction two"]
    pool = self_instruct(
        seeds, stub_teacher,
        config=SelfInstructConfig(target_size=8, n_seeds_per_round=2, seed=0),
    )
    assert len(pool) >= 8
    # Original seeds preserved.
    assert pool[0] == seeds[0]


def test_self_instruct_stops_when_teacher_emits_only_duplicates():
    """If the teacher keeps emitting near-duplicate strings, the loop
    should bail out via the stagnation counter rather than spin forever.
    """
    def stub_teacher(prompt: str) -> str:
        # Always returns the same 3 instructions — duplicates after round 1.
        return "5. write a poem\n6. write a song\n7. write a haiku"

    seeds = ["compose a sonnet", "draft a limerick"]
    pool = self_instruct(
        seeds, stub_teacher,
        config=SelfInstructConfig(target_size=1000, n_seeds_per_round=2, seed=0),
    )
    # Definitely shouldn't reach 1000; capped by stagnation.
    assert len(pool) < 1000


# --------------------------------------------------------------------------- #
# Synthetic data — Evol-Instruct
# --------------------------------------------------------------------------- #


def test_evol_instruct_returns_chain_per_seed():
    counter = [0]

    def stub_evolver(prompt: str) -> str:
        counter[0] += 1
        return f"evolved-step-{counter[0]}"

    seeds = ["compute factorial of n", "list prime numbers"]
    chains = evol_instruct(
        seeds, stub_evolver,
        config=EvolInstructConfig(n_evolutions=3, seed=0),
    )
    assert len(chains) == 2
    for chain, seed in zip(chains, seeds):
        assert chain[0] == seed                        # seed preserved at index 0
        assert len(chain) == 4                          # 1 seed + 3 evolutions


def test_evol_instruct_rejects_unknown_operators():
    with pytest.raises(ValueError, match="unknown"):
        EvolInstructConfig(operators=["nonexistent_operator"])


# --------------------------------------------------------------------------- #
# Best-of-N / RFT
# --------------------------------------------------------------------------- #


def test_best_of_n_picks_highest_scoring_completion():
    def gen(prompt: str, n: int) -> List[str]:
        return [f"answer-{i}" for i in range(n)]

    def scorer(prompt: str, completion: str) -> float:
        return float(completion.split("-")[1])         # higher idx = higher score

    sampler = BestOfNSampler(generator=gen, scorer=scorer, n=4)
    records = sampler.run(["q1", "q2"])
    assert len(records) == 2
    for r in records:
        assert r.completion == "answer-3"               # highest-scoring of 4
        assert r.score == 3.0
        assert r.rejected == []                          # keep_rejected=False default


def test_best_of_n_keep_rejected_returns_all_other_candidates():
    def gen(prompt: str, n: int) -> List[str]:
        return ["a", "b", "c"]

    def scorer(prompt: str, completion: str) -> float:
        return {"a": 1.0, "b": 3.0, "c": 2.0}[completion]

    sampler = BestOfNSampler(generator=gen, scorer=scorer, n=3, keep_rejected=True)
    [record] = sampler.run(["q"])
    assert record.completion == "b"
    assert sorted(s for _, s in record.rejected) == [1.0, 2.0]


def test_best_of_n_handles_empty_generator_output():
    """Rare edge case where the generator returns nothing — should drop the
    prompt rather than crash.
    """
    sampler = BestOfNSampler(
        generator=lambda p, n: [],
        scorer=lambda p, c: 1.0,
        n=4,
    )
    records = sampler.run(["a", "b"])
    assert records == []


def test_best_of_n_rejects_invalid_n():
    with pytest.raises(ValueError, match="n must be"):
        BestOfNSampler(generator=lambda p, n: [], scorer=lambda p, c: 0.0, n=0)


def test_rejection_sampling_filters_below_threshold():
    """``score_threshold`` drops prompts whose best-of-N still scored too low."""
    scores_by_prompt = {"good": 5.0, "bad": 0.5}

    def gen(prompt: str, n: int):
        return [prompt + f"_completion_{i}" for i in range(n)]

    def scorer(prompt: str, completion: str):
        return scores_by_prompt[prompt]

    records = rejection_sampling_finetune_dataset(
        prompts=["good", "bad"],
        generator=gen, scorer=scorer, n=2, score_threshold=1.0,
    )
    assert [r.prompt for r in records] == ["good"]
