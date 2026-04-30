"""Data-pipeline utilities for training and evaluation.

Five modules, each independent enough to be used on its own:

- ``streaming``: thin wrapper over HF ``datasets`` streaming mode that
  adds tokenisation + on-the-fly chunking.
- ``memmap``: ``np.memmap``-backed dataset for >RAM token-id corpora.
- ``minhash``: MinHash + LSH-style document deduplication.
- ``synth``: orchestration for Self-Instruct / Evol-Instruct synthetic
  data generation (plug your own LLM in).
- ``rft``: best-of-N / rejection-sampling fine-tuning (generate N
  candidates per prompt, score with a reward model, keep the best).
"""

from .streaming import StreamingDataset, StreamingConfig
from .memmap import MemmapTokenDataset
from .minhash import MinHashDeduplicator, jaccard, minhash_signature
from .synth import (
    SynthGenerator,
    SelfInstructConfig,
    EvolInstructConfig,
    self_instruct,
    evol_instruct,
)
from .rft import (
    BestOfNSampler,
    RFTRecord,
    rejection_sampling_finetune_dataset,
)
from .curriculum import (
    CompositeScorer,
    CurriculumDataset,
    CurriculumStepHook,
    DifficultyScorer,
    ExponentialPacing,
    LengthScorer,
    LinearPacing,
    MetadataScorer,
    PacingFunction,
    PerplexityScorer,
    SelfPacedSampler,
    SqrtPacing,
    StepPacing,
)

__all__ = [
    "StreamingDataset", "StreamingConfig",
    "MemmapTokenDataset",
    "MinHashDeduplicator", "jaccard", "minhash_signature",
    "SynthGenerator", "SelfInstructConfig", "EvolInstructConfig",
    "self_instruct", "evol_instruct",
    "BestOfNSampler", "RFTRecord", "rejection_sampling_finetune_dataset",
    "DifficultyScorer", "PacingFunction",
    "LengthScorer", "MetadataScorer", "PerplexityScorer", "CompositeScorer",
    "LinearPacing", "SqrtPacing", "ExponentialPacing", "StepPacing",
    "CurriculumDataset", "SelfPacedSampler", "CurriculumStepHook",
]
