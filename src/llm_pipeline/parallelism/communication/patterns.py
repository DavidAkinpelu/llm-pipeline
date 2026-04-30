"""Communication patterns for parallelism."""

from enum import Enum


class CommunicationPattern(Enum):
    """Communication patterns for parallelism strategies."""
    
    # Basic communication patterns
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    
    # Pipeline-specific patterns
    PIPELINE_FORWARD = "pipeline_forward"
    PIPELINE_BACKWARD = "pipeline_backward"
    
    # Data parallelism patterns
    DATA_PARALLEL = "data_parallel"
    
    # Model parallelism patterns
    MODEL_PARALLEL = "model_parallel"
    
    # Custom patterns
    CUSTOM_ALL_TO_ALL = "custom_all_to_all"
