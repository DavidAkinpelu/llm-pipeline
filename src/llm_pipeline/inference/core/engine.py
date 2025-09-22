"""Core inference engine implementation."""

import torch
import time
from typing import List, Dict, Optional, Union, Any, Iterator
from dataclasses import dataclass
from enum import Enum

from .kv_cache import KVCache
from .batch_processor import BatchProcessor
from .memory_manager import MemoryManager
from .context_manager import ContextManager, ContextConfig
from ..generation.sampling import SamplingStrategy
from ..generation.constraints import GenerationConstraints
from ..generation.streaming import StreamingGenerator
from ..adapters.composition import AdapterComposer
from ..merging.strategy_selector import MergeStrategySelector, MergeStrategy


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    max_context_length: int = 4096
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stop_tokens: List[str] = None
    use_kv_cache: bool = True
    kv_cache_size: int = 1000
    batch_size: int = 1
    memory_budget_gb: float = 8.0
    default_merge_strategy: MergeStrategy = MergeStrategy.AUTO
    
    def __post_init__(self):
        if self.stop_tokens is None:
            self.stop_tokens = ["<|endoftext|>", "<|end|>", "\n\n"]


class InferenceEngine:
    """Main inference engine for llm_pipeline framework."""
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        config: Optional[InferenceConfig] = None
    ):
        """Initialize inference engine.
        
        Args:
            model: The model to use for inference
            tokenizer: Tokenizer for the model
            config: Inference configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or InferenceConfig()
        
        # Detect paradigm and initialize components
        self.paradigm = self._detect_paradigm(model)
        self._initialize_components()
        
        # Performance monitoring
        self.performance_metrics = {
            "tokens_generated": 0,
            "inference_time": 0.0,
            "memory_used": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
    def _detect_paradigm(self, model) -> str:
        """Detect the training paradigm of the model."""
        if hasattr(model, 'lora_modules') and hasattr(model, 'quantization_config'):
            return "qlora"
        elif hasattr(model, 'lora_modules'):
            return "lora"
        elif hasattr(model, 'quantization_config') and not hasattr(model, 'lora_modules'):
            return "quantized_full"
        else:
            return "full_finetune"
            
    def _initialize_components(self):
        """Initialize all inference components."""
        self.kv_cache = KVCache(max_size=self.config.kv_cache_size)
        self.batch_processor = BatchProcessor()
        self.memory_manager = MemoryManager(
            max_memory_gb=self.config.memory_budget_gb
        )
        self.context_manager = ContextManager(
            config=ContextConfig(max_context_length=self.config.max_context_length)
        )
        self.sampling_strategy = SamplingStrategy()
        self.constraints = GenerationConstraints(
            stop_tokens=self.config.stop_tokens,
            repetition_penalty=self.config.repetition_penalty
        )
        self.streaming_generator = StreamingGenerator(self.model, self.tokenizer)
        self.adapter_composer = AdapterComposer()
        self.merge_strategy_selector = MergeStrategySelector()
        
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        adapter: Optional[str] = None,
        merge_strategy: Optional[Union[str, MergeStrategy]] = None,
        **kwargs
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            adapter: LoRA adapter to use
            merge_strategy: Merging strategy for QLoRA
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Update config with provided parameters
        config = self._update_config(max_tokens, temperature, top_k, top_p)
        
        # Set active adapter if specified
        if adapter:
            self._set_active_adapter(adapter, merge_strategy)
            
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Check context length
        input_ids = self.context_manager.truncate_input(input_ids, config.max_context_length)
        
        # Generate
        start_time = time.time()
        
        if self.config.use_kv_cache:
            output_ids = self._generate_with_cache(input_ids, config)
        else:
            output_ids = self._generate_without_cache(input_ids, config)
            
        generation_time = time.time() - start_time
        
        # Decode output
        generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Apply constraints
        generated_text = self.constraints.apply_stop_tokens(generated_text)
        
        # Update metrics
        self._update_metrics(len(output_ids), generation_time)
        
        return generated_text
        
    def stream_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """Generate text with streaming output.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Yields:
            Generated tokens one by one
        """
        config = self._update_config(max_tokens)
        
        for token in self.streaming_generator.stream(
            prompt, 
            max_tokens=config.max_new_tokens,
            **kwargs
        ):
            yield token
            
    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        return self.batch_processor.process_batch(
            self, prompts, max_tokens=max_tokens, **kwargs
        )
        
    def set_default_merge_strategy(self, strategy: Union[str, MergeStrategy]):
        """Set default merging strategy."""
        if isinstance(strategy, str):
            strategy = MergeStrategy(strategy)
        self.config.default_merge_strategy = strategy
        
    def get_available_merge_strategies(self) -> List[str]:
        """Get list of available merging strategies."""
        return [strategy.value for strategy in MergeStrategy]
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
        
    def clear_cache(self):
        """Clear KV cache."""
        self.kv_cache.clear()
        self.performance_metrics["cache_hits"] = 0
        self.performance_metrics["cache_misses"] = 0
        
    def _update_config(self, max_tokens=None, temperature=None, top_k=None, top_p=None):
        """Update configuration with provided parameters."""
        config = InferenceConfig(
            max_context_length=self.config.max_context_length,
            max_new_tokens=max_tokens or self.config.max_new_tokens,
            temperature=temperature or self.config.temperature,
            top_k=top_k or self.config.top_k,
            top_p=top_p or self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            stop_tokens=self.config.stop_tokens,
            use_kv_cache=self.config.use_kv_cache,
            kv_cache_size=self.config.kv_cache_size,
            batch_size=self.config.batch_size,
            memory_budget_gb=self.config.memory_budget_gb,
            default_merge_strategy=self.config.default_merge_strategy
        )
        return config
        
    def _set_active_adapter(self, adapter: str, merge_strategy: Optional[Union[str, MergeStrategy]]):
        """Set active adapter with optional merge strategy."""
        if self.paradigm in ["lora", "qlora"]:
            if merge_strategy:
                if isinstance(merge_strategy, str):
                    merge_strategy = MergeStrategy(merge_strategy)
                self.model.set_merge_strategy(merge_strategy)
                
            self.model.set_active_adapter(adapter)
            
    def _generate_with_cache(self, input_ids, config):
        """Generate with KV cache."""
        # Check cache
        cache_key = self._generate_cache_key(input_ids)
        cached_kv = self.kv_cache.get(cache_key)
        
        if cached_kv:
            self.performance_metrics["cache_hits"] += 1
            # Use cached KV states
            return self._generate_from_cache(input_ids, cached_kv, config)
        else:
            self.performance_metrics["cache_misses"] += 1
            # Generate and cache
            output_ids, kv_states = self._generate_and_cache(input_ids, config)
            self.kv_cache.store(cache_key, kv_states)
            return output_ids
            
    def _generate_without_cache(self, input_ids, config):
        """Generate without KV cache."""
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return output_ids[0][len(input_ids[0]):]
        
    def _generate_cache_key(self, input_ids):
        """Generate cache key for input sequence."""
        # Convert to a hashable format
        if input_ids.dim() > 1:
            # Flatten multi-dimensional tensor
            flat_ids = input_ids.flatten().tolist()
        else:
            flat_ids = input_ids.tolist()
        return hash(tuple(flat_ids))
        
    def _generate_from_cache(self, input_ids, cached_kv, config):
        """Generate using cached KV states."""
        # Implementation would depend on model architecture
        # For now, fall back to normal generation
        return self._generate_without_cache(input_ids, config)
        
    def _generate_and_cache(self, input_ids, config):
        """Generate and cache KV states."""
        # Implementation would depend on model architecture
        output_ids = self._generate_without_cache(input_ids, config)
        kv_states = {}  # Would contain actual KV cache states
        return output_ids, kv_states
        
    def _update_metrics(self, tokens_generated: int, inference_time: float):
        """Update performance metrics."""
        self.performance_metrics["tokens_generated"] += tokens_generated
        self.performance_metrics["inference_time"] += inference_time
        self.performance_metrics["memory_used"] = self.memory_manager.get_current_usage()
