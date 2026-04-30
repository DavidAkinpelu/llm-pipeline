"""Qwen3-specific inference engine with all advanced features."""

import torch
import torch.nn.functional as F
import time
from typing import Optional, List, Dict, Any, Union, Iterator
from dataclasses import dataclass
from ..models.qwen3.factory import Qwen3Factory
from .sampling import SamplingConfig, Sampler
from .batching import BatchProcessor, BatchConfig
from .streaming import StreamingGenerator
from .continuous_batching import ContinuousBatcher, ContinuousBatchConfig
from .kv_cache import KVCache
from .paged_attention import PagedAttentionManager, PagedAttentionConfig
from .caching import PrefixCache, PrefixCacheConfig
from .attention import get_attention_layer, AttentionConfig
from ..parallelism import ParallelismFactory, TensorParallelConfig
from typing import Literal


@dataclass
class Qwen3InferenceConfig:
    """Configuration for Qwen3 inference engine with all advanced features."""
    
    # Model configuration
    model_path: str
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    use_cache: bool = True
    
    # Parallelism configuration
    enable_tensor_parallel: bool = False
    tensor_parallel_size: int = 1
    tensor_parallel_rank: int = 0
    
    # Generation configuration
    max_length: int = 2048
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    typical_p: float = 1.0
    min_p: float = 0.0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # Batch configuration
    max_batch_size: int = 8
    max_wait_time: float = 0.1
    
    # Streaming configuration
    enable_streaming: bool = True
    stream_chunk_size: int = 1
    
    # Continuous batching configuration
    enable_continuous_batching: bool = False
    continuous_batch_size: int = 4
    
    # KV Cache configuration
    kv_cache_size: int = 1000
    
    # Paged attention settings
    use_paged_attention: bool = False
    block_size: int = 16
    num_blocks: int = 1024
    max_sequence_length: int = 2048
    memory_fraction: float = 0.8
    
    # Optimization settings
    use_torch_compile: bool = False
    use_cuda_graphs: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "max-autotune"
    
    # Core prefix caching (always enabled by default)
    prefix_cache_size: int = 1000
    max_prefix_length: int = 128
    
    # Core Flash Attention (auto-detected)
    use_flash_attention: bool = True


class Qwen3InferenceEngine:
    """Qwen3-specific inference engine with all advanced features."""
    
    def __init__(self, config: Qwen3InferenceConfig):
        """Initialize Qwen3 inference engine with all advanced features."""
        self.config = config
        self.device = torch.device(config.device) if isinstance(config.device, str) else config.device
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        if self.config.pad_token_id is None and self.tokenizer.pad_token_id is not None:
            self.config.pad_token_id = self.tokenizer.pad_token_id
        if self.config.eos_token_id is None and self.tokenizer.eos_token_id is not None:
            self.config.eos_token_id = self.tokenizer.eos_token_id
        
        prefix_cache_config = PrefixCacheConfig(
            max_cache_size=self.config.prefix_cache_size,
            max_prefix_length=self.config.max_prefix_length
        )
        self.prefix_cache = PrefixCache(prefix_cache_config)
        
        attention_config = AttentionConfig(use_flash_attention=self.config.use_flash_attention)
        self.attention_layer = get_attention_layer(attention_config)
        
        # Aliases: ``parallelism`` is the modern name; ``tp_engine`` kept for back-compat.
        self.parallelism = None
        self.tp_engine = None
        if self.config.enable_tensor_parallel:
            tp_config = TensorParallelConfig(
                tensor_parallel_size=self.config.tensor_parallel_size,
                tensor_parallel_rank=self.config.tensor_parallel_rank
            )
            self.parallelism = ParallelismFactory.create_parallelism(tp_config)
            self.tp_engine = self.parallelism
            if self.parallelism:
                self.parallelism.setup_distributed()
                self.model = self.parallelism.wrap_model(self.model)
        
        self.paged_attention_manager = None
        if self.config.use_paged_attention:
            paged_config = PagedAttentionConfig(
                block_size=self.config.block_size,
                num_blocks=self.config.num_blocks,
                max_sequence_length=self.config.max_sequence_length,
                memory_fraction=self.config.memory_fraction
            )
            self.paged_attention_manager = PagedAttentionManager(
                paged_config, self.model.config, self.device
            )
            
        self.kv_cache = None
        if self.config.use_cache and not self.config.use_paged_attention and not self.config.use_torch_compile:
            self.kv_cache = KVCache(self.config.kv_cache_size)
        
        sampling_config = SamplingConfig(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            typical_p=self.config.typical_p,
            min_p=self.config.min_p,
            mirostat_tau=self.config.mirostat_tau,
            mirostat_eta=self.config.mirostat_eta,
            do_sample=self.config.do_sample,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id
        )
        self.sampler = Sampler(sampling_config)
        
        batch_config = BatchConfig(
            max_batch_size=self.config.max_batch_size,
            max_wait_time=self.config.max_wait_time
        )
        self.batch_processor = BatchProcessor(batch_config)
        
        self.streaming_generator = StreamingGenerator(self)
        
        self.continuous_batcher = None
        if self.config.enable_continuous_batching:
            continuous_config = ContinuousBatchConfig(
                max_batch_size=self.config.continuous_batch_size
            )
            self.continuous_batcher = ContinuousBatcher(self, continuous_config)
        
        self._apply_optimizations()
    
    def _load_model_and_tokenizer(self):
        """Load Qwen3 model and HuggingFace tokenizer."""
        print(f"Loading Qwen3 model from {self.config.model_path}")
        
        model, tokenizer = Qwen3Factory.create_model_and_tokenizer(
            model_path=self.config.model_path,
            device=self.config.device,
            dtype=self.config.dtype
        )
        
        print(f"Model loaded successfully. Device: {self.device}, dtype: {self.config.dtype}")
        return model, tokenizer
    
    def _apply_optimizations(self):
        """Apply torch.compile and CUDA graph optimizations."""
        if self.config.use_torch_compile and not self.config.use_cuda_graphs:
            self._compile_model()
        elif self.config.use_torch_compile and self.config.use_cuda_graphs:
            print("torch.compile and CUDA graphs are incompatible. Using torch.compile only.")
            self.config.use_cuda_graphs = False
            self._compile_model()
        
        if self.config.use_cuda_graphs and torch.cuda.is_available() and not self.config.use_torch_compile:
            self._setup_cuda_graphs()
    
    def _compile_model(self):
        """Compile the model with torch.compile."""
        try:
            self.model = torch.compile(
                self.model,
                mode=self.config.compile_mode,
                dynamic=True,  # Allow dynamic shapes
                backend='inductor'  # Use Triton backend for GPU
            )
            print("Model compiled successfully with torch.compile")
        except Exception as e:
            print(f"Model compilation failed: {e}")
            self.config.use_torch_compile = False
    
    def _setup_cuda_graphs(self):
        """Setup CUDA graphs for common batch sizes."""
        self.cuda_graphs = {}
        
        if self.config.use_paged_attention:
            self._setup_paged_attention_graphs()
        else:
            self._setup_standard_graphs()
    
    def _setup_paged_attention_graphs(self):
        """Setup CUDA graphs for paged attention."""
        max_batch = min(self.config.max_batch_size, 4)  # Cap at 4 to avoid OOM
        batch_sizes = [1, 2, 4, 8][:max_batch] if max_batch >= 4 else [1, 2][:max_batch]
        
        for batch_size in batch_sizes:
            try:
                self._create_paged_attention_graph(batch_size)
            except Exception as e:
                print(f"Paged attention graph creation failed for batch size {batch_size}: {e}")
    
    def _setup_standard_graphs(self):
        """Setup CUDA graphs for standard attention."""
        max_batch = min(self.config.max_batch_size, 4)  # Cap at 4 to avoid OOM
        batch_sizes = [1, 2, 4, 8][:max_batch] if max_batch >= 4 else [1, 2][:max_batch]
        
        for batch_size in batch_sizes:
            try:
                self._create_standard_graph(batch_size)
            except Exception as e:
                print(f"Standard graph creation failed for batch size {batch_size}: {e}")
    
    def _create_paged_attention_graph(self, batch_size: int):
        """Create CUDA graph for paged attention."""
        try:
            graph = torch.cuda.CUDAGraph()
            
            # Use smaller tensors to avoid memory issues
            max_seq_len = min(512, self.config.max_sequence_length)
            max_blocks = max_seq_len // self.config.block_size
            
            # Input tensors
            input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=self.device)
            positions = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=self.device)
            
            # Paged attention tensors
            slot_mapping = torch.zeros(batch_size, max_seq_len, dtype=torch.int32, device=self.device)
            context_lens = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
            block_tables = torch.zeros(batch_size, max_blocks, dtype=torch.int32, device=self.device)
            
            # Pre-allocate output tensor
            vocab_size = getattr(self.model.config, 'vocab_size', 32000)
            static_logits = torch.zeros(batch_size, max_seq_len, vocab_size, device=self.device, dtype=self.config.dtype)
            
            # Capture graph with a simple forward pass
            with torch.cuda.graph(graph):
                outputs = self.model(input_ids, attention_mask=None)
                static_logits.zero_()
                static_logits.copy_(outputs.logits)
            
            # Store graph and tensors
            self.cuda_graphs[batch_size] = {
                'graph': graph,
                'input_ids': input_ids,
                'positions': positions,
                'slot_mapping': slot_mapping,
                'context_lens': context_lens,
                'block_tables': block_tables,
                'logits': static_logits
            }
        except Exception as e:
            print(f"Failed to create paged attention graph for batch size {batch_size}: {e}")
    
    def _create_standard_graph(self, batch_size: int):
        """Create CUDA graph for standard attention."""
        try:
            graph = torch.cuda.CUDAGraph()
            max_seq_len = 512  # Or get from model config
            
            # Pre-allocate tensors
            input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=self.device)
            attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=self.device)
            
            # Pre-allocate output tensor
            vocab_size = getattr(self.model.config, 'vocab_size', 32000)
            static_logits = torch.zeros(batch_size, max_seq_len, vocab_size, device=self.device, dtype=self.config.dtype)
            
            # Capture graph
            with torch.cuda.graph(graph):
                outputs = self.model(input_ids, attention_mask=attention_mask)
                static_logits.zero_()
                static_logits.copy_(outputs.logits)
            
            self.cuda_graphs[batch_size] = {
                'graph': graph,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'logits': static_logits
            }
        except Exception as e:
            print(f"Failed to create standard graph for batch size {batch_size}: {e}")
    
    def generate(
        self, 
        prompt: Union[str, List[str]], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate text from prompt(s).
        
        Args:
            prompt: Input text prompt(s)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text(s)
        """
        # Handle single prompt
        if isinstance(prompt, str):
            # Use paged attention if enabled
            if self.config.use_paged_attention:
                return self._generate_with_paged_attention(prompt, max_tokens, temperature, top_k, top_p, **kwargs)
            
            # Standard generation
            return self._generate_standard(prompt, max_tokens, temperature, top_k, top_p, **kwargs)
        
        # Handle batch prompts
        batch_kwargs = {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            **kwargs
        }
        return self.batch_generate(prompt, **batch_kwargs)
    
    def _generate_with_paged_attention(self, prompt: str, max_tokens: Optional[int], 
                                     temperature: Optional[float], top_k: Optional[int], 
                                     top_p: Optional[float], **kwargs) -> str:
        """Generate text using paged attention."""
        # Tokenize input
        input_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        sequence_id = f"seq_{id(prompt)}"  # Unique sequence ID
        
        # Allocate memory blocks
        try:
            block_ids = self.paged_attention_manager.allocate_sequence(sequence_id, len(input_tokens))
            
            # Generate with paged attention
            result = self._generate_paged_attention_tokens(sequence_id, input_tokens, max_tokens, temperature, top_k, top_p, **kwargs)
            return result
            
        finally:
            # Clean up memory
            self.paged_attention_manager.deallocate_sequence(sequence_id)
    
    def _generate_paged_attention_tokens(self, sequence_id: str, input_tokens: List[int], 
                                       max_length: Optional[int], temperature: Optional[float],
                                       top_k: Optional[int], top_p: Optional[float], **kwargs) -> str:
        """Generate tokens using paged attention mechanism."""
        # Get block mapping (kept for future integration)
        block_mapping = self.paged_attention_manager.get_block_mapping(sequence_id)
        
        # Prepare inputs and attention mask
        input_ids = torch.tensor([input_tokens], device=self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        
        # Use config defaults if not specified
        max_gen_length = max_length or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        
        generated_tokens: List[int] = []
        past_key_values = None
        
        for _ in range(max_gen_length):
            # Paged-attention generation currently relies on ``past_key_values``
            # after the prefill step. The CUDA-graph path only captures logits,
            # so it cannot drive this decode loop correctly yet.
            if past_key_values is None:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            else:
                last_token = input_ids[:, -1:]
                outputs = self.model(
                    input_ids=last_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            next_token_logits = logits[:, -1, :]
            
            # Sample via centralized sampler
            next_token = self.sampler.sample_token(
                next_token_logits,
                {
                    'temperature': temperature,
                    'top_k': top_k,
                    'top_p': top_p,
                    'repetition_penalty': self.config.repetition_penalty,
                }
            ).view(1, 1)
            
            token_id = int(next_token.item())
            generated_tokens.append(token_id)
            
            # Append token and grow attention mask
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            
            # Stop on EOS
            if self.config.eos_token_id is not None and token_id == self.config.eos_token_id:
                break
        
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        if isinstance(decoded, str) and decoded.startswith(" ".join(map(str, input_tokens))):
            decoded = decoded[len(" ".join(map(str, input_tokens))):].lstrip()
        return decoded
    
    def _generate_standard(self, prompt: str, max_tokens: Optional[int], 
                          temperature: Optional[float], top_k: Optional[int], 
                          top_p: Optional[float], **kwargs) -> str:
        """Standard text generation."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        max_tokens = max_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        
        with torch.no_grad():
            output_ids = self._generate_tokens(
                input_ids, 
                attention_mask, 
                max_tokens, 
                temperature, 
                top_k, 
                top_p,
                **kwargs
            )

        # Decode only generated portion to avoid including prompt
        prompt_len = input_ids.shape[1]
        gen_only = output_ids[0][prompt_len:]
        decoded = self.tokenizer.decode(gen_only, skip_special_tokens=True)
        # Defensive: some tokenizers re-emit the prompt prefix when decoding;
        # strip it so callers reliably get just the completion.
        if isinstance(decoded, str) and decoded.startswith(prompt):
            decoded = decoded[len(prompt):].lstrip()
        return decoded
        
    def _generate_tokens(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        **kwargs
    ) -> torch.Tensor:
        """Generate tokens using the model with automatic prefix caching."""
        
        cached_kv, prefix_len = self.prefix_cache.get_cached_prefix(input_ids)

        if cached_kv is not None:
            past_key_values = cached_kv
            generated_ids = input_ids[:, :prefix_len].clone()
            residual_ids = input_ids[:, prefix_len:]
            # Prefill token-by-token to guarantee KV compatibility
            if residual_ids.numel() > 0:
                for step_idx in range(residual_ids.shape[1]):
                    step_token = residual_ids[:, step_idx:step_idx+1]
                    out = self.model(
                        input_ids=step_token,
                        attention_mask=None,
                        past_key_values=past_key_values,
                        use_cache=True,
                        **kwargs
                    )
                    past_key_values = out.past_key_values
                    generated_ids = torch.cat([generated_ids, step_token], dim=1)
        else:
            past_key_values = None
            generated_ids = input_ids.clone()
            prefix_len = 0
        
        for step in range(max_tokens):
            use_cache = (self.config.use_cache and not self.config.use_torch_compile)  # Honor use_cache flag
            if past_key_values is not None and use_cache:
                model_input_ids = generated_ids[:, -1:]
                current_attention_mask = None
            else:
                model_input_ids = generated_ids
                current_attention_mask = attention_mask

            outputs = self.model(
                input_ids=model_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values if use_cache else None,
                use_cache=use_cache,
                **kwargs
            )
            
            if step == 0 and prefix_len == 0:
                self.prefix_cache.cache_prefix(input_ids, outputs.past_key_values)
            
            next_token_logits = outputs.logits[:, -1, :]
            
            next_token_id = self.sampler.sample_token(
                next_token_logits,
                {
                    'temperature': temperature,
                    'top_k': top_k,
                    'top_p': top_p
                },
                generated_tokens=self.tokenizer.convert_ids_to_tokens(generated_ids[0].tolist()) if False else None
            )
            
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(1)], dim=1)
            
            # Keep original attention mask; when using past we pass None to avoid shape mismatches
            
            if use_cache:
                past_key_values = outputs.past_key_values
            
            if self.config.eos_token_id is not None and next_token_id.item() == self.config.eos_token_id:
                break
                
        return generated_ids
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts using batch processing."""
        return self.batch_processor.process_batch(self, prompts, **kwargs)
        
    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generate text with streaming output."""
        if not self.config.enable_streaming:
            raise ValueError("Streaming not enabled")
        return self.streaming_generator.stream_generate(prompt, **kwargs)
    
    def continuous_batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text using continuous batching."""
        if not self.config.enable_continuous_batching:
            raise ValueError("Continuous batching not enabled")
        
        # Add all requests
        request_ids = []
        for i, prompt in enumerate(prompts):
            request_ids.append(
                self.continuous_batcher.add_request(f"req_{i}", prompt, **kwargs)
            )
        
        # Get results
        return self.continuous_batcher.get_results(request_ids)
    
    def _can_use_cuda_graphs(self, batch_size: int) -> bool:
        """Check if CUDA graphs can be used for given batch size."""
        return (self.config.use_cuda_graphs and 
                hasattr(self, 'cuda_graphs') and 
                batch_size in self.cuda_graphs)
    
    def _forward_with_cuda_graph(self, input_ids: torch.Tensor, block_mapping: Optional[Dict] = None):
        """Forward pass using CUDA graphs."""
        batch_size = input_ids.shape[0]
        
        if batch_size not in self.cuda_graphs:
            # Fallback to standard forward pass
            return self.model(input_ids)
        
        graph_data = self.cuda_graphs[batch_size]
        
        # Copy input data to graph tensors
        if self.config.use_paged_attention and block_mapping is not None:
            # Paged attention graph
            seq_len = input_ids.shape[1]
            graph_data['input_ids'][:batch_size, :seq_len].copy_(input_ids)
            graph_data['context_lens'][:batch_size].fill_(seq_len)
            # Update block tables and slot mappings if needed
        else:
            # Standard graph
            seq_len = input_ids.shape[1]
            graph_data['input_ids'][:batch_size, :seq_len].copy_(input_ids)
            graph_data['attention_mask'][:batch_size, :seq_len].fill_(1)
        
        # Replay the graph
        graph_data['graph'].replay()
        
        # Extract outputs - return the captured logits
        seq_len = input_ids.shape[1]
        logits_slice = graph_data['logits'][:batch_size, :seq_len]
        return type('Output', (), {'logits': logits_slice})()
    
    def get_kv_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get KV cache statistics."""
        return self.kv_cache.get_stats() if self.kv_cache else None
        
    def clear_kv_cache(self):
        """Clear the KV cache."""
        if self.kv_cache:
            self.kv_cache.clear()
    
    def start_continuous_batching(self):
        """Start continuous batching if configured."""
        if self.continuous_batcher:
            self.continuous_batcher.start()
    
    def stop_continuous_batching(self):
        """Stop continuous batching if running."""
        if self.continuous_batcher:
            self.continuous_batcher.stop()
    
    def add_continuous_request(self, prompt: str, max_tokens: int = 50, **kwargs) -> str:
        """Add a request to continuous batching.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Request ID
        """
        if not self.continuous_batcher:
            raise RuntimeError("Continuous batching not configured")
        
        # Generate a unique request ID
        request_id = f"req_{id(prompt)}_{int(time.time() * 1000)}"
        return self.continuous_batcher.add_request(request_id, prompt, max_tokens, **kwargs)
    
    def get_continuous_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get result from continuous batching.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Result dictionary or None if not completed
        """
        if not self.continuous_batcher:
            raise RuntimeError("Continuous batching not configured")
        
        return self.continuous_batcher.get_result(request_id)
    
    def get_continuous_stats(self) -> Dict[str, Any]:
        """Get continuous batching statistics."""
        if not self.continuous_batcher:
            raise RuntimeError("Continuous batching not configured")
        
        return self.continuous_batcher.get_stats()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = {
            'torch_compile_enabled': self.config.use_torch_compile,
            'cuda_graphs_enabled': self.config.use_cuda_graphs,
            'paged_attention_enabled': self.config.use_paged_attention,
            'flash_attention_enabled': self.config.use_flash_attention,
            'prefix_caching_enabled': True,  # Always enabled
            'tensor_parallel_enabled': self.config.enable_tensor_parallel,
        }
        
        if self.config.use_cuda_graphs and hasattr(self, 'cuda_graphs'):
            stats['cuda_graphs'] = {
                'available_batch_sizes': list(self.cuda_graphs.keys()),
                'num_graphs': len(self.cuda_graphs)
            }
        
        if self.config.use_paged_attention and self.paged_attention_manager:
            stats['paged_attention'] = self.paged_attention_manager.get_stats()
        
        if self.prefix_cache:
            stats['prefix_cache'] = self.prefix_cache.get_stats()
        
        return stats
    
    def cleanup(self):
        """Cleanup resources."""
        if self.parallelism is not None:
            self.parallelism.cleanup_distributed()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return Qwen3Factory.get_model_info(self.config.model_path)
    
    def estimate_memory_usage(self) -> Dict[str, Any]:
        """Estimate memory usage."""
        return Qwen3Factory.estimate_memory_usage(self.config.model_path, self.config.dtype)
