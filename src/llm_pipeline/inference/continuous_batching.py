"""Continuous batching for ultra-low latency inference."""

import torch
import time
import threading
from typing import Dict, List, Any, Optional, Iterator, Union, Tuple

try:
    from transformers.cache_utils import Cache
    CACHE_TYPE = Union[Cache, Tuple[Tuple[torch.Tensor, torch.Tensor], ...], None]
except ImportError:
    # Fallback for older versions or if Cache is not available
    CACHE_TYPE = Union[Tuple[Tuple[torch.Tensor, torch.Tensor], ...], Any, None]
from dataclasses import dataclass
from collections import defaultdict
import queue


@dataclass
class ContinuousBatchConfig:
    """Configuration for continuous batching.
    
    Args:
        max_batch_size: Maximum number of requests in a continuous batch
        max_sequence_length: Maximum sequence length per request
        max_wait_time: Maximum wait time in seconds before processing requests
        memory_limit_gb: Memory limit in GB for batch processing
        enable_kv_cache: Whether to enable KV caching for efficiency
    """
    max_batch_size: int = 16
    max_sequence_length: int = 2048
    max_wait_time: float = 0.05 
    memory_limit_gb: float = 8.0
    enable_kv_cache: bool = True


class RequestState:
    """State tracking for individual requests."""
    
    def __init__(self, request_id: str, prompt: str, max_tokens: int, **kwargs):
        self.request_id = request_id
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        
        self.input_tokens = []
        self.generated_tokens = []
        self.kv_cache = {}
        self.current_length = 0
        self.is_finished = False
        self.start_time = time.time()
        self.finish_time = None
        
        self.batch_position = -1
        
        # Paged attention fields
        self.block_ids = None
        self.uses_paged_attention = False
        
    def get_total_length(self) -> int:
        """Get total sequence length (input + generated)."""
        return len(self.input_tokens) + len(self.generated_tokens)
    
    def get_generation_length(self) -> int:
        """Get number of generated tokens."""
        return len(self.generated_tokens)
    
    def is_complete(self, eos_token_id: int) -> bool:
        """Check if request is complete.
        
        Args:
            eos_token_id: End-of-sequence token ID.
        """
        if self.is_finished or self.get_generation_length() >= self.max_tokens:
            return True
            
        if self.generated_tokens:
            last_token = self.generated_tokens[-1]
            return last_token == eos_token_id
            
        return False


class ContinuousBatch:
    """Continuous batch state manager."""
    
    def __init__(self, config: ContinuousBatchConfig):
        self.config = config
        self.active_requests: Dict[str, RequestState] = {}
        self.batch_input_ids = None
        self.batch_attention_mask = None
        self.batch_kv_caches = None
        self.max_length = 0
        self.device = None
        
    def add_request(self, request_state: RequestState) -> bool:
        """Add request to active batch."""
        if len(self.active_requests) >= self.config.max_batch_size:
            return False
            
        self.active_requests[request_state.request_id] = request_state
        request_state.batch_position = len(self.active_requests) - 1
        # Invalidate combined KV cache when batch changes
        self.batch_kv_caches = None
        return True
    
    def remove_request(self, request_id: str) -> Optional[RequestState]:
        """Remove completed request from batch."""
        if request_id in self.active_requests:
            request_state = self.active_requests.pop(request_id)
            
            for i, req in enumerate(self.active_requests.values()):
                req.batch_position = i
            # Invalidate combined KV cache when batch changes
            self.batch_kv_caches = None
            return request_state
        return
    
    def update_batch_tensors(self, tokenizer):
        """Update batch tensors for current requests."""
        if not self.active_requests:
            return
        
        # Ensure pad token is set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Find max length among active requests
        self.max_length = max(req.get_total_length() for req in self.active_requests.values())
        
        # Ensure we don't exceed max sequence length
        self.max_length = min(self.max_length, self.config.max_sequence_length)
        
        # Prepare batch tensors
        batch_inputs = []
        attention_masks = []
        
        for req in sorted(self.active_requests.values(), key=lambda x: x.batch_position):
            # Get current sequence (input + generated)
            current_tokens = req.input_tokens + req.generated_tokens
            
            # Pad to max length
            padded_tokens = current_tokens[:self.max_length]
            if len(padded_tokens) < self.max_length:
                padded_tokens.extend([tokenizer.pad_token_id] * (self.max_length - len(padded_tokens)))
            
            # Create attention mask
            attention_mask = self._create_attention_mask(len(current_tokens), self.max_length)
            
            batch_inputs.append(padded_tokens)
            attention_masks.append(attention_mask)
        
        # Convert to tensors
        self.batch_input_ids = torch.tensor(batch_inputs, device=self.device)
        self.batch_attention_mask = torch.tensor(attention_masks, device=self.device)
    
    def _create_attention_mask(self, seq_length: int, max_length: int) -> List[int]:
        mask = [1] * seq_length + [0] * (max_length - seq_length)
        return mask
    
    def update_kv_cache(self, new_kv_cache: CACHE_TYPE) -> None:
        """Update KV cache for all active requests."""
        if not self.config.enable_kv_cache or new_kv_cache is None:
            return

        self.batch_kv_caches = new_kv_cache
    
    def get_active_request_ids(self) -> List[str]:
        """Get list of active request IDs."""
        return list(self.active_requests.keys())
    
    def has_active_requests(self) -> bool:
        """Check if batch has active requests."""
        return len(self.active_requests) > 0
    
    def get_batch_size(self) -> int:
        """Get current batch size."""
        return len(self.active_requests)


class ContinuousBatcher:
    """Continuous batcher for ultra-low latency inference."""
    
    def __init__(self, engine, config: Optional[ContinuousBatchConfig] = None):
        """Initialize continuous batcher.
        
        Args:
            engine: Inference engine
            config: Continuous batching configuration
        """
        self.engine = engine
        self.config = config or ContinuousBatchConfig()
        
        # Request management
        self.pending_queue = queue.Queue()
        self.active_batch = ContinuousBatch(self.config)
        self.active_batch.device = engine.device
        self.completed_requests: Dict[str, RequestState] = {}
        
        # Threading
        self.lock = threading.Lock()
        self.processing_thread = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "active_requests": 0,
            "average_latency": 0.0,
            "throughput_per_second": 0.0
        }
    
    def start(self):
        """Start the continuous batching process."""
        if self.is_running:
            return
            
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._continuous_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop(self):
        """Stop the continuous batching process."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
    
    def add_request(self, request_id: str, prompt: str, max_tokens: int = 50, **kwargs) -> str:
        """Add a request to the continuous batch queue.
        
        Args:
            request_id: Unique request identifier
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Request ID
        """
        # Tokenize input
        input_tokens = self.engine.tokenizer.encode(prompt, add_special_tokens=True)
        
        request_state = RequestState(request_id, prompt, max_tokens, **kwargs)
        request_state.input_tokens = input_tokens
        request_state.current_length = len(input_tokens)
        
        # Allocate paged attention memory if enabled
        if self.engine.config.use_paged_attention:
            try:
                block_ids = self.engine.paged_attention_manager.allocate_sequence(request_id, len(input_tokens))
                request_state.block_ids = block_ids
                request_state.uses_paged_attention = True
            except RuntimeError as e:
                print(f"Failed to allocate paged attention memory for request {request_id}: {e}")
                # Fall back to standard processing
                request_state.uses_paged_attention = False
        
        with self.lock:
            self.pending_queue.put(request_state)
            self.stats["total_requests"] += 1
        
        return request_id
    
    def get_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the result for a completed request.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Result dictionary or None if not completed
        """
        with self.lock:
            if request_id in self.completed_requests:
                request_state = self.completed_requests[request_id]
                
                generated_text = ""
                if request_state.generated_tokens:
                    generated_text = self.engine.tokenizer.decode(
                        request_state.generated_tokens, 
                        skip_special_tokens=True
                    )
                
                return {
                    "request_id": request_id,
                    "prompt": request_state.prompt,
                    "generated_text": generated_text,
                    "generated_tokens": request_state.generated_tokens,
                    "latency_ms": (request_state.finish_time - request_state.start_time) * 1000,
                    "tokens_generated": len(request_state.generated_tokens)
                }
        
        return None

    def get_results(self, request_ids: List[str], timeout_s: float = 5.0) -> List[str]:
        """Block until the requested IDs complete, then return their texts."""
        deadline = time.time() + timeout_s

        while time.time() < deadline:
            with self.lock:
                all_done = all(request_id in self.completed_requests for request_id in request_ids)
            if all_done:
                break

            if not self.is_running:
                self._add_pending_requests()
                if self.active_batch.has_active_requests():
                    self._process_batch_step()
                self._cleanup_completed_requests()

            time.sleep(0.001)

        missing = []
        results = []
        for request_id in request_ids:
            result = self.get_result(request_id)
            if result is None:
                missing.append(request_id)
            else:
                results.append(result["generated_text"])

        if missing:
            raise TimeoutError(f"Timed out waiting for continuous batch results: {missing}")

        return results
    
    def _continuous_processing_loop(self):
        """Main continuous processing loop."""
        while self.is_running:
            try:
                self._add_pending_requests()
                
                if self.active_batch.has_active_requests():
                    self._process_batch_step()
                
                self._cleanup_completed_requests()
                
                time.sleep(0.001)  # 1ms
                
            except Exception as e:
                print(f"Error in continuous processing loop: {e}")
                time.sleep(0.01)  # 10ms on error
    
    def _add_pending_requests(self):
        """Add pending requests to active batch."""
        while not self.pending_queue.empty():
            try:
                request_state = self.pending_queue.get_nowait()
                
                if self.active_batch.add_request(request_state):
                    with self.lock:
                        self.stats["active_requests"] += 1
                else:
                    self.pending_queue.put(request_state)
                    break
                    
            except queue.Empty:
                break
    
    def _process_batch_step(self):
        """Process one step of the active batch."""
        if not self.active_batch.has_active_requests():
            return
        
        self.active_batch.update_batch_tensors(self.engine.tokenizer)
        
        # Forward pass
        with torch.no_grad():
            # Handle KV cache properly - pass None if empty dict
            past_key_values = self.active_batch.batch_kv_caches if self.active_batch.batch_kv_caches else None
            
            # Prefill with full sequences on first step; then last-token decode
            if past_key_values is None:
                model_input_ids = self.active_batch.batch_input_ids
            else:
                model_input_ids = self.active_batch.batch_input_ids[:, -1:]
            
            outputs = self.engine.model(
                input_ids=model_input_ids,
                attention_mask=self.active_batch.batch_attention_mask,
                past_key_values=past_key_values,
                use_cache=self.config.enable_kv_cache,
            )
        
        # Update KV cache
        if self.config.enable_kv_cache and outputs.past_key_values:
            self.active_batch.update_kv_cache(outputs.past_key_values)
        
        # Sample next tokens for each request
        next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        
        for i, request_id in enumerate(self.active_batch.get_active_request_ids()):
            request_state = self.active_batch.active_requests[request_id]
            
            # Sample next token
            next_token = self._sample_token(
                next_token_logits[i:i+1], 
                request_state.kwargs
            )
            
            # Update request state
            request_state.generated_tokens.append(next_token.item())
            
            # Check if request is complete
            eos_token_id = self.engine.tokenizer.eos_token_id
            if eos_token_id is not None and request_state.is_complete(eos_token_id):
                request_state.is_finished = True
                request_state.finish_time = time.time()
    
    def _sample_token(self, logits: torch.Tensor, kwargs: Dict[str, Any]) -> torch.Tensor:
        """Sample next token using centralized sampler."""
        return self.engine.sampler.sample_token(logits, kwargs)
    
    
    def _cleanup_completed_requests(self):
        """Move completed requests to completed_requests dict."""
        completed_ids = []
        
        for request_id, request_state in self.active_batch.active_requests.items():
            if request_state.is_finished:
                # Clean up paged attention memory if used
                if (self.engine.config.use_paged_attention and 
                    hasattr(request_state, 'uses_paged_attention') and 
                    request_state.uses_paged_attention):
                    try:
                        self.engine.paged_attention_manager.deallocate_sequence(request_id)
                    except Exception as e:
                        print(f"Failed to deallocate paged attention memory for request {request_id}: {e}")
                
                completed_ids.append(request_id)
        
        for request_id in completed_ids:
            request_state = self.active_batch.remove_request(request_id)
            if request_state:
                with self.lock:
                    self.completed_requests[request_id] = request_state
                    self.stats["completed_requests"] += 1
                    self.stats["active_requests"] -= 1
                    
                    # Update average latency
                    latency = (request_state.finish_time - request_state.start_time) * 1000
                    total_completed = self.stats["completed_requests"]
                    current_avg = self.stats["average_latency"]
                    self.stats["average_latency"] = (current_avg * (total_completed - 1) + latency) / total_completed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        with self.lock:
            stats_copy = self.stats.copy()
            stats_copy["pending_requests"] = self.pending_queue.qsize()
            stats_copy["active_batch_size"] = self.active_batch.get_batch_size()
            
            # Calculate throughput
            if stats_copy["completed_requests"] > 0:
                total_time = time.time() - min(
                    (req.start_time for req in self.completed_requests.values()), 
                    default=time.time()
                )
                stats_copy["throughput_per_second"] = stats_copy["completed_requests"] / max(total_time, 1.0)
            
            return stats_copy
    
    def clear_completed(self):
        """Clear completed requests to free memory."""
        with self.lock:
            self.completed_requests.clear()
