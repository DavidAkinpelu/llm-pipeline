"""Dynamic batching for efficient multi-request handling."""

import torch
import time
from typing import List, Dict, Any, Optional, Callable, Literal
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from ..utils.padding import configure_tokenizer_padding


@dataclass
class BatchConfig:
    """Configuration for dynamic batching.
    
    Args:
        max_batch_size: Maximum number of requests in a batch
        max_wait_time: Maximum wait time in seconds before processing batch
        padding_strategy: Padding strategy - "longest" or "max_length"
    """
    max_batch_size: int = 8
    max_wait_time: float = 0.1 
    padding_strategy: Literal["longest", "max_length"] = "longest"


class DynamicBatcher:
    """Dynamic batcher for efficient request processing."""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize dynamic batcher.
        
        Args:
            config: Batching configuration
        """
        self.config = config or BatchConfig()
        self.pending_requests = []
        self.lock = threading.Lock()
        self.executor = None
        
    def add_request(self, request_id: str, prompt: str, **kwargs) -> str:
        """Add a request to the batch queue.
        
        Args:
            request_id: Unique request identifier
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Request ID
        """
        with self.lock:
            self.pending_requests.append({
                'id': request_id,
                'prompt': prompt,
                'kwargs': kwargs,
                'timestamp': time.time()
            })
        return request_id
        
    def process_batch(self, engine, max_wait_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """Process pending requests in a batch.
        
        Args:
            engine: Inference engine
            max_wait_time: Maximum wait time for batching
            
        Returns:
            List of results
        """
        max_wait = max_wait_time or self.config.max_wait_time
        
        with self.lock:
            if not self.pending_requests:
                return []
                
            current_time = time.time()
            ready_requests = []
            
            for req in self.pending_requests:
                if len(ready_requests) >= self.config.max_batch_size:
                    break
                if current_time - req['timestamp'] >= max_wait or len(ready_requests) == 0:
                    ready_requests.append(req)
                    
            for req in ready_requests:
                self.pending_requests.remove(req)
                
        if not ready_requests:
            return []
            
        return self._process_request_batch(engine, ready_requests)
        
    def _process_request_batch(self, engine, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of requests.
        
        Args:
            engine: Inference engine
            requests: List of requests to process
            
        Returns:
            List of results
        """
        groups: Dict[str, Dict[str, Any]] = {}
        for req in requests:
            key = str(sorted(req['kwargs'].items()))
            groups.setdefault(key, {'kwargs': req['kwargs'], 'reqs': []})
            groups[key]['reqs'].append(req)

        id_to_text: Dict[str, str] = {}
        for group in groups.values():
            prompts = [r['prompt'] for r in group['reqs']]
            gen_texts = engine.batch_generate(prompts, **group['kwargs'])
            for r, text in zip(group['reqs'], gen_texts):
                id_to_text[r['id']] = text
    
        batch_results = []
        for req in requests:
            batch_results.append({
                'id': req['id'],
                'prompt': req['prompt'],
                'generated_text': id_to_text.get(req['id'], ''),
                'timestamp': time.time()
            })
            
        return batch_results
        
    def get_pending_count(self) -> int:
        """Get number of pending requests."""
        with self.lock:
            return len(self.pending_requests)
            
    def clear_pending(self):
        """Clear all pending requests."""
        with self.lock:
            self.pending_requests.clear()
            
    def shutdown(self):
        """Shutdown the batcher."""
        self.executor.shutdown(wait=True)


class BatchProcessor:
    """Simple batch processor for synchronous batching."""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize batch processor.
        
        Args:
            config: Batching configuration
        """
        self.config = config or BatchConfig()
        
    def process_batch(self, engine, prompts: List[str], **kwargs) -> List[str]:
        """Process multiple prompts in a single batch with proper padding.
        
        Args:
            engine: Inference engine
            prompts: List of prompts to process
            **kwargs: Generation parameters
            
        Returns:
            List of generated texts
        """
        if not prompts:
            return []
            
        if len(prompts) > self.config.max_batch_size:
            prompts = prompts[:self.config.max_batch_size]
            
        if len(prompts) == 1:
            return [engine.generate(prompts[0], **kwargs)]
        
        # Use paged attention if enabled
        if engine.config.use_paged_attention:
            return self.process_batch_with_paged_attention(engine, prompts, **kwargs)
        
        # Standard batch processing
        return self.process_batch_standard(engine, prompts, **kwargs)
    
    def process_batch_with_paged_attention(self, engine, prompts: List[str], **kwargs) -> List[str]:
        """Process batch using paged attention."""
        # Allocate sequences for all prompts
        sequence_allocations = {}
        try:
            # Allocate memory for all sequences
            for i, prompt in enumerate(prompts):
                tokens = engine.tokenizer.encode(prompt, add_special_tokens=True)
                sequence_id = f"batch_seq_{i}"
                block_ids = engine.paged_attention_manager.allocate_sequence(sequence_id, len(tokens))
                sequence_allocations[sequence_id] = {
                    'prompt': prompt,
                    'tokens': tokens,
                    'block_ids': block_ids
                }
            
            # Process batch with paged attention
            results = self._process_paged_attention_batch(engine, sequence_allocations, **kwargs)
            return results
            
        finally:
            # Clean up all allocated sequences
            for sequence_id in sequence_allocations:
                engine.paged_attention_manager.deallocate_sequence(sequence_id)
    
    def _process_paged_attention_batch(self, engine, sequence_allocations: Dict, **kwargs) -> List[str]:
        """Process batch using paged attention mechanism."""
        results = []
        
        for sequence_id, allocation in sequence_allocations.items():
            # Get block mapping
            block_mapping = engine.paged_attention_manager.get_block_mapping(sequence_id)
            
            # Generate text for this sequence
            result = engine._generate_paged_attention_tokens(
                sequence_id, allocation['tokens'], 
                kwargs.get('max_length'), kwargs.get('temperature'),
                kwargs.get('top_k'), kwargs.get('top_p'), **kwargs
            )
            results.append(result)
        
        return results
    
    def process_batch_standard(self, engine, prompts: List[str], **kwargs) -> List[str]:
        """Standard batch processing."""
            
        tokenized_inputs = []
        original_lengths = []
        
        # Ensure pad token is set
        if engine.tokenizer.pad_token_id is None:
            engine.tokenizer.pad_token = engine.tokenizer.eos_token
            
        original_padding_side = configure_tokenizer_padding(
            engine.tokenizer, 
            engine.model.config
        )
        
        for prompt in prompts:
            # Tokenize with special tokens
            tokens = engine.tokenizer.encode(prompt, add_special_tokens=True)
            tokenized_inputs.append(tokens)
            original_lengths.append(len(tokens))
        
        # Find max length for padding
        max_input_length = max(original_lengths)
        
        # Apply sequence length limit
        max_new_tokens = kwargs.get('max_tokens', kwargs.get('max_length', 50))
        max_total_length = min(max_input_length + max_new_tokens, 
                              getattr(engine.model.config, 'max_position_embeddings', 2048))
        
        # Truncate if necessary
        if max_input_length > max_total_length - max_new_tokens:
            max_input_length = max_total_length - max_new_tokens
            for i, tokens in enumerate(tokenized_inputs):
                if len(tokens) > max_input_length:
                    tokenized_inputs[i] = tokens[:max_input_length]
                    original_lengths[i] = max_input_length
        
        padded_inputs = []
        attention_masks = []
        
        for tokens in tokenized_inputs:
            padded = tokens + [engine.tokenizer.pad_token_id] * (max_input_length - len(tokens))
            attention_mask = [1] * len(tokens) + [0] * (max_input_length - len(tokens))
            
            padded_inputs.append(padded)
            attention_masks.append(attention_mask)
        
        input_ids = torch.tensor(padded_inputs, device=engine.device, dtype=torch.long)
        attention_mask = torch.tensor(attention_masks, device=engine.device, dtype=torch.long)
        
        # Extract sampling args and avoid passing duplicates via **kwargs
        sample_temperature = kwargs.get('temperature')
        if sample_temperature is None:
            sample_temperature = getattr(engine.config, 'temperature', 1.0)
        sample_top_p = kwargs.get('top_p')
        if sample_top_p is None:
            sample_top_p = getattr(engine.config, 'top_p', 1.0)
        sample_top_k = kwargs.get('top_k')
        if sample_top_k is None:
            sample_top_k = getattr(engine.config, 'top_k', 50)
        filtered_kwargs = dict(kwargs)
        for k in ['temperature', 'top_p', 'top_k', 'max_tokens', 'max_length']:
            filtered_kwargs.pop(k, None)

        outputs = self._generate_batch_with_sampler(
            engine, input_ids, attention_mask, max_new_tokens,
            temperature=sample_temperature,
            top_p=sample_top_p,
            top_k=sample_top_k,
            **filtered_kwargs
        )
        
        results = []
        for i, output in enumerate(outputs):
            generated_tokens = output[original_lengths[i]:]

            generated_text = engine.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results.append(generated_text)
        
        # Restore original padding side
        if original_padding_side is not None:
            engine.tokenizer.padding_side = original_padding_side
        
        return results
        
    def process_with_padding(self, engine, prompts: List[str], **kwargs) -> List[str]:
        """Process prompts with padding for efficient batching."""
        return self.process_batch(engine, prompts, **kwargs)
    
    def _generate_batch_with_sampler(
        self, 
        engine, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """Generate tokens using engine's internal sampler for batched inputs.
        
        Args:
            engine: Inference engine with sampler
            input_ids: Batched input token IDs [batch_size, seq_len]
            attention_mask: Batched attention mask [batch_size, seq_len]
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token sequences [batch_size, seq_len + new_tokens]
        """
        batch_size = input_ids.shape[0]
        generated_ids = input_ids.clone()
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=engine.device)
        eos_id = engine.config.eos_token_id if engine.config.eos_token_id is not None else engine.tokenizer.eos_token_id
        pad_id = engine.config.pad_token_id if engine.config.pad_token_id is not None else engine.tokenizer.pad_token_id
        
        # Sampling config override
        sampling_override = {
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'repetition_penalty': kwargs.get('repetition_penalty', 1.0)
        }
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                use_cache = not engine.config.use_torch_compile  # Disable cache with torch.compile
                
                # For first step, use full input; for subsequent steps, use only last token if using cache
                model_input_ids = generated_ids[:, -1:] if past_key_values is not None and use_cache else generated_ids
                current_attention_mask = attention_mask if past_key_values is None else attention_mask
                
                outputs = engine.model(
                    input_ids=model_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values if use_cache else None,
                    use_cache=use_cache
                )
                
                # Get next token logits for all batch elements
                next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Sample next tokens for each batch element
                next_token_ids = engine.sampler.sample_token(next_token_logits, sampling_override)
                
                # Respect finished: force pad for completed sequences
                if pad_id is not None:
                    next_token_ids = torch.where(finished, torch.tensor(pad_id, device=engine.device, dtype=next_token_ids.dtype), next_token_ids)
                
                # Update generated sequences
                generated_ids = torch.cat([generated_ids, next_token_ids.unsqueeze(1)], dim=1)
                
                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=engine.device, dtype=torch.long)
                ], dim=1)
                
                # Update past key values if using cache
                if use_cache:
                    past_key_values = outputs.past_key_values
                
                # Early stop per sequence on EOS token
                if eos_id is not None:
                    eos_mask = (next_token_ids == eos_id)
                    finished = finished | eos_mask
                    if torch.all(finished):
                        break
        
        return generated_ids
