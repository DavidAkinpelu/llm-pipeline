"""Streaming generation for real-time text output."""

import torch
from typing import Iterator, Optional, Dict, Any
from ..core.kv_cache import KVCache


class StreamingGenerator:
    """Real-time streaming text generator."""
    
    def __init__(self, model, tokenizer, kv_cache: Optional[KVCache] = None):
        """Initialize streaming generator.
        
        Args:
            model: The model to use for generation
            tokenizer: Tokenizer for the model
            kv_cache: Optional KV cache for optimization
        """
        self.model = model
        self.tokenizer = tokenizer
        self.kv_cache = kv_cache
        
    def stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        **kwargs
    ) -> Iterator[str]:
        """Stream generated tokens in real-time.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters
            
        Yields:
            Generated tokens one by one
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Initialize generation state
        generated_ids = torch.empty(0, dtype=torch.long)
        past_key_values = None
        
        for _ in range(max_tokens):
            # Get next token
            next_token, past_key_values = self._get_next_token(
                input_ids=input_ids,
                generated_ids=generated_ids,
                past_key_values=past_key_values,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                **kwargs
            )
            
            # Check for end of sequence
            if next_token == self.tokenizer.eos_token_id:
                break
                
            # Update generated tokens
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)])
            
            # Decode and yield token
            token_text = self.tokenizer.decode(next_token, skip_special_tokens=True)
            yield token_text
            
    def stream_with_stop_tokens(
        self,
        prompt: str,
        max_tokens: int = 100,
        stop_tokens: Optional[list] = None,
        **kwargs
    ) -> Iterator[str]:
        """Stream with stop token checking.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            stop_tokens: List of stop tokens
            **kwargs: Additional generation parameters
            
        Yields:
            Generated tokens one by one
        """
        stop_tokens = stop_tokens or ["<|endoftext|>", "<|end|>"]
        generated_text = ""
        
        for token in self.stream(prompt, max_tokens=max_tokens, **kwargs):
            generated_text += token
            
            # Check for stop tokens
            should_stop = False
            for stop_token in stop_tokens:
                if stop_token in generated_text:
                    should_stop = True
                    # Yield only up to the stop token
                    stop_index = generated_text.find(stop_token)
                    if stop_index >= 0:
                        yield generated_text[:stop_index]
                    break
                    
            if should_stop:
                break
                
            yield token
            
    def _get_next_token(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        past_key_values: Optional[tuple],
        temperature: float,
        top_k: int,
        top_p: float,
        **kwargs
    ) -> tuple[torch.Tensor, tuple]:
        """Get next token from model.
        
        Args:
            input_ids: Input token IDs
            generated_ids: Previously generated token IDs
            past_key_values: Past key-value states
            temperature: Sampling temperature
            top_k: Top-k parameter
            top_p: Top-p parameter
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (next_token, updated_past_key_values)
        """
        # Prepare input for model
        if past_key_values is None:
            # First token - use input_ids
            model_input = input_ids
        else:
            # Subsequent tokens - use last generated token
            model_input = generated_ids[-1:].unsqueeze(0)
            
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=model_input,
                past_key_values=past_key_values,
                use_cache=True,
                **kwargs
            )
            
        # Extract logits and past key values
        logits = outputs.logits[0, -1, :]  # Last token logits
        past_key_values = outputs.past_key_values
        
        # Apply sampling
        next_token = self._sample_token(
            logits, temperature=temperature, top_k=top_k, top_p=top_p
        )
        
        return next_token, past_key_values
        
    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Sample next token from logits.
        
        Args:
            logits: Model output logits
            temperature: Sampling temperature
            top_k: Top-k parameter
            top_p: Top-p parameter
            
        Returns:
            Sampled token ID
        """
        if temperature == 0:
            # Greedy sampling
            return torch.argmax(logits, dim=-1)
            
        # Apply temperature
        logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            filtered_logits = torch.full_like(logits, float('-inf'))
            filtered_logits[top_k_indices] = top_k_logits
            logits = filtered_logits
            
        # Apply top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumsum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Find cutoff
            cutoff = torch.searchsorted(cumsum_probs, top_p, right=True)
            cutoff = max(1, cutoff.item())
            
            # Filter logits
            filtered_logits = torch.full_like(logits, float('-inf'))
            filtered_logits[sorted_indices[:cutoff]] = sorted_logits[:cutoff]
            logits = filtered_logits
            
        # Sample from filtered logits
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token.squeeze()
        
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming generation statistics.
        
        Returns:
            Dictionary with streaming statistics
        """
        return {
            "model_type": type(self.model).__name__,
            "tokenizer_type": type(self.tokenizer).__name__,
            "kv_cache_enabled": self.kv_cache is not None,
            "vocab_size": len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else None
        }
