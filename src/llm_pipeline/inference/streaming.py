"""Streaming generation for real-time text output."""

import torch
from typing import Iterator, List, Dict, Any, Optional
import time


class StreamingGenerator:
    """Streaming text generator for real-time output."""
    
    def __init__(self, engine, chunk_size: int = 1):
        """Initialize streaming generator.
        
        Args:
            engine: Inference engine
            chunk_size: Number of tokens to yield at once
        """
        self.engine = engine
        self.chunk_size = chunk_size
        
    def stream_generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Iterator[str]:
        """Generate text with streaming output.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Generation parameters
            
        Yields:
            Generated text chunks
        """
        inputs = self.engine.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.engine.device)
        attention_mask = inputs["attention_mask"].to(self.engine.device)
        
        max_tokens = max_tokens or self.engine.config.max_new_tokens
        
        generated_tokens = []
        past_key_values = None
        
        for step in range(max_tokens):
            outputs = self.engine.model(
                input_ids=input_ids[:, -1:] if past_key_values is not None else input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                **kwargs
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            
            # Respect sampler overrides
            next_token_id = self.engine.sampler.sample_token(
                next_token_logits,
                {
                    'temperature': temperature if temperature is not None else self.engine.config.temperature,
                    'top_k': top_k if top_k is not None else self.engine.config.top_k,
                    'top_p': top_p if top_p is not None else self.engine.config.top_p,
                }
            )
            
            generated_tokens.append(next_token_id.item())
            
            if len(generated_tokens) >= self.chunk_size:
                chunk_text = self.engine.tokenizer.decode(generated_tokens[-self.chunk_size:], skip_special_tokens=True)
                if chunk_text.strip():
                    yield chunk_text
                    
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(1)], dim=1)
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones((attention_mask.shape[0], 1), device=self.engine.device)
            ], dim=1)
            past_key_values = outputs.past_key_values
            
            if next_token_id.item() == self.engine.tokenizer.eos_token_id:
                break
                
        if len(generated_tokens) % self.chunk_size != 0:
            remaining_tokens = generated_tokens[-(len(generated_tokens) % self.chunk_size):]
            if remaining_tokens:
                chunk_text = self.engine.tokenizer.decode(remaining_tokens, skip_special_tokens=True)
                if chunk_text.strip():
                    yield chunk_text
                    


class BufferedStreamingGenerator(StreamingGenerator):
    """Buffered streaming generator with configurable buffer size."""
    
    def __init__(self, engine, buffer_size: int = 5, chunk_size: int = 1):
        """Initialize buffered streaming generator.
        
        Args:
            engine: Inference engine
            buffer_size: Number of tokens to buffer before yielding
            chunk_size: Number of tokens to yield at once
        """
        super().__init__(engine, chunk_size)
        self.buffer_size = buffer_size
        
    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generate text with buffered streaming."""
        buffer = []
        
        for token_chunk in super().stream_generate(prompt, **kwargs):
            buffer.append(token_chunk)
            
            if len(buffer) >= self.buffer_size:
                yield ''.join(buffer)
                buffer.clear()
                
        if buffer:
            yield ''.join(buffer)
