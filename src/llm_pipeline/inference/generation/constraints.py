"""Generation constraints and controls."""

import torch
import re
from typing import List, Optional, Dict, Any, Union


class GenerationConstraints:
    """Generation constraints and controls."""
    
    def __init__(
        self,
        stop_tokens: Optional[List[str]] = None,
        max_length: int = 2048,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 3,
        min_length: int = 1,
        bad_words: Optional[List[str]] = None,
        force_words: Optional[List[str]] = None
    ):
        """Initialize generation constraints.
        
        Args:
            stop_tokens: List of stop tokens/sequences
            max_length: Maximum generation length
            repetition_penalty: Penalty for repeated tokens
            no_repeat_ngram_size: Size of n-grams to avoid repeating
            min_length: Minimum generation length
            bad_words: Words/phrases to avoid
            force_words: Words/phrases to encourage
        """
        self.stop_tokens = stop_tokens or ["<|endoftext|>", "<|end|>"]
        self.max_length = max_length
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.min_length = min_length
        self.bad_words = bad_words or []
        self.force_words = force_words or []
        
        # Compiled regex patterns for efficiency
        self.stop_patterns = [re.compile(re.escape(token)) for token in self.stop_tokens]
        self.bad_word_patterns = [re.compile(re.escape(word)) for word in self.bad_words]
        
    def apply_stop_tokens(self, text: str) -> str:
        """Apply stop token constraints.
        
        Args:
            text: Generated text
            
        Returns:
            Text with stop tokens applied
        """
        for pattern in self.stop_patterns:
            match = pattern.search(text)
            if match:
                text = text[:match.start()]
                break
        return text
        
    def apply_repetition_penalty(
        self, 
        logits: torch.Tensor, 
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply repetition penalty to logits.
        
        Args:
            logits: Current logits
            input_ids: Input token IDs
            generated_ids: Previously generated token IDs
            **kwargs: Additional parameters
            
        Returns:
            Modified logits
        """
        if self.repetition_penalty == 1.0:
            return logits
            
        # Combine input and generated tokens
        all_tokens = torch.cat([input_ids.flatten(), generated_ids.flatten()])
        
        # Create penalty mask
        penalty_mask = torch.zeros_like(logits)
        
        # Apply penalty to repeated tokens
        for token_id in all_tokens:
            if token_id < logits.size(-1):
                penalty_mask[..., token_id] = self.repetition_penalty
                
        # Apply penalty
        logits = logits / penalty_mask
        
        return logits
        
    def apply_no_repeat_ngram(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply no-repeat n-gram constraint.
        
        Args:
            logits: Current logits
            generated_ids: Previously generated token IDs
            
        Returns:
            Modified logits
        """
        if len(generated_ids) < self.no_repeat_ngram_size:
            return logits
            
        # Get last n-gram
        last_ngram = generated_ids[-self.no_repeat_ngram_size:]
        
        # Set probability to 0 for tokens that would create repeated n-grams
        for i, token_id in enumerate(last_ngram):
            if token_id < logits.size(-1):
                logits[..., token_id] = float('-inf')
                
        return logits
        
    def apply_bad_words(self, text: str) -> str:
        """Apply bad words filter.
        
        Args:
            text: Generated text
            
        Returns:
            Filtered text
        """
        for pattern in self.bad_word_patterns:
            text = pattern.sub("[REDACTED]", text)
        return text
        
    def check_min_length(self, generated_length: int) -> bool:
        """Check if minimum length constraint is satisfied.
        
        Args:
            generated_length: Current generation length
            
        Returns:
            True if minimum length is satisfied
        """
        return generated_length >= self.min_length
        
    def check_max_length(self, generated_length: int) -> bool:
        """Check if maximum length constraint is exceeded.
        
        Args:
            generated_length: Current generation length
            
        Returns:
            True if maximum length is exceeded
        """
        return generated_length >= self.max_length
        
    def should_stop_generation(
        self, 
        generated_text: str, 
        generated_length: int
    ) -> bool:
        """Check if generation should stop.
        
        Args:
            generated_text: Currently generated text
            generated_length: Current generation length
            
        Returns:
            True if generation should stop
        """
        # Check max length
        if self.check_max_length(generated_length):
            return True
            
        # Check stop tokens
        for pattern in self.stop_patterns:
            if pattern.search(generated_text):
                return True
                
        return False
        
    def get_constraint_info(self) -> Dict[str, Any]:
        """Get information about current constraints.
        
        Returns:
            Dictionary with constraint information
        """
        return {
            "stop_tokens": self.stop_tokens,
            "max_length": self.max_length,
            "min_length": self.min_length,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "bad_words_count": len(self.bad_words),
            "force_words_count": len(self.force_words)
        }
        
    def update_constraints(self, **kwargs):
        """Update constraint parameters.
        
        Args:
            **kwargs: New constraint parameters
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
        # Recompile patterns if needed
        if 'stop_tokens' in kwargs:
            self.stop_patterns = [re.compile(re.escape(token)) for token in self.stop_tokens]
        if 'bad_words' in kwargs:
            self.bad_word_patterns = [re.compile(re.escape(word)) for word in self.bad_words]
