"""Context window management for long sequences."""

import torch
from typing import Union, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ContextConfig:
    """Configuration for context management."""
    max_context_length: int = 4096
    sliding_window_size: int = 2048
    sliding_window_stride: int = 1024
    truncation_strategy: str = "right"  # "left", "right", "middle"
    preserve_prompt: bool = True
    preserve_generation: bool = True


class ContextManager:
    """Context window manager for handling long sequences."""
    
    def __init__(self, config: Optional[ContextConfig] = None):
        """Initialize context manager.
        
        Args:
            config: Context management configuration
        """
        self.config = config or ContextConfig()
        
    def truncate_input(
        self, 
        input_ids: torch.Tensor, 
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """Truncate input to fit within context window.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum length (uses config if None)
            
        Returns:
            Truncated input IDs
        """
        max_length = max_length or self.config.max_context_length
        
        if input_ids.size(-1) <= max_length:
            return input_ids
            
        if self.config.truncation_strategy == "right":
            return input_ids[..., :max_length]
        elif self.config.truncation_strategy == "left":
            return input_ids[..., -max_length:]
        elif self.config.truncation_strategy == "middle":
            return self._truncate_middle(input_ids, max_length)
        else:
            raise ValueError(f"Unknown truncation strategy: {self.config.truncation_strategy}")
            
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit.
        
        Args:
            text: Input text
            max_tokens: Maximum number of tokens
            
        Returns:
            Truncated text
        """
        # Simple word-based truncation
        words = text.split()
        if len(words) <= max_tokens:
            return text
            
        if self.config.truncation_strategy == "right":
            truncated_words = words[:max_tokens]
        elif self.config.truncation_strategy == "left":
            truncated_words = words[-max_tokens:]
        elif self.config.truncation_strategy == "middle":
            start = (len(words) - max_tokens) // 2
            truncated_words = words[start:start + max_tokens]
        else:
            raise ValueError(f"Unknown truncation strategy: {self.config.truncation_strategy}")
            
        return " ".join(truncated_words)
        
    def sliding_window(
        self, 
        text: str, 
        window_size: Optional[int] = None,
        stride: Optional[int] = None
    ) -> List[str]:
        """Create sliding windows over text.
        
        Args:
            text: Input text
            window_size: Size of sliding window
            stride: Stride between windows
            
        Returns:
            List of text windows
        """
        window_size = window_size or self.config.sliding_window_size
        stride = stride or self.config.sliding_window_stride
        
        words = text.split()
        if len(words) <= window_size:
            return [text]
            
        windows = []
        for i in range(0, len(words) - window_size + 1, stride):
            window = words[i:i + window_size]
            windows.append(" ".join(window))
            
        return windows
        
    def split_long_sequence(
        self, 
        input_ids: torch.Tensor,
        max_length: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Split long sequence into manageable chunks.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum length per chunk
            
        Returns:
            List of token ID chunks
        """
        max_length = max_length or self.config.max_context_length
        
        if input_ids.size(-1) <= max_length:
            return [input_ids]
            
        chunks = []
        for i in range(0, input_ids.size(-1), max_length):
            chunk = input_ids[..., i:i + max_length]
            chunks.append(chunk)
            
        return chunks
        
    def merge_contexts(
        self, 
        contexts: List[str],
        separator: str = "\n\n"
    ) -> str:
        """Merge multiple contexts into single text.
        
        Args:
            contexts: List of context strings
            separator: Separator between contexts
            
        Returns:
            Merged context string
        """
        return separator.join(contexts)
        
    def estimate_context_length(self, text: str) -> int:
        """Estimate context length in tokens.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4
        
    def check_context_fit(
        self, 
        text: str, 
        max_length: Optional[int] = None
    ) -> bool:
        """Check if text fits within context window.
        
        Args:
            text: Input text
            max_length: Maximum context length
            
        Returns:
            True if text fits within context window
        """
        max_length = max_length or self.config.max_context_length
        estimated_length = self.estimate_context_length(text)
        return estimated_length <= max_length
        
    def _truncate_middle(self, input_ids: torch.Tensor, max_length: int) -> torch.Tensor:
        """Truncate from middle, keeping both ends.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum length
            
        Returns:
            Truncated input IDs
        """
        total_length = input_ids.size(-1)
        if total_length <= max_length:
            return input_ids
            
        # Keep beginning and end, remove middle
        keep_from_start = max_length // 2
        keep_from_end = max_length - keep_from_start
        
        start_part = input_ids[..., :keep_from_start]
        end_part = input_ids[..., -keep_from_end:]
        
        # Concatenate parts
        return torch.cat([start_part, end_part], dim=-1)
        
    def get_context_info(self) -> dict:
        """Get context management information.
        
        Returns:
            Dictionary with context management info
        """
        return {
            "max_context_length": self.config.max_context_length,
            "sliding_window_size": self.config.sliding_window_size,
            "sliding_window_stride": self.config.sliding_window_stride,
            "truncation_strategy": self.config.truncation_strategy,
            "preserve_prompt": self.config.preserve_prompt,
            "preserve_generation": self.config.preserve_generation
        }
