"""Centralized sampling logic for text generation."""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import math


@dataclass
class SamplingConfig:
    """Configuration for text generation sampling.
    
    Args:
        temperature: Sampling temperature (0.0 = greedy, >1.0 = more random)
        top_k: Top-k sampling - keep only top k tokens (0 = disabled)
        top_p: Top-p (nucleus) sampling - keep tokens with cumulative probability <= top_p (1.0 = disabled)
        repetition_penalty: Repetition penalty (1.0 = no penalty, >1.0 = reduce repetition)
        typical_p: Typical-p sampling - keep tokens with typical probability (1.0 = disabled)
        min_p: Min-p sampling - minimum probability threshold (0.0 = disabled)
        mirostat_tau: Mirostat v2 target entropy
        mirostat_eta: Mirostat v2 learning rate
        do_sample: Whether to use sampling (False = greedy decoding)
        pad_token_id: Padding token ID for batch processing
        eos_token_id: End-of-sequence token ID
    """
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    typical_p: float = 1.0
    min_p: float = 0.0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


class Sampler:
    """Centralized sampling logic for all inference approaches."""
    
    def __init__(self, config: Optional[SamplingConfig] = None):
        """Initialize sampler with configuration.
        
        Args:
            config: Sampling configuration
        """
        self.config = config or SamplingConfig()
        
    def sample_token(
        self, 
        logits: torch.Tensor, 
        config_override: Optional[Dict[str, Any]] = None,
        generated_tokens: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Sample next token using specified strategy.
        
        Args:
            logits: Token logits from model [batch_size, vocab_size]
            config_override: Override sampling parameters
            generated_tokens: Previously generated tokens for repetition penalty
            
        Returns:
            Sampled token IDs [batch_size]
        """
        config = self._merge_config(config_override)
        
        if generated_tokens and config.repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(logits, generated_tokens, config.repetition_penalty)
        
        if config.temperature > 0:
            logits = logits / config.temperature
        else:
            return torch.argmax(logits, dim=-1)
        
        logits = self._apply_top_k(logits, config.top_k)
        logits = self._apply_top_p(logits, config.top_p)
        logits = self._apply_min_p(logits, config.min_p)
        
        if config.typical_p < 1.0:
            logits = self._apply_typical_p(logits, config.typical_p)
        
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)
    
    def _merge_config(self, config_override: Optional[Dict[str, Any]]) -> SamplingConfig:
        if not config_override:
            return self.config
            
        config_dict = {
            'temperature': self.config.temperature,
            'top_k': self.config.top_k,
            'top_p': self.config.top_p,
            'repetition_penalty': self.config.repetition_penalty,
            'typical_p': self.config.typical_p,
            'min_p': self.config.min_p,
            'mirostat_tau': self.config.mirostat_tau,
            'mirostat_eta': self.config.mirostat_eta,
            'do_sample': self.config.do_sample,
            'pad_token_id': self.config.pad_token_id,
            'eos_token_id': self.config.eos_token_id,
        }
        
        config_dict.update(config_override)
        return SamplingConfig(**config_dict)
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_tokens: List[int],
        penalty: float
    ) -> torch.Tensor:
        """Apply CTRL-style repetition penalty to tokens already generated.

        For each previously generated token t:
          if logit[t] > 0: logit[t] /= penalty
          else:            logit[t] *= penalty
        """
        if not generated_tokens or penalty == 1.0:
            return logits

        device = logits.device
        vocab_size = logits.size(-1)
        tokens_tensor = torch.tensor(generated_tokens, device=device)
        tokens_tensor = tokens_tensor[(tokens_tensor >= 0) & (tokens_tensor < vocab_size)]
        if tokens_tensor.numel() == 0:
            return logits

        unique_tokens = torch.unique(tokens_tensor)
        # Select logits for unique tokens: [batch_size, U]
        selected = logits.index_select(dim=-1, index=unique_tokens)
        adjusted = torch.where(selected > 0, selected / penalty, selected * penalty)
        # Scatter back
        logits = logits.clone()
        logits[:, unique_tokens] = adjusted
        return logits
    
    def _apply_top_k(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits.
        
        Args:
            logits: Token logits [batch_size, vocab_size]
            top_k: Number of top tokens to keep
            
        Returns:
            Filtered logits
        """
        if top_k <= 0:
            return logits
            
        top_k = min(top_k, logits.size(-1))
        top_k_logits, _ = torch.topk(logits, top_k, dim=-1)
        min_logits = top_k_logits[:, -1].unsqueeze(-1)
        
        return torch.where(logits < min_logits, torch.full_like(logits, float('-inf')), logits)
    
    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits.
        
        Args:
            logits: Token logits [batch_size, vocab_size]
            top_p: Cumulative probability threshold
            
        Returns:
            Filtered logits
        """
        if top_p >= 1.0:
            return logits
            
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        # Scatter the sorted mask back to original indexing
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits
    
    def _apply_min_p(self, logits: torch.Tensor, min_p: float) -> torch.Tensor:
        """Apply min-p filtering to logits.
        
        Args:
            logits: Token logits [batch_size, vocab_size]
            min_p: Minimum probability threshold (as fraction of max probability)
            
        Returns:
            Filtered logits
        """
        if min_p <= 0.0:
            return logits
            
        # Compute a probability threshold relative to the max probability
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1, keepdim=True)
        threshold = min_p * max_probs
        mask = probs < threshold
        return logits.masked_fill(mask, float('-inf'))
    
    def _apply_typical_p(self, logits: torch.Tensor, typical_p: float) -> torch.Tensor:
        """Apply typical-p filtering to logits.
        
        Args:
            logits: Token logits [batch_size, vocab_size]
            typical_p: Typical probability threshold
            
        Returns:
            Filtered logits
        """
        if typical_p >= 1.0:
            return logits
            
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
        
        typicality_threshold = torch.exp(entropy) * typical_p
        
        information = -log_probs
        
        mask = information <= typicality_threshold
        
        for i in range(mask.size(0)):
            if not mask[i].any():
                max_idx = torch.argmax(logits[i])
                mask[i, max_idx] = True
        
        logits = torch.where(mask, logits, torch.full_like(logits, float('-inf')))
        
        return logits
    
    def sample_mirostat_v2(
        self, 
        logits: torch.Tensor, 
        config_override: Optional[Dict[str, Any]] = None,
        previous_entropy: Optional[float] = None
    ) -> tuple[torch.Tensor, float]:
        """Sample using Mirostat v2 algorithm.
        
        Args:
            logits: Token logits [batch_size, vocab_size]
            config_override: Override sampling parameters
            previous_entropy: Previous entropy value for Mirostat
            
        Returns:
            Tuple of (sampled_token_ids, updated_entropy)
        """
        config = self._merge_config(config_override)
        
        if config.temperature > 0:
            logits = logits / config.temperature
        
        probs = F.softmax(logits, dim=-1)
        
        log_probs = F.log_softmax(logits, dim=-1)
        current_entropy = -torch.sum(probs * log_probs, dim=-1)
        
        if previous_entropy is None:
            previous_entropy = current_entropy.item()
        
        error = current_entropy.item() - config.mirostat_tau
        previous_entropy -= config.mirostat_eta * error
        
        if previous_entropy > 0:
            estimated_k = max(1, int(math.exp(previous_entropy)))
            logits = self._apply_top_k(logits, estimated_k)
        
        probs = F.softmax(logits, dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        return sampled_tokens, previous_entropy
    
    def get_sampling_info(self) -> Dict[str, Any]:
        """Get current sampling configuration information."""
        return {
            'temperature': self.config.temperature,
            'top_k': self.config.top_k,
            'top_p': self.config.top_p,
            'repetition_penalty': self.config.repetition_penalty,
            'typical_p': self.config.typical_p,
            'min_p': self.config.min_p,
            'mirostat_tau': self.config.mirostat_tau,
            'mirostat_eta': self.config.mirostat_eta,
            'do_sample': self.config.do_sample,
        }