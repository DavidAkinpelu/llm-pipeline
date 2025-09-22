"""Sampling strategies for text generation."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union


class SamplingStrategy:
    """Collection of sampling strategies for text generation."""
    
    @staticmethod
    def greedy(logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Greedy sampling - select highest probability token.
        
        Args:
            logits: Model output logits
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Selected token indices
        """
        return torch.argmax(logits, dim=-1)
        
    @staticmethod
    def temperature(
        logits: torch.Tensor, 
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """Temperature sampling.
        
        Args:
            logits: Model output logits
            temperature: Temperature parameter (0 for greedy, > 0 for sampling)
            **kwargs: Additional parameters
            
        Returns:
            Selected token indices
        """
        if temperature == 0:
            # Greedy sampling when temperature is 0
            return torch.argmax(logits, dim=-1)
        elif temperature < 0:
            raise ValueError("Temperature must be non-negative")
            
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Apply softmax and sample
        probs = F.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
        
    @staticmethod
    def top_k(
        logits: torch.Tensor,
        k: int = 50,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """Top-k sampling.
        
        Args:
            logits: Model output logits
            k: Number of top tokens to consider
            temperature: Temperature parameter
            **kwargs: Additional parameters
            
        Returns:
            Selected token indices
        """
        if k <= 0:
            raise ValueError("k must be positive")
            
        # Get top-k logits
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        
        # Apply temperature
        if temperature != 1.0:
            top_k_logits = top_k_logits / temperature
            
        # Sample from top-k
        probs = F.softmax(top_k_logits, dim=-1)
        sampled_indices = torch.multinomial(probs, num_samples=1)
        
        # Map back to original indices
        batch_indices = torch.arange(logits.size(0)).unsqueeze(-1)
        selected_tokens = top_k_indices[batch_indices, sampled_indices].squeeze(-1)
        
        return selected_tokens
        
    @staticmethod
    def top_p(
        logits: torch.Tensor,
        p: float = 0.9,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """Top-p (nucleus) sampling.
        
        Args:
            logits: Model output logits
            p: Cumulative probability threshold
            temperature: Temperature parameter
            **kwargs: Additional parameters
            
        Returns:
            Selected token indices
        """
        if not 0 < p <= 1:
            raise ValueError("p must be in (0, 1]")
            
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # Apply temperature
        if temperature != 1.0:
            sorted_logits = sorted_logits / temperature
            
        # Compute cumulative probabilities
        probs = F.softmax(sorted_logits, dim=-1)
        cumsum_probs = torch.cumsum(probs, dim=-1)
        
        # Find cutoff point
        cutoff_mask = cumsum_probs <= p
        
        # Handle case where all probabilities are needed
        cutoff_mask[:, -1] = True
        
        # Set probabilities to zero for tokens beyond cutoff
        filtered_probs = probs * cutoff_mask.float()
        
        # Renormalize
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        
        # Sample
        sampled_indices = torch.multinomial(filtered_probs, num_samples=1)
        
        # Map back to original indices
        batch_indices = torch.arange(logits.size(0)).unsqueeze(-1)
        selected_tokens = sorted_indices[batch_indices, sampled_indices].squeeze(-1)
        
        return selected_tokens
        
    @staticmethod
    def typical_p(
        logits: torch.Tensor,
        p: float = 0.95,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """Typical-p sampling.
        
        Args:
            logits: Model output logits
            p: Typical probability threshold
            temperature: Temperature parameter
            **kwargs: Additional parameters
            
        Returns:
            Selected token indices
        """
        if not 0 < p <= 1:
            raise ValueError("p must be in (0, 1]")
            
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
            
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Compute negative log probabilities (entropy)
        neg_log_probs = -torch.log(probs + 1e-8)
        
        # Compute entropy
        entropy = (probs * neg_log_probs).sum(dim=-1, keepdim=True)
        
        # Find tokens with typical probability
        typical_mask = torch.abs(neg_log_probs - entropy) < -torch.log(torch.tensor(p))
        
        # Set probabilities to zero for non-typical tokens
        filtered_probs = probs * typical_mask.float()
        
        # Handle case where no tokens are typical
        all_zero = (filtered_probs.sum(dim=-1) == 0).unsqueeze(-1)
        filtered_probs = torch.where(all_zero, probs, filtered_probs)
        
        # Renormalize
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        
        # Sample
        return torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
        
    @staticmethod
    def mirostat(
        logits: torch.Tensor,
        tau: float = 5.0,
        eta: float = 0.1,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """Mirostat sampling.
        
        Args:
            logits: Model output logits
            tau: Target entropy
            eta: Learning rate
            temperature: Temperature parameter
            **kwargs: Additional parameters (may include mu)
            
        Returns:
            Selected token indices
        """
        # Get or initialize mu (target surprise)
        mu = kwargs.get('mu', tau * 2)
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
            
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample token
        selected_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # Compute surprise for selected tokens
        batch_indices = torch.arange(logits.size(0))
        selected_probs = probs[batch_indices, selected_tokens]
        surprise = -torch.log(selected_probs + 1e-8)
        
        # Update mu
        mu = mu - eta * (surprise - tau)
        
        # Store updated mu for next call
        kwargs['mu'] = mu
        
        return selected_tokens
        
    @staticmethod
    def apply_sampling_strategy(
        logits: torch.Tensor,
        strategy: str,
        **kwargs
    ) -> torch.Tensor:
        """Apply specified sampling strategy.
        
        Args:
            logits: Model output logits
            strategy: Sampling strategy name
            **kwargs: Strategy-specific parameters
            
        Returns:
            Selected token indices
        """
        strategies = {
            "greedy": SamplingStrategy.greedy,
            "temperature": SamplingStrategy.temperature,
            "top_k": SamplingStrategy.top_k,
            "top_p": SamplingStrategy.top_p,
            "typical_p": SamplingStrategy.typical_p,
            "mirostat": SamplingStrategy.mirostat,
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
            
        return strategies[strategy](logits, **kwargs)
