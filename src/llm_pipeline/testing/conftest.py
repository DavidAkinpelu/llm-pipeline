"""Pytest configuration and shared fixtures."""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from typing import Dict, Any, Optional


@pytest.fixture
def device():
    """Get available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_configs():
    """Sample configurations for testing."""
    from llm_pipeline.core.config import LoRAConfig, DoRAConfig, RSLoRAConfig
    
    return {
        "lora_small": LoRAConfig(r=4, alpha=8.0, dropout=0.1),
        "lora_medium": LoRAConfig(r=8, alpha=16.0, dropout=0.1),
        "lora_large": LoRAConfig(r=16, alpha=32.0, dropout=0.1),
        "dora": DoRAConfig(r=8, alpha=16.0, magnitude_init="ones"),
        "rslora": RSLoRAConfig(r=8, alpha=16.0),
    }


@pytest.fixture
def mock_llama_model():
    """Create a mock Llama model for testing."""
    class MockLlamaModel(nn.Module):
        def __init__(self, hidden_size=128, num_layers=2, vocab_size=1000):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'self_attn': nn.ModuleDict({
                        'q_proj': nn.Linear(hidden_size, hidden_size),
                        'k_proj': nn.Linear(hidden_size, hidden_size),
                        'v_proj': nn.Linear(hidden_size, hidden_size),
                        'o_proj': nn.Linear(hidden_size, hidden_size),
                    }),
                    'mlp': nn.ModuleDict({
                        'gate_proj': nn.Linear(hidden_size, hidden_size * 4),
                        'up_proj': nn.Linear(hidden_size, hidden_size * 4),
                        'down_proj': nn.Linear(hidden_size * 4, hidden_size),
                    })
                }) for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, input_ids):
            x = self.embed_tokens(input_ids)
            for layer in self.layers:
                # Simple forward pass for testing
                x = x + torch.randn_like(x) * 0.01
            x = self.norm(x)
            return self.lm_head(x)
    
    return MockLlamaModel()


@pytest.fixture
def mock_bert_model():
    """Create a mock BERT model for testing."""
    class MockBERTModel(nn.Module):
        def __init__(self, hidden_size=128, num_layers=2, vocab_size=1000):
            super().__init__()
            self.embeddings = nn.ModuleDict({
                'word_embeddings': nn.Embedding(vocab_size, hidden_size),
                'position_embeddings': nn.Embedding(512, hidden_size),
                'token_type_embeddings': nn.Embedding(2, hidden_size),
                'LayerNorm': nn.LayerNorm(hidden_size),
                'dropout': nn.Dropout(0.1)
            })
            self.encoder = nn.ModuleList([
                nn.ModuleDict({
                    'attention': nn.ModuleDict({
                        'self': nn.ModuleDict({
                            'query': nn.Linear(hidden_size, hidden_size),
                            'key': nn.Linear(hidden_size, hidden_size),
                            'value': nn.Linear(hidden_size, hidden_size),
                            'dense': nn.Linear(hidden_size, hidden_size),
                        })
                    }),
                    'intermediate': nn.ModuleDict({
                        'dense': nn.Linear(hidden_size, hidden_size * 4)
                    }),
                    'output': nn.ModuleDict({
                        'dense': nn.Linear(hidden_size * 4, hidden_size)
                    })
                }) for _ in range(num_layers)
            ])
            self.pooler = nn.ModuleDict({
                'dense': nn.Linear(hidden_size, hidden_size),
                'activation': nn.Tanh()
            })
        
        def forward(self, input_ids):
            x = self.embeddings.word_embeddings(input_ids)
            x = x + self.embeddings.position_embeddings(torch.arange(x.size(1), device=x.device))
            x = self.embeddings.LayerNorm(x)
            x = self.embeddings.dropout(x)
            
            for layer in self.encoder:
                # Simple forward pass for testing
                x = x + torch.randn_like(x) * 0.01
            
            return self.pooler.dense(x[:, 0])
    
    return MockBERTModel()


@pytest.fixture
def sample_input_data():
    """Sample input data for testing."""
    return {
        "input_ids": torch.randint(0, 1000, (2, 10)),
        "attention_mask": torch.ones(2, 10),
        "labels": torch.randint(0, 1000, (2, 10)),
    }


@pytest.fixture
def quantization_configs():
    """Sample quantization configurations for testing."""
    from llm_pipeline.quantization.configs import BnBConfig, AQLMConfig, LoftQConfig
    
    configs = {}
    
    # BitsAndBytes configs
    try:
        configs["bnb_4bit"] = BnBConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        configs["bnb_8bit"] = BnBConfig(load_in_8bit=True)
    except:
        pass
    
    # AQLM configs
    try:
        configs["aqlm_2bit"] = AQLMConfig()
    except:
        pass
    
    # LoftQ configs
    try:
        configs["loftq"] = LoftQConfig(loftq_bits=4, loftq_iter=1, loftq_rank=8)
    except:
        pass
    
    return configs


@pytest.fixture(scope="session")
def test_model_registry():
    """Test model registry with all supported models."""
    from llm_pipeline.core.registry import ModelRegistry
    
    return ModelRegistry


@pytest.fixture
def memory_constraints():
    """Memory constraints for testing."""
    return {
        "max_memory_gb": 8.0,
        "safety_margin": 0.8,
        "batch_size_limits": [1, 2, 4, 8, 16, 32],
        "sequence_length_limits": [128, 256, 512, 1024, 2048],
    }


@pytest.fixture
def performance_benchmarks():
    """Performance benchmark thresholds."""
    return {
        "memory_reduction_lora": 0.5,  # LoRA should use <50% memory of full fine-tuning
        "memory_reduction_quantized": 0.25,  # Quantized should use <25% memory
        "speedup_inference": 1.5,  # Should be at least 1.5x faster
        "accuracy_threshold": 0.95,  # Should maintain >95% accuracy
        "convergence_steps": 100,  # Should converge within 100 steps
    }


# Skip tests based on available dependencies
def pytest_configure(config):
    """Configure pytest with custom markers and skip conditions."""
    
    # Check for optional dependencies
    optional_deps = {
        "bitsandbytes": False,
        "aqlm": False,
        "transformers": True,  # Should always be available
        "torch": True,  # Should always be available
    }
    
    try:
        import bitsandbytes
        optional_deps["bitsandbytes"] = True
    except ImportError:
        pass
    
    try:
        import aqlm
        optional_deps["aqlm"] = True
    except ImportError:
        pass
    
    # Store in config for use in tests
    config.optional_deps = optional_deps




def pytest_runtest_setup(item):
    """Set up tests with skip conditions."""
    
    # Skip GPU tests if no CUDA
    if item.get_closest_marker("gpu"):
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
    
    # Skip quantization tests if dependencies not available
    if item.get_closest_marker("quantization"):
        deps = getattr(item.config, "optional_deps", {})
        if "bnb" in item.name and not deps.get("bitsandbytes", False):
            pytest.skip("BitsAndBytes not available")
        if "aqlm" in item.name and not deps.get("aqlm", False):
            pytest.skip("AQLM not available")


# Cleanup function for tests
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
