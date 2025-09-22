"""Unit tests for inference engine."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch

from llm_pipeline.inference.core.engine import InferenceEngine, InferenceConfig, MergeStrategy
from llm_pipeline.inference.core.kv_cache import KVCache
from llm_pipeline.inference.core.memory_manager import MemoryManager
from llm_pipeline.inference.core.batch_processor import BatchProcessor
from llm_pipeline.inference.core.context_manager import ContextManager
from llm_pipeline.inference.generation.sampling import SamplingStrategy
from llm_pipeline.inference.generation.constraints import GenerationConstraints
from llm_pipeline.inference.generation.streaming import StreamingGenerator
from llm_pipeline.inference.merging.quantized_merger import QuantizedMerger


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.eos_token_id = 2
        self.vocab_size = 1000
        
    def encode(self, text, return_tensors=None):
        """Mock encode method."""
        # Simple word-based encoding
        words = text.split()
        token_ids = [hash(word) % self.vocab_size for word in words]
        if return_tensors == "pt":
            return torch.tensor([token_ids])
        return token_ids
        
    def decode(self, token_ids, skip_special_tokens=True):
        """Mock decode method."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and len(token_ids) > 1:
            # Handle batch dimension
            if isinstance(token_ids[0], list):
                token_ids = token_ids[0]  # Take first sequence in batch
        return f"decoded_{token_ids}"
        
    def __len__(self):
        return self.vocab_size


class MockModel:
    """Mock model for testing."""
    
    def __init__(self):
        self.config = Mock()
        
    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        """Mock generate method."""
        batch_size, seq_len = input_ids.shape
        # Generate random tokens
        new_tokens = torch.randint(0, 1000, (batch_size, max_new_tokens))
        return torch.cat([input_ids, new_tokens], dim=-1)
        
    def forward(self, input_ids, past_key_values=None, use_cache=False, **kwargs):
        """Mock forward method."""
        batch_size, seq_len = input_ids.shape
        hidden_size = 768
        
        # Mock logits
        logits = torch.randn(batch_size, seq_len, 1000)
        
        # Mock past key values
        past_kv = None
        if use_cache:
            past_kv = torch.randn(2, 2, batch_size, 8, seq_len, hidden_size // 8)
            
        return Mock(logits=logits, past_key_values=past_kv)


class TestInferenceConfig:
    """Test InferenceConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = InferenceConfig()
        
        assert config.max_context_length == 4096
        assert config.max_new_tokens == 512
        assert config.temperature == 1.0
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.repetition_penalty == 1.1
        assert config.use_kv_cache is True
        assert config.kv_cache_size == 1000
        assert config.batch_size == 1
        assert config.memory_budget_gb == 8.0
        assert config.default_merge_strategy == MergeStrategy.AUTO
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = InferenceConfig(
            max_context_length=2048,
            temperature=0.8,
            top_k=40,
            memory_budget_gb=4.0
        )
        
        assert config.max_context_length == 2048
        assert config.temperature == 0.8
        assert config.top_k == 40
        assert config.memory_budget_gb == 4.0


class TestInferenceEngine:
    """Test InferenceEngine."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        return MockModel()
        
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        return MockTokenizer()
        
    @pytest.fixture
    def engine(self, mock_model, mock_tokenizer):
        """Create inference engine."""
        return InferenceEngine(mock_model, mock_tokenizer)
        
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.model is not None
        assert engine.tokenizer is not None
        assert engine.config is not None
        assert engine.paradigm == "full_finetune"  # Default for mock model
        
    def test_paradigm_detection(self, mock_tokenizer):
        """Test paradigm detection."""
        # Test full finetune
        model = MockModel()
        engine = InferenceEngine(model, mock_tokenizer)
        assert engine.paradigm == "full_finetune"
        
        # Test LoRA
        model.lora_modules = {}
        engine = InferenceEngine(model, mock_tokenizer)
        assert engine.paradigm == "lora"
        
        # Test QLoRA
        model.quantization_config = {}
        engine = InferenceEngine(model, mock_tokenizer)
        assert engine.paradigm == "qlora"
        
        # Test quantized full
        delattr(model, 'lora_modules')
        engine = InferenceEngine(model, mock_tokenizer)
        assert engine.paradigm == "quantized_full"
        
    def test_generate_basic(self, engine):
        """Test basic text generation."""
        prompt = "Hello world"
        
        with patch.object(engine, '_generate_without_cache') as mock_generate:
            mock_generate.return_value = torch.tensor([1, 2, 3, 4, 5])
            
            result = engine.generate(prompt)
            
            assert result == "decoded_[1, 2, 3, 4, 5]"
            mock_generate.assert_called_once()
            
    def test_generate_with_parameters(self, engine):
        """Test generation with custom parameters."""
        prompt = "Test prompt"
        
        with patch.object(engine, '_generate_without_cache') as mock_generate:
            mock_generate.return_value = torch.tensor([1, 2, 3])
            
            result = engine.generate(
                prompt,
                max_tokens=50,
                temperature=0.8,
                top_k=40,
                top_p=0.9
            )
            
            assert result == "decoded_[1, 2, 3]"
            
    def test_stream_generate(self, engine):
        """Test streaming generation."""
        prompt = "Test prompt"
        
        with patch.object(engine.streaming_generator, 'stream') as mock_stream:
            mock_stream.return_value = iter(["token1", "token2", "token3"])
            
            tokens = list(engine.stream_generate(prompt))
            
            assert tokens == ["token1", "token2", "token3"]
            mock_stream.assert_called_once()
            
    def test_batch_generate(self, engine):
        """Test batch generation."""
        prompts = ["prompt1", "prompt2", "prompt3"]
        
        with patch.object(engine.batch_processor, 'process_batch') as mock_batch:
            mock_batch.return_value = ["result1", "result2", "result3"]
            
            results = engine.batch_generate(prompts)
            
            assert results == ["result1", "result2", "result3"]
            mock_batch.assert_called_once()
            
    def test_merge_strategy_management(self, engine):
        """Test merge strategy management."""
        # Test setting default strategy
        engine.set_default_merge_strategy(MergeStrategy.QUANTIZED_4BIT)
        assert engine.config.default_merge_strategy == MergeStrategy.QUANTIZED_4BIT
        
        # Test getting available strategies
        strategies = engine.get_available_merge_strategies()
        assert len(strategies) == 6  # All strategies including AUTO
        assert "quantized_4bit" in strategies
        
    def test_performance_metrics(self, engine):
        """Test performance metrics."""
        metrics = engine.get_performance_metrics()
        
        assert "tokens_generated" in metrics
        assert "inference_time" in metrics
        assert "memory_used" in metrics
        assert "cache_hits" in metrics
        assert "cache_misses" in metrics
        
        assert metrics["tokens_generated"] == 0
        assert metrics["inference_time"] == 0.0
        
    def test_cache_management(self, engine):
        """Test cache management."""
        # Test clearing cache
        engine.clear_cache()
        
        metrics = engine.get_performance_metrics()
        assert metrics["cache_hits"] == 0
        assert metrics["cache_misses"] == 0


class TestKVCache:
    """Test KV cache functionality."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = KVCache(max_size=100, cache_type="lru")
        
        assert cache.max_size == 100
        assert cache.cache_type == "lru"
        assert len(cache.cache) == 0
        
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["cache_size"] == 0
        
    def test_cache_operations(self):
        """Test basic cache operations."""
        cache = KVCache(max_size=3)
        
        # Test storing and retrieving
        kv_states = {"key": torch.randn(2, 2, 8, 64), "value": torch.randn(2, 2, 8, 64)}
        cache.store("key1", kv_states)
        
        retrieved = cache.get("key1")
        assert retrieved is not None
        assert "key" in retrieved
        assert "value" in retrieved
        
        # Test cache miss
        retrieved = cache.get("key2")
        assert retrieved is None
        
    def test_cache_eviction(self):
        """Test cache eviction."""
        cache = KVCache(max_size=2, cache_type="lru")
        
        # Fill cache beyond capacity
        cache.store("key1", {"data": torch.randn(10)})
        cache.store("key2", {"data": torch.randn(10)})
        cache.store("key3", {"data": torch.randn(10)})  # Should evict key1
        
        # Check that key1 was evicted
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
        
        stats = cache.get_stats()
        assert stats["evictions"] == 1
        
    def test_cache_key_generation(self):
        """Test cache key generation."""
        cache = KVCache()
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        key = cache.create_cache_key(input_ids)
        
        # Key should be deterministic
        key2 = cache.create_cache_key(input_ids)
        assert key == key2
        
        # Different input should produce different key
        input_ids2 = torch.tensor([[1, 2, 3, 4, 6]])
        key3 = cache.create_cache_key(input_ids2)
        assert key != key3
        
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = KVCache(max_size=10)
        
        # Generate some cache activity
        kv_states = {"data": torch.randn(10)}
        
        # Cache hit
        cache.store("key1", kv_states)
        cache.get("key1")
        
        # Cache miss
        cache.get("key2")
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["total_requests"] == 2
        assert stats["hit_rate"] == 0.5


class TestSamplingStrategy:
    """Test sampling strategies."""
    
    def test_greedy_sampling(self):
        """Test greedy sampling."""
        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
        
        result = SamplingStrategy.greedy(logits)
        
        assert result.item() == 1  # Index of max value (5.0)
        
    def test_temperature_sampling(self):
        """Test temperature sampling."""
        logits = torch.tensor([[1.0, 2.0, 1.0, 1.0]])
        
        # Test with temperature = 1.0 (should be same as softmax)
        result = SamplingStrategy.temperature(logits, temperature=1.0)
        assert result.item() in range(logits.size(-1))
        
        # Test with temperature = 0 (should be greedy)
        result = SamplingStrategy.temperature(logits, temperature=0.0)
        assert result.item() == 1  # Index of max value
        
        # Test with high temperature
        result = SamplingStrategy.temperature(logits, temperature=10.0)
        assert result.item() in range(logits.size(-1))
        
    def test_top_k_sampling(self):
        """Test top-k sampling."""
        logits = torch.tensor([[10.0, 5.0, 3.0, 1.0, 0.0]])
        
        # Test with k=3
        result = SamplingStrategy.top_k(logits, k=3)
        assert result.item() in [0, 1, 2]  # Should be in top-3
        
        # Test with k=1 (should be greedy)
        result = SamplingStrategy.top_k(logits, k=1)
        assert result.item() == 0  # Should be max value
        
    def test_top_p_sampling(self):
        """Test top-p sampling."""
        logits = torch.tensor([[10.0, 5.0, 3.0, 1.0, 0.0]])
        
        # Test with p=0.9
        result = SamplingStrategy.top_p(logits, p=0.9)
        assert result.item() in range(logits.size(-1))
        
        # Test with p=1.0 (should include all tokens)
        result = SamplingStrategy.top_p(logits, p=1.0)
        assert result.item() in range(logits.size(-1))
        
    def test_apply_sampling_strategy(self):
        """Test applying sampling strategies by name."""
        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
        
        # Test greedy
        result = SamplingStrategy.apply_sampling_strategy(logits, "greedy")
        assert result.item() == 1
        
        # Test temperature
        result = SamplingStrategy.apply_sampling_strategy(logits, "temperature", temperature=1.0)
        assert result.item() in range(logits.size(-1))
        
        # Test invalid strategy
        with pytest.raises(ValueError):
            SamplingStrategy.apply_sampling_strategy(logits, "invalid_strategy")


class TestGenerationConstraints:
    """Test generation constraints."""
    
    def test_constraints_initialization(self):
        """Test constraints initialization."""
        constraints = GenerationConstraints(
            stop_tokens=["<|end|>", "\n\n"],
            max_length=1024,
            repetition_penalty=1.2
        )
        
        assert constraints.stop_tokens == ["<|end|>", "\n\n"]
        assert constraints.max_length == 1024
        assert constraints.repetition_penalty == 1.2
        
    def test_stop_token_application(self):
        """Test stop token application."""
        constraints = GenerationConstraints(stop_tokens=["<|end|>", "STOP"])
        
        # Test with stop token
        text = "This is a test<|end|>more text"
        result = constraints.apply_stop_tokens(text)
        assert result == "This is a test"
        
        # Test without stop token
        text = "This is a test without stop"
        result = constraints.apply_stop_tokens(text)
        assert result == text
        
    def test_repetition_penalty(self):
        """Test repetition penalty application."""
        constraints = GenerationConstraints(repetition_penalty=1.5)
        
        logits = torch.tensor([[1.0, 2.0, 1.0, 1.0]])
        input_ids = torch.tensor([[0, 1]])  # Tokens 0 and 1
        generated_ids = torch.tensor([2])   # Token 2
        
        result = constraints.apply_repetition_penalty(logits, input_ids, generated_ids)
        
        # Tokens 0, 1, 2 should have reduced logits due to repetition penalty
        assert result[0, 0] < logits[0, 0]  # Token 0 penalized
        assert result[0, 1] < logits[0, 1]  # Token 1 penalized
        assert result[0, 2] < logits[0, 2]  # Token 2 penalized
        
    def test_length_constraints(self):
        """Test length constraints."""
        constraints = GenerationConstraints(min_length=5, max_length=10)
        
        # Test min length
        assert constraints.check_min_length(5) is True
        assert constraints.check_min_length(4) is False
        
        # Test max length
        assert constraints.check_max_length(10) is True
        assert constraints.check_max_length(11) is True  # >= max_length
        
    def test_should_stop_generation(self):
        """Test stop generation logic."""
        constraints = GenerationConstraints(
            stop_tokens=["<|end|>"],
            max_length=10
        )
        
        # Test stop token
        assert constraints.should_stop_generation("test<|end|>", 5) is True
        
        # Test max length
        assert constraints.should_stop_generation("test", 10) is True
        
        # Test continue generation
        assert constraints.should_stop_generation("test", 5) is False


class TestQuantizedMerger:
    """Test quantized merger."""
    
    def test_merger_initialization(self):
        """Test merger initialization."""
        merger = QuantizedMerger(quantization_type="nf4")
        
        assert merger.quantization_type == "nf4"
        assert merger.get_quality_estimate() == 0.8
        
    def test_memory_usage_estimation(self):
        """Test memory usage estimation."""
        merger = QuantizedMerger()
        
        # Mock model
        model = Mock()
        model.parameters.return_value = [torch.randn(1000, 1000) for _ in range(10)]
        
        lora_adapter = {"test": torch.randn(100, 100)}
        
        memory = merger.get_memory_usage(model, lora_adapter)
        
        # Should return a positive number
        assert memory > 0
        assert isinstance(memory, float)
        
    def test_input_validation(self):
        """Test input validation."""
        merger = QuantizedMerger()
        
        # Valid inputs
        assert merger.validate_inputs(Mock(), Mock()) is True
        
        # Invalid inputs
        assert merger.validate_inputs(None, Mock()) is False
        assert merger.validate_inputs(Mock(), None) is False
        assert merger.validate_inputs(None, None) is False
        
    def test_merge_info(self):
        """Test merge information."""
        merger = QuantizedMerger()
        
        info = merger.get_merge_info()
        
        assert info["merger_type"] == "QuantizedMerger"
        assert info["quality_estimate"] == 0.8
        assert info["supports_precision"] is True
        assert info["requires_requantization"] is False


if __name__ == "__main__":
    pytest.main([__file__])
