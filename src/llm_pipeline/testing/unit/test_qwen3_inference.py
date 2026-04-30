"""Unit tests for Qwen3 inference engine."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from llm_pipeline.inference.qwen3_engine import Qwen3InferenceEngine, Qwen3InferenceConfig
from llm_pipeline.models.qwen3 import Qwen3Factory


class TestQwen3InferenceConfig:
    """Test Qwen3InferenceConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = Qwen3InferenceConfig(model_path="/test/path")
        
        assert config.model_path == "/test/path"
        assert config.device == "cuda:0"
        assert config.dtype == torch.float16
        assert config.use_cache is True
        assert config.enable_tensor_parallel is False
        assert config.tensor_parallel_size == 1
        assert config.max_length == 2048
        assert config.max_new_tokens == 512
        assert config.temperature == 1.0
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.repetition_penalty == 1.0
        assert config.do_sample is True
        assert config.max_batch_size == 8
        assert config.enable_streaming is True
        assert config.enable_continuous_batching is False
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = Qwen3InferenceConfig(
            model_path="/custom/path",
            device="cpu",
            dtype=torch.float32,
            use_cache=False,
            enable_tensor_parallel=True,
            tensor_parallel_size=4,
            max_length=4096,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            repetition_penalty=1.1,
            do_sample=False,
            max_batch_size=16,
            enable_streaming=False,
            enable_continuous_batching=True
        )
        
        assert config.model_path == "/custom/path"
        assert config.device == "cpu"
        assert config.dtype == torch.float32
        assert config.use_cache is False
        assert config.enable_tensor_parallel is True
        assert config.tensor_parallel_size == 4
        assert config.max_length == 4096
        assert config.max_new_tokens == 1024
        assert config.temperature == 0.7
        assert config.top_p == 0.8
        assert config.top_k == 40
        assert config.repetition_penalty == 1.1
        assert config.do_sample is False
        assert config.max_batch_size == 16
        assert config.enable_streaming is False
        assert config.enable_continuous_batching is True


class TestQwen3InferenceEngine:
    """Test Qwen3InferenceEngine."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        return Qwen3InferenceConfig(
            model_path="/test/path",
            device="cpu",
            dtype=torch.float32,
            enable_tensor_parallel=False
        )
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.parameters.return_value = [torch.randn(10, 10)]
        model.return_value = {
            'last_hidden_state': torch.randn(2, 10, 128),
            'past_key_values': None
        }
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        tokenizer.decode.return_value = "Test prompt Generated text"
        tokenizer.eos_token_id = 151643
        return tokenizer
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_engine_initialization(self, mock_factory, mock_config, mock_model, mock_tokenizer):
        """Test engine initialization."""
        # Mock factory
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        # Mock other components
        with patch('llm_pipeline.inference.qwen3_engine.Sampler'), \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor'), \
             patch('llm_pipeline.inference.qwen3_engine.StreamingGenerator'):
            
            engine = Qwen3InferenceEngine(mock_config)
            
            assert engine.config == mock_config
            assert engine.model == mock_model
            assert engine.tokenizer == mock_tokenizer
            assert engine.device == torch.device("cpu")
            mock_factory.create_model_and_tokenizer.assert_called_once()
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_engine_with_tensor_parallel(self, mock_factory, mock_model, mock_tokenizer):
        """Test engine with tensor parallelism."""
        config = Qwen3InferenceConfig(
            model_path="/test/path",
            enable_tensor_parallel=True,
            tensor_parallel_size=2
        )
        
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        # Mock parallelism components
        with patch('llm_pipeline.inference.qwen3_engine.ParallelismFactory') as mock_parallelism, \
             patch('llm_pipeline.inference.qwen3_engine.Sampler'), \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor'), \
             patch('llm_pipeline.inference.qwen3_engine.StreamingGenerator'):
            
            mock_parallelism.create_parallelism.return_value = Mock()
            mock_parallelism.create_parallelism.return_value.setup_distributed.return_value = None
            mock_parallelism.create_parallelism.return_value.wrap_model.return_value = mock_model
            
            engine = Qwen3InferenceEngine(config)
            
            assert engine.tp_engine is not None
            mock_parallelism.create_parallelism.assert_called_once()
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_generate_single(self, mock_factory, mock_config, mock_model, mock_tokenizer):
        """Test single text generation."""
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler') as mock_sampler, \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor'), \
             patch('llm_pipeline.inference.qwen3_engine.StreamingGenerator'):
            
            # Mock sampler
            mock_sampler_instance = Mock()
            mock_sampler_instance.sample.return_value = torch.tensor([[42]])
            mock_sampler.return_value = mock_sampler_instance
            
            engine = Qwen3InferenceEngine(mock_config)
            
            # Mock the _generate_tokens method
            with patch.object(engine, '_generate_tokens') as mock_generate:
                mock_generate.return_value = torch.tensor([[1, 2, 3, 42]])
                
                result = engine.generate("Test prompt")
                
                assert result == "Generated text"
                mock_tokenizer.decode.assert_called()
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_generate_batch(self, mock_factory, mock_config, mock_model, mock_tokenizer):
        """Test batch text generation."""
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler') as mock_sampler, \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor') as mock_batch_processor, \
             patch('llm_pipeline.inference.qwen3_engine.StreamingGenerator'):
            
            # Mock batch processor
            mock_batch_processor_instance = Mock()
            mock_batch_processor_instance.process_batch.return_value = ["Result 1", "Result 2"]
            mock_batch_processor.return_value = mock_batch_processor_instance
            
            mock_sampler.return_value = Mock()
            
            engine = Qwen3InferenceEngine(mock_config)
            
            prompts = ["Prompt 1", "Prompt 2"]
            results = engine.generate(prompts)
            
            assert results == ["Result 1", "Result 2"]
            mock_batch_processor_instance.process_batch.assert_called_once()
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_generate_with_custom_params(self, mock_factory, mock_config, mock_model, mock_tokenizer):
        """Test generation with custom parameters."""
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler') as mock_sampler, \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor'), \
             patch('llm_pipeline.inference.qwen3_engine.StreamingGenerator'):
            
            # Mock sampler creation with custom config
            mock_sampler_instance = Mock()
            mock_sampler_instance.sample.return_value = torch.tensor([[42]])
            mock_sampler.return_value = mock_sampler_instance
            
            engine = Qwen3InferenceEngine(mock_config)
            
            with patch.object(engine, '_generate_tokens') as mock_generate:
                mock_generate.return_value = torch.tensor([[1, 2, 3, 42]])
                
                result = engine.generate(
                    "Test prompt",
                    max_new_tokens=100,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=40
                )
                
                assert result == "Generated text"
                # Verify that a new sampler was created with custom parameters
                mock_sampler.assert_called()
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_stream_generate(self, mock_factory, mock_config, mock_model, mock_tokenizer):
        """Test streaming generation."""
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler'), \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor'), \
             patch('llm_pipeline.inference.qwen3_engine.StreamingGenerator') as mock_streaming:
            
            # Mock streaming generator
            mock_streaming_instance = Mock()
            mock_streaming_instance.stream_generate.return_value = ["chunk1", "chunk2", "chunk3"]
            mock_streaming.return_value = mock_streaming_instance
            
            engine = Qwen3InferenceEngine(mock_config)
            
            chunks = list(engine.stream_generate("Test prompt"))
            
            assert chunks == ["chunk1", "chunk2", "chunk3"]
            mock_streaming_instance.stream_generate.assert_called_once()
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_stream_generate_not_enabled(self, mock_factory, mock_config, mock_model, mock_tokenizer):
        """Test streaming generation when not enabled."""
        config = Qwen3InferenceConfig(
            model_path="/test/path",
            enable_streaming=False
        )
        
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler'), \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor'):
            
            engine = Qwen3InferenceEngine(config)
            
            with pytest.raises(ValueError, match="Streaming not enabled"):
                list(engine.stream_generate("Test prompt"))
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_continuous_batch_generate(self, mock_factory, mock_config, mock_model, mock_tokenizer):
        """Test continuous batch generation."""
        config = Qwen3InferenceConfig(
            model_path="/test/path",
            enable_continuous_batching=True
        )
        
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler'), \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor'), \
             patch('llm_pipeline.inference.qwen3_engine.StreamingGenerator'), \
             patch('llm_pipeline.inference.qwen3_engine.ContinuousBatcher') as mock_continuous:
            
            # Mock continuous batcher
            mock_continuous_instance = Mock()
            mock_continuous_instance.add_request.side_effect = ["req_0", "req_1"]
            mock_continuous_instance.get_results.return_value = ["Result 1", "Result 2"]
            mock_continuous.return_value = mock_continuous_instance
            
            engine = Qwen3InferenceEngine(config)
            
            requests = ["Request 1", "Request 2"]
            results = engine.continuous_batch_generate(requests)
            
            assert results == ["Result 1", "Result 2"]
            assert mock_continuous_instance.add_request.call_count == 2
            mock_continuous_instance.get_results.assert_called_once_with(["req_0", "req_1"])

    def test_prefix_cache_truncates_cached_kv_to_prefix_length(self):
        """Cached KV state must match the cached prefix length, not the full prompt."""
        from llm_pipeline.inference.caching.prefix_cache import PrefixCache, PrefixCacheConfig

        cache = PrefixCache(PrefixCacheConfig(max_prefix_length=4))
        input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
        past_key_values = tuple(
            (torch.zeros(1, 1, 6, 8), torch.zeros(1, 1, 6, 8))
            for _ in range(2)
        )

        cache.cache_prefix(input_ids, past_key_values)
        cached_kv, prefix_len = cache.get_cached_prefix(torch.tensor([[10, 11, 12, 13, 99]]))

        assert prefix_len == 4
        assert cached_kv is not None
        assert cached_kv[0][0].shape[2] == prefix_len
        assert cached_kv[0][1].shape[2] == prefix_len

    def test_generate_paged_attention_returns_completion_only(self):
        """Paged-attention generation should match standard generate() semantics."""
        engine = object.__new__(Qwen3InferenceEngine)
        engine.device = torch.device("cpu")
        engine.config = Qwen3InferenceConfig(
            model_path="/test/path",
            device="cpu",
            dtype=torch.float32,
            eos_token_id=7,
        )
        engine.paged_attention_manager = Mock()
        engine.paged_attention_manager.get_block_mapping.return_value = {}
        engine.sampler = Mock()
        engine.sampler.sample_token.return_value = torch.tensor([7])
        engine._can_use_cuda_graphs = Mock(return_value=False)
        engine.tokenizer = Mock()
        engine.tokenizer.decode.side_effect = lambda token_ids, skip_special_tokens=True: (
            "prompt plus completion" if list(token_ids) == [10, 11, 7] else "completion"
        )
        engine.model = Mock(return_value=type(
            "Output",
            (),
            {
                "logits": torch.zeros(1, 2, 16),
                "past_key_values": tuple((torch.zeros(1, 1, 2, 4), torch.zeros(1, 1, 2, 4)) for _ in range(1)),
            },
        )())

        out = engine._generate_paged_attention_tokens("seq", [10, 11], 1, None, None, None)

        assert out == "completion"

    def test_paged_attention_generation_skips_incompatible_cuda_graph_path(self):
        """Paged-attention generation should not use CUDA-graph outputs without KV cache."""
        engine = object.__new__(Qwen3InferenceEngine)
        engine.device = torch.device("cpu")
        engine.config = Qwen3InferenceConfig(
            model_path="/test/path",
            device="cpu",
            dtype=torch.float32,
            eos_token_id=7,
            use_cuda_graphs=True,
        )
        engine.paged_attention_manager = Mock()
        engine.paged_attention_manager.get_block_mapping.return_value = {}
        engine.sampler = Mock()
        engine.sampler.sample_token.return_value = torch.tensor([7])
        engine._can_use_cuda_graphs = Mock(return_value=True)
        engine._forward_with_cuda_graph = Mock(side_effect=AssertionError("should not be called"))
        engine.tokenizer = Mock()
        engine.tokenizer.decode.return_value = "completion"
        engine.model = Mock(return_value=type(
            "Output",
            (),
            {
                "logits": torch.zeros(1, 2, 16),
                "past_key_values": tuple((torch.zeros(1, 1, 2, 4), torch.zeros(1, 1, 2, 4)) for _ in range(1)),
            },
        )())

        out = engine._generate_paged_attention_tokens("seq", [10, 11], 1, None, None, None)

        assert out == "completion"
        engine._forward_with_cuda_graph.assert_not_called()
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_continuous_batch_generate_not_enabled(self, mock_factory, mock_config, mock_model, mock_tokenizer):
        """Test continuous batch generation when not enabled."""
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler'), \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor'), \
             patch('llm_pipeline.inference.qwen3_engine.StreamingGenerator'):
            
            engine = Qwen3InferenceEngine(mock_config)
            
            with pytest.raises(ValueError, match="Continuous batching not enabled"):
                engine.continuous_batch_generate(["Request 1"])
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_cleanup(self, mock_factory, mock_config, mock_model, mock_tokenizer):
        """Test cleanup."""
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler'), \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor'), \
             patch('llm_pipeline.inference.qwen3_engine.StreamingGenerator'), \
             patch('torch.cuda.empty_cache') as mock_empty_cache:
            
            engine = Qwen3InferenceEngine(mock_config)
            engine.cleanup()
            
            # Should not call CUDA cache clear for CPU
            mock_empty_cache.assert_not_called()
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_cleanup_with_parallelism(self, mock_factory, mock_config, mock_model, mock_tokenizer):
        """Test cleanup with parallelism."""
        config = Qwen3InferenceConfig(
            model_path="/test/path",
            enable_tensor_parallel=True,
            tensor_parallel_size=2
        )
        
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        # Mock parallelism
        mock_parallelism = Mock()
        mock_parallelism.cleanup_distributed.return_value = None
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler'), \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor'), \
             patch('llm_pipeline.inference.qwen3_engine.StreamingGenerator'), \
             patch('llm_pipeline.inference.qwen3_engine.ParallelismFactory') as mock_parallelism_factory:
            
            mock_parallelism_factory.create_parallelism.return_value = mock_parallelism
            mock_parallelism.wrap_model.return_value = mock_model
            
            engine = Qwen3InferenceEngine(config)
            engine.cleanup()
            
            mock_parallelism.cleanup_distributed.assert_called_once()
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_get_model_info(self, mock_factory, mock_config, mock_model, mock_tokenizer):
        """Test getting model info."""
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        mock_factory.get_model_info.return_value = {
            'model_type': 'qwen3',
            'vocab_size': 32000,
            'hidden_size': 768
        }
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler'), \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor'), \
             patch('llm_pipeline.inference.qwen3_engine.StreamingGenerator'):
            
            engine = Qwen3InferenceEngine(mock_config)
            info = engine.get_model_info()
            
            assert info['model_type'] == 'qwen3'
            assert info['vocab_size'] == 32000
            assert info['hidden_size'] == 768
            mock_factory.get_model_info.assert_called_once_with("/test/path")
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_estimate_memory_usage(self, mock_factory, mock_config, mock_model, mock_tokenizer):
        """Test memory usage estimation."""
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        mock_factory.estimate_memory_usage.return_value = {
            'total_parameters': 1000000,
            'model_memory_mb': 100,
            'model_memory_gb': 0.1,
            'recommended_gpu_memory_gb': 0.15
        }
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler'), \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor'), \
             patch('llm_pipeline.inference.qwen3_engine.StreamingGenerator'):
            
            engine = Qwen3InferenceEngine(mock_config)
            memory_info = engine.estimate_memory_usage()
            
            assert memory_info['total_parameters'] == 1000000
            assert memory_info['model_memory_mb'] == 100
            assert memory_info['model_memory_gb'] == 0.1
            assert memory_info['recommended_gpu_memory_gb'] == 0.15
            mock_factory.estimate_memory_usage.assert_called_once_with("/test/path", torch.float32)


if __name__ == "__main__":
    pytest.main([__file__])
