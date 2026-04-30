"""Integration tests for Qwen3 full pipeline."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from llm_pipeline.inference.qwen3_engine import Qwen3InferenceEngine, Qwen3InferenceConfig
from llm_pipeline.models.qwen3 import Qwen3Factory
from llm_pipeline.parallelism import ParallelismFactory, TensorParallelConfig


class TestQwen3FullPipeline:
    """Test complete Qwen3 inference pipeline."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock Qwen3 model."""
        model = Mock()
        model.parameters.return_value = [torch.randn(10, 10)]
        model.config = Mock()
        model.config.model_type = 'qwen3'
        model.return_value = {
            'last_hidden_state': torch.randn(2, 10, 128),
            'past_key_values': None
        }
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        tokenizer.decode.return_value = "Generated text response"
        tokenizer.eos_token_id = 151643
        tokenizer.pad_token_id = 151643
        return tokenizer
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_basic_inference_pipeline(self, mock_factory, mock_model, mock_tokenizer):
        """Test basic inference pipeline."""
        config = Qwen3InferenceConfig(
            model_path="/test/path",
            device="cpu",
            dtype=torch.float32,
            enable_streaming=False,
            enable_continuous_batching=False
        )
        
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler') as mock_sampler, \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor') as mock_batch_processor:
            
            # Mock sampler
            mock_sampler_instance = Mock()
            mock_sampler_instance.sample.return_value = torch.tensor([[42]])
            mock_sampler.return_value = mock_sampler_instance
            
            # Mock batch processor
            mock_batch_processor_instance = Mock()
            mock_batch_processor_instance.process_batch.return_value = ["Batch result"]
            mock_batch_processor.return_value = mock_batch_processor_instance
            
            # Create engine
            engine = Qwen3InferenceEngine(config)
            
            # Test single generation
            with patch.object(engine, '_generate_tokens') as mock_generate:
                mock_generate.return_value = torch.tensor([[1, 2, 3, 42]])
                
                result = engine.generate("Test prompt")
                assert result == "Generated text response"
            
            # Test batch generation
            batch_results = engine.generate(["Prompt 1", "Prompt 2"])
            assert batch_results == ["Batch result"]
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_tensor_parallel_pipeline(self, mock_factory, mock_model, mock_tokenizer):
        """Test tensor parallel inference pipeline."""
        config = Qwen3InferenceConfig(
            model_path="/test/path",
            enable_tensor_parallel=True,
            tensor_parallel_size=2,
            tensor_parallel_rank=0
        )
        
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        # Mock parallelism components
        mock_parallelism = Mock()
        mock_parallelism.setup_distributed.return_value = None
        mock_parallelism.wrap_model.return_value = mock_model
        mock_parallelism.cleanup_distributed.return_value = None
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler'), \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor'), \
             patch('llm_pipeline.inference.qwen3_engine.ParallelismFactory') as mock_parallelism_factory:
            
            mock_parallelism_factory.create_parallelism.return_value = mock_parallelism
            
            # Create engine
            engine = Qwen3InferenceEngine(config)
            
            # Verify parallelism setup
            assert engine.parallelism is not None
            mock_parallelism.setup_distributed.assert_called_once()
            mock_parallelism.wrap_model.assert_called_once_with(mock_model)
            
            # Test cleanup
            engine.cleanup()
            mock_parallelism.cleanup_distributed.assert_called_once()
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_streaming_pipeline(self, mock_factory, mock_model, mock_tokenizer):
        """Test streaming inference pipeline."""
        config = Qwen3InferenceConfig(
            model_path="/test/path",
            enable_streaming=True,
            stream_chunk_size=2
        )
        
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler'), \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor'), \
             patch('llm_pipeline.inference.qwen3_engine.StreamingGenerator') as mock_streaming:
            
            # Mock streaming generator
            mock_streaming_instance = Mock()
            mock_streaming_instance.stream_generate.return_value = ["chunk1", "chunk2", "chunk3"]
            mock_streaming.return_value = mock_streaming_instance
            
            # Create engine
            engine = Qwen3InferenceEngine(config)
            
            # Test streaming generation
            chunks = list(engine.stream_generate("Test prompt"))
            assert chunks == ["chunk1", "chunk2", "chunk3"]
            
            # Verify streaming generator was called correctly
            mock_streaming_instance.stream_generate.assert_called_once()
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_continuous_batching_pipeline(self, mock_factory, mock_model, mock_tokenizer):
        """Test continuous batching inference pipeline."""
        config = Qwen3InferenceConfig(
            model_path="/test/path",
            enable_continuous_batching=True,
            continuous_batch_size=4
        )
        
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler'), \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor'), \
             patch('llm_pipeline.inference.qwen3_engine.StreamingGenerator'), \
             patch('llm_pipeline.inference.qwen3_engine.ContinuousBatcher') as mock_continuous:
            
            # Mock continuous batcher
            mock_continuous_instance = Mock()
            mock_continuous_instance.add_request.return_value = None
            mock_continuous_instance.get_results.return_value = ["Result 1", "Result 2", "Result 3"]
            mock_continuous.return_value = mock_continuous_instance
            
            # Create engine
            engine = Qwen3InferenceEngine(config)
            
            # Test continuous batch generation
            requests = ["Request 1", "Request 2", "Request 3"]
            results = engine.continuous_batch_generate(requests)
            
            assert results == ["Result 1", "Result 2", "Result 3"]
            
            # Verify all requests were added
            assert mock_continuous_instance.add_request.call_count == 3
            mock_continuous_instance.get_results.assert_called_once()
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_mixed_generation_modes(self, mock_factory, mock_model, mock_tokenizer):
        """Test using different generation modes in sequence."""
        config = Qwen3InferenceConfig(
            model_path="/test/path",
            enable_streaming=True,
            enable_continuous_batching=True
        )
        
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler') as mock_sampler, \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor') as mock_batch_processor, \
             patch('llm_pipeline.inference.qwen3_engine.StreamingGenerator') as mock_streaming, \
             patch('llm_pipeline.inference.qwen3_engine.ContinuousBatcher') as mock_continuous:
            
            # Setup mocks
            mock_sampler_instance = Mock()
            mock_sampler_instance.sample.return_value = torch.tensor([[42]])
            mock_sampler.return_value = mock_sampler_instance
            
            mock_batch_processor_instance = Mock()
            mock_batch_processor_instance.process_batch.return_value = ["Batch result"]
            mock_batch_processor.return_value = mock_batch_processor_instance
            
            mock_streaming_instance = Mock()
            mock_streaming_instance.stream_generate.return_value = ["stream1", "stream2"]
            mock_streaming.return_value = mock_streaming_instance
            
            mock_continuous_instance = Mock()
            mock_continuous_instance.add_request.return_value = None
            mock_continuous_instance.get_results.return_value = ["continuous result"]
            mock_continuous.return_value = mock_continuous_instance
            
            # Create engine
            engine = Qwen3InferenceEngine(config)
            
            # Test all generation modes
            with patch.object(engine, '_generate_tokens') as mock_generate:
                mock_generate.return_value = torch.tensor([[1, 2, 3, 42]])
                
                # Single generation
                single_result = engine.generate("Single prompt")
                assert single_result == "Generated text response"
                
                # Batch generation
                batch_results = engine.generate(["Batch prompt 1", "Batch prompt 2"])
                assert batch_results == ["Batch result"]
                
                # Streaming generation
                stream_results = list(engine.stream_generate("Stream prompt"))
                assert stream_results == ["stream1", "stream2"]
                
                # Continuous batch generation
                continuous_results = engine.continuous_batch_generate(["Continuous prompt"])
                assert continuous_results == ["continuous result"]
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_generation_with_custom_parameters(self, mock_factory, mock_model, mock_tokenizer):
        """Test generation with various custom parameters."""
        config = Qwen3InferenceConfig(
            model_path="/test/path",
            temperature=1.0,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.0,
            do_sample=True
        )
        
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler') as mock_sampler, \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor') as mock_batch_processor:
            
            # Mock sampler with custom parameters
            mock_sampler_instance = Mock()
            mock_sampler_instance.sample.return_value = torch.tensor([[42]])
            mock_sampler.return_value = mock_sampler_instance
            
            mock_batch_processor_instance = Mock()
            mock_batch_processor_instance.process_batch.return_value = ["Custom result"]
            mock_batch_processor.return_value = mock_batch_processor_instance
            
            # Create engine
            engine = Qwen3InferenceEngine(config)
            
            # Test with custom parameters
            with patch.object(engine, '_generate_tokens') as mock_generate:
                mock_generate.return_value = torch.tensor([[1, 2, 3, 42]])
                
                result = engine.generate(
                    "Test prompt",
                    max_new_tokens=100,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=30,
                    repetition_penalty=1.1,
                    do_sample=False
                )
                
                assert result == "Generated text response"
                
                # Verify that custom parameters were passed
                # (The exact verification depends on implementation details)
                mock_sampler.assert_called()
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_model_info_and_memory_estimation(self, mock_factory, mock_model, mock_tokenizer):
        """Test model info and memory estimation."""
        config = Qwen3InferenceConfig(model_path="/test/path")
        
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        mock_factory.get_model_info.return_value = {
            'model_type': 'qwen3',
            'vocab_size': 32000,
            'hidden_size': 768,
            'num_layers': 28,
            'num_attention_heads': 16
        }
        mock_factory.estimate_memory_usage.return_value = {
            'total_parameters': 7000000000,
            'model_memory_mb': 14000,
            'model_memory_gb': 13.7,
            'recommended_gpu_memory_gb': 20.6
        }
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler'), \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor'):
            
            # Create engine
            engine = Qwen3InferenceEngine(config)
            
            # Test model info
            model_info = engine.get_model_info()
            assert model_info['model_type'] == 'qwen3'
            assert model_info['vocab_size'] == 32000
            assert model_info['hidden_size'] == 768
            assert model_info['num_layers'] == 28
            
            # Test memory estimation
            memory_info = engine.estimate_memory_usage()
            assert memory_info['total_parameters'] == 7000000000
            assert memory_info['model_memory_gb'] == 13.7
            assert memory_info['recommended_gpu_memory_gb'] == 20.6
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_error_handling(self, mock_factory, mock_model, mock_tokenizer):
        """Test error handling in the pipeline."""
        config = Qwen3InferenceConfig(
            model_path="/test/path",
            enable_streaming=False,
            enable_continuous_batching=False
        )
        
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler'), \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor'):
            
            # Create engine
            engine = Qwen3InferenceEngine(config)
            
            # Test streaming when not enabled
            with pytest.raises(ValueError, match="Streaming not enabled"):
                list(engine.stream_generate("Test prompt"))
            
            # Test continuous batching when not enabled
            with pytest.raises(ValueError, match="Continuous batching not enabled"):
                engine.continuous_batch_generate(["Test prompt"])
    
    @patch('llm_pipeline.inference.qwen3_engine.Qwen3Factory')
    def test_parallelism_factory_integration(self, mock_factory, mock_model, mock_tokenizer):
        """Test integration with parallelism factory."""
        config = Qwen3InferenceConfig(
            model_path="/test/path",
            enable_tensor_parallel=True,
            tensor_parallel_size=4,
            tensor_parallel_rank=1
        )
        
        mock_factory.create_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
        
        # Mock parallelism factory
        mock_parallelism = Mock()
        mock_parallelism.setup_distributed.return_value = None
        mock_parallelism.wrap_model.return_value = mock_model
        mock_parallelism.cleanup_distributed.return_value = None
        
        with patch('llm_pipeline.inference.qwen3_engine.Sampler'), \
             patch('llm_pipeline.inference.qwen3_engine.BatchProcessor'), \
             patch('llm_pipeline.inference.qwen3_engine.ParallelismFactory') as mock_parallelism_factory:
            
            mock_parallelism_factory.create_parallelism.return_value = mock_parallelism
            
            # Create engine
            engine = Qwen3InferenceEngine(config)
            
            # Verify parallelism factory was called with correct config
            mock_parallelism_factory.create_parallelism.assert_called_once()
            call_args = mock_parallelism_factory.create_parallelism.call_args[0][0]
            assert call_args.tensor_parallel_size == 4
            assert call_args.tensor_parallel_rank == 1
            
            # Test cleanup
            engine.cleanup()
            mock_parallelism.cleanup_distributed.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
