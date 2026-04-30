"""Unit tests for Qwen3 models."""

import pytest
import torch
from unittest.mock import Mock, patch
from llm_pipeline.models.qwen3 import Qwen3Model, Qwen3Config, Qwen3Factory, Qwen3ForCausalLM


class TestQwen3Config:
    """Test Qwen3Config."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = Qwen3Config()
        
        assert config.model_type == "qwen3"
        assert config.vocab_size == 32000
        assert config.hidden_size == 768
        assert config.num_layers == 28
        assert config.num_attention_heads == 16
        assert config.num_key_value_heads == 8
        assert config.head_dim == 64
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=128,
            num_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4
        )
        
        assert config.vocab_size == 1000
        assert config.hidden_size == 128
        assert config.num_layers == 4
        assert config.num_attention_heads == 8
        assert config.num_key_value_heads == 4
    
    def test_from_huggingface_config(self):
        """Test creating config from HuggingFace config."""
        # Mock HuggingFace config
        mock_config = Mock()
        mock_config.vocab_size = 5000
        mock_config.hidden_size = 256
        mock_config.num_hidden_layers = 6
        mock_config.num_attention_heads = 8
        mock_config.num_key_value_heads = 4
        mock_config.intermediate_size = 1024
        mock_config.max_position_embeddings = 2048
        mock_config.head_dim = 32
        mock_config.model_type = "qwen3"
        mock_config.hidden_act = "silu"
        mock_config.rms_norm_eps = 1e-5
        mock_config.rope_theta = 10000.0
        mock_config.attention_dropout = 0.1
        mock_config.pad_token_id = 151643
        mock_config.eos_token_id = 151643
        mock_config.bos_token_id = 151643
        mock_config.tie_word_embeddings = False
        mock_config.use_cache = True
        mock_config.torch_dtype = "float16"
        
        config = Qwen3Config.from_huggingface_config(mock_config)
        
        assert config.vocab_size == 5000
        assert config.hidden_size == 256
        assert config.num_layers == 6
        assert config.num_attention_heads == 8
        assert config.num_key_value_heads == 4
        assert config.model_type == "qwen3"


class TestQwen3Model:
    """Test Qwen3Model."""
    
    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        return Qwen3Config(
            vocab_size=100,
            hidden_size=32,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            intermediate_size=64,
            max_position_embeddings=128
        )
    
    def test_model_creation(self, small_config):
        """Test model creation."""
        model = Qwen3Model(small_config)
        
        assert model.config == small_config
        assert model.vocab_size == small_config.vocab_size
        assert hasattr(model, 'embed_tokens')
        assert hasattr(model, 'layers')
        assert hasattr(model, 'norm')
        assert len(model.layers) == small_config.num_layers
    
    def test_forward_pass(self, small_config):
        """Test forward pass."""
        model = Qwen3Model(small_config)
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert 'last_hidden_state' in outputs
        assert outputs['last_hidden_state'].shape == (batch_size, seq_len, small_config.hidden_size)
    
    def test_forward_with_cache(self, small_config):
        """Test forward pass with KV cache."""
        model = Qwen3Model(small_config)
        model.config.use_cache = True
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
        
        assert 'last_hidden_state' in outputs
        assert 'past_key_values' in outputs
        assert outputs['past_key_values'] is not None
    
    def test_forward_with_attention_mask(self, small_config):
        """Test forward pass with attention mask."""
        model = Qwen3Model(small_config)
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        assert 'last_hidden_state' in outputs
        assert outputs['last_hidden_state'].shape == (batch_size, seq_len, small_config.hidden_size)
    
    def test_forward_with_position_ids(self, small_config):
        """Test forward pass with position IDs."""
        model = Qwen3Model(small_config)
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        with torch.no_grad():
            outputs = model(input_ids, position_ids=position_ids)
        
        assert 'last_hidden_state' in outputs
        assert outputs['last_hidden_state'].shape == (batch_size, seq_len, small_config.hidden_size)

    def test_lm_head_respects_tie_word_embeddings_flag(self, small_config):
        untied_config = Qwen3Config(**{**small_config.__dict__, "tie_word_embeddings": False})
        untied = Qwen3ForCausalLM(untied_config)
        assert untied.lm_head.weight.data_ptr() != untied.model.embed_tokens.weight.data_ptr()

        tied_config = Qwen3Config(**{**small_config.__dict__, "tie_word_embeddings": True})
        tied = Qwen3ForCausalLM(tied_config)
        assert tied.lm_head.weight.data_ptr() == tied.model.embed_tokens.weight.data_ptr()

    def test_generate_stops_finished_sequences_individually(self):
        class ScriptedModel(Qwen3ForCausalLM):
            def __init__(self):
                config = Qwen3Config(
                    vocab_size=6,
                    hidden_size=8,
                    num_layers=1,
                    num_attention_heads=1,
                    num_key_value_heads=1,
                    head_dim=8,
                    intermediate_size=16,
                    max_position_embeddings=32,
                    eos_token_id=2,
                    pad_token_id=0,
                )
                super().__init__(config)
                self.step = 0

            def forward(
                self,
                input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                use_cache=None,
                **kwargs,
            ):
                batch_size, seq_len = input_ids.shape
                logits = torch.full((batch_size, seq_len, 6), -1000.0)
                if self.step == 0:
                    logits[0, -1, 2] = 0.0
                    logits[1, -1, 4] = 0.0
                else:
                    logits[:, -1, 5] = 0.0
                self.step += 1
                cache = tuple(
                    (torch.zeros(batch_size, 1, seq_len, 8), torch.zeros(batch_size, 1, seq_len, 8))
                    for _ in range(1)
                )
                return type("Out", (), {"logits": logits, "past_key_values": cache})

        model = ScriptedModel()
        outputs = model.generate(
            torch.tensor([[1], [1]]),
            max_new_tokens=3,
            eos_token_id=2,
            pad_token_id=0,
            do_sample=False,
        )

        assert outputs.tolist() == [[1, 2, 0, 0], [1, 4, 5, 5]]


class TestQwen3Factory:
    """Test Qwen3Factory."""
    
    @patch('llm_pipeline.models.qwen3.factory.AutoTokenizer')
    @patch('llm_pipeline.models.qwen3.factory.AutoConfig')
    def test_create_model_and_tokenizer(self, mock_config, mock_tokenizer):
        """Test creating model and tokenizer."""
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_config.from_pretrained.return_value = Mock()

        with patch('llm_pipeline.models.qwen3.factory.load_model') as mock_loader, \
             patch('llm_pipeline.models.qwen3.factory.Qwen3ForCausalLM') as mock_model_cls:
            mock_loader.return_value = None
            mock_model_cls.return_value.to.return_value = Mock()

            model, tokenizer = Qwen3Factory.create_model_and_tokenizer("/test/path", device="cpu", dtype=torch.float32)

            assert model is not None
            assert tokenizer is not None
            mock_tokenizer.from_pretrained.assert_called_once_with("/test/path")
            mock_config.from_pretrained.assert_called_once_with("/test/path")
    
    @patch('llm_pipeline.models.qwen3.factory.AutoConfig')
    def test_get_model_info(self, mock_config):
        """Test getting model info."""
        # Mock config
        mock_config_instance = Mock()
        mock_config_instance.vocab_size = 5000
        mock_config_instance.hidden_size = 256
        mock_config_instance.num_hidden_layers = 6
        mock_config_instance.num_attention_heads = 8
        mock_config_instance.num_key_value_heads = 4
        mock_config_instance.intermediate_size = 1024
        mock_config_instance.max_position_embeddings = 2048
        mock_config_instance.head_dim = 32
        mock_config_instance.model_type = "qwen3"
        mock_config_instance.hidden_act = "silu"
        mock_config_instance.rms_norm_eps = 1e-5
        mock_config_instance.rope_theta = 10000.0
        mock_config_instance.attention_dropout = 0.1
        mock_config_instance.pad_token_id = 151643
        mock_config_instance.eos_token_id = 151643
        mock_config_instance.bos_token_id = 151643
        mock_config_instance.tie_word_embeddings = False
        mock_config_instance.use_cache = True
        mock_config_instance.torch_dtype = "float16"
        mock_config_instance.attention_bias = False
        
        mock_config.from_pretrained.return_value = mock_config_instance
        
        info = Qwen3Factory.get_model_info("/test/path")
        
        assert info['model_type'] == "qwen3"
        assert info['vocab_size'] == 5000
        assert info['hidden_size'] == 256
        assert info['num_layers'] == 6
        assert info['num_attention_heads'] == 8

    @patch('llm_pipeline.models.qwen3.factory.AutoConfig')
    def test_get_model_info_uses_gqa_and_tied_embeddings(self, mock_config):
        mock_config_instance = Mock()
        mock_config_instance.vocab_size = 1000
        mock_config_instance.hidden_size = 256
        mock_config_instance.num_hidden_layers = 4
        mock_config_instance.num_attention_heads = 8
        mock_config_instance.num_key_value_heads = 4
        mock_config_instance.intermediate_size = 1024
        mock_config_instance.max_position_embeddings = 2048
        mock_config_instance.head_dim = 32
        mock_config_instance.model_type = "qwen3"
        mock_config_instance.torch_dtype = "float16"
        mock_config_instance.tie_word_embeddings = True
        mock_config_instance.attention_bias = False
        mock_config.from_pretrained.return_value = mock_config_instance

        info = Qwen3Factory.get_model_info("/test/path")

        expected_attention = (256 * (8 * 32)) + (256 * (4 * 32) * 2) + ((8 * 32) * 256)
        expected_per_layer = expected_attention + (32 * 2) + (256 * 1024 * 3) + (256 * 2)
        expected_total = (1000 * 256) + (expected_per_layer * 4) + 256

        assert info["attention_parameters_per_layer"] == expected_attention
        assert info["parameters_per_layer"] == expected_per_layer
        assert info["lm_head_parameters"] == 0
        assert info["total_parameters"] == expected_total
    
    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        with patch('llm_pipeline.models.qwen3.factory.Qwen3Factory.get_model_info') as mock_info:
            mock_info.return_value = {
                'vocab_size': 1000,
                'hidden_size': 128,
                'num_layers': 4,
                'num_attention_heads': 8,
                'num_key_value_heads': 4,
                'head_dim': 16,
                'intermediate_size': 256,
                'torch_dtype': 'float16'
            }
            
            memory_info = Qwen3Factory.estimate_memory_usage("/test/path")
            
            assert 'total_parameters' in memory_info
            assert 'model_memory_mb' in memory_info
            assert 'model_memory_gb' in memory_info
            assert 'bytes_per_param' in memory_info
            assert 'dtype' in memory_info
            assert 'recommended_gpu_memory_gb' in memory_info
            
            # Check that memory estimation is reasonable
            assert memory_info['total_parameters'] > 0
            assert memory_info['model_memory_mb'] > 0
            assert memory_info['bytes_per_param'] == 2  # fp16


if __name__ == "__main__":
    pytest.main([__file__])
