"""Unit tests for model registry and architecture detection."""

import pytest
from llm_pipeline.core.registry import ModelRegistry


class TestModelRegistry:
    """Test model registry functionality."""
    
    def test_supported_models_list(self):
        """Test that supported models list is populated."""
        supported_models = ModelRegistry.list_supported_models()
        
        assert isinstance(supported_models, list)
        assert len(supported_models) > 0
        
        # Check for some expected models
        expected_models = ["llama", "mistral", "bert", "roberta", "t5"]
        for model in expected_models:
            assert model in supported_models
    
    def test_model_info_retrieval(self):
        """Test model info retrieval for known models."""
        # Test Llama model info
        llama_info = ModelRegistry.get_model_info("llama")
        
        assert "target_modules" in llama_info
        assert "modules_to_save" in llama_info
        assert "pattern" in llama_info
        assert "architecture_type" in llama_info
        
        assert llama_info["architecture_type"] == "decoder_only"
        assert "q_proj" in llama_info["target_modules"]
        assert "k_proj" in llama_info["target_modules"]
        assert "v_proj" in llama_info["target_modules"]
        assert "o_proj" in llama_info["target_modules"]
        
        # Test BERT model info
        bert_info = ModelRegistry.get_model_info("bert")
        
        assert bert_info["architecture_type"] == "encoder_only"
        assert "query" in bert_info["target_modules"]
        assert "key" in bert_info["target_modules"]
        assert "value" in bert_info["target_modules"]
    
    def test_model_type_detection(self):
        """Test automatic model type detection."""
        # Test exact matches
        assert ModelRegistry.detect_model_type("llama") == "llama"
        assert ModelRegistry.detect_model_type("mistral") == "mistral"
        assert ModelRegistry.detect_model_type("bert") == "bert"
        
        # Test pattern matching
        assert ModelRegistry.detect_model_type("llama-7b") == "llama"
        assert ModelRegistry.detect_model_type("mistral-7b-instruct") == "mistral"
        assert ModelRegistry.detect_model_type("bert-base-uncased") == "bert"
        
        # Test case insensitive
        assert ModelRegistry.detect_model_type("LLAMA") == "llama"
        assert ModelRegistry.detect_model_type("Mistral") == "mistral"
        assert ModelRegistry.detect_model_type("BERT") == "bert"
        
        # Test unknown models
        assert ModelRegistry.detect_model_type("unknown-model") is None
        assert ModelRegistry.detect_model_type("custom-architecture") is None
    
    def test_target_modules_retrieval(self):
        """Test target modules retrieval."""
        # Test known models
        llama_modules = ModelRegistry.get_target_modules("llama")
        assert isinstance(llama_modules, list)
        assert len(llama_modules) > 0
        assert "q_proj" in llama_modules
        
        bert_modules = ModelRegistry.get_target_modules("bert")
        assert isinstance(bert_modules, list)
        assert "query" in bert_modules
        
        # Test unknown model (should return default)
        unknown_modules = ModelRegistry.get_target_modules("unknown")
        assert isinstance(unknown_modules, list)
        assert len(unknown_modules) > 0
    
    def test_modules_to_save_retrieval(self):
        """Test modules to save retrieval."""
        # Test known models
        llama_save = ModelRegistry.get_modules_to_save("llama")
        assert isinstance(llama_save, list)
        assert "embed_tokens" in llama_save
        assert "lm_head" in llama_save
        
        bert_save = ModelRegistry.get_modules_to_save("bert")
        assert isinstance(bert_save, list)
        assert "embeddings" in bert_save
        
        # Test unknown model (should return empty list)
        unknown_save = ModelRegistry.get_modules_to_save("unknown")
        assert isinstance(unknown_save, list)
    
    def test_architecture_type_retrieval(self):
        """Test architecture type retrieval."""
        assert ModelRegistry.get_architecture_type("llama") == "decoder_only"
        assert ModelRegistry.get_architecture_type("mistral") == "decoder_only"
        assert ModelRegistry.get_architecture_type("bert") == "encoder_only"
        assert ModelRegistry.get_architecture_type("roberta") == "encoder_only"
        assert ModelRegistry.get_architecture_type("t5") == "encoder_decoder"
        
        # Test unknown model (should raise error)
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelRegistry.get_architecture_type("unknown")
    
    def test_model_registration(self):
        """Test dynamic model registration."""
        # Register a new model
        ModelRegistry.register_model(
            model_name="test_model",
            target_modules=["test_proj", "test_attn"],
            modules_to_save=["test_embed"],
            pattern="test",
            architecture_type="decoder_only"
        )
        
        # Verify registration
        assert "test_model" in ModelRegistry.list_supported_models()
        
        info = ModelRegistry.get_model_info("test_model")
        assert info["target_modules"] == ["test_proj", "test_attn"]
        assert info["modules_to_save"] == ["test_embed"]
        assert info["pattern"] == "test"
        assert info["architecture_type"] == "decoder_only"
        
        # Test detection
        assert ModelRegistry.detect_model_type("test-model-7b") == "test_model"
    
    def test_fallback_configuration(self):
        """Test fallback configuration for unknown models."""
        info = ModelRegistry.get_model_info("completely_unknown_model")
        
        # Should return default configuration
        assert "target_modules" in info
        assert "modules_to_save" in info
        assert "pattern" in info
        assert "architecture_type" in info
        
        # Should have reasonable defaults
        assert info["architecture_type"] == "decoder_only"
        assert isinstance(info["target_modules"], list)
        assert len(info["target_modules"]) > 0
        assert isinstance(info["modules_to_save"], list)
    
    def test_pattern_matching_edge_cases(self):
        """Test pattern matching edge cases."""
        # Test models with similar names
        assert ModelRegistry.detect_model_type("llama-2") == "llama"
        assert ModelRegistry.detect_model_type("llama-3-70b") == "llama"
        assert ModelRegistry.detect_model_type("alpaca-7b") == "llama"  # Alpaca is Llama-based
        
        assert ModelRegistry.detect_model_type("mistral-7b") == "mistral"
        assert ModelRegistry.detect_model_type("mixtral-8x7b") == "mistral"  # Mixtral is Mistral-based
        
        assert ModelRegistry.detect_model_type("bert-base") == "bert"
        assert ModelRegistry.detect_model_type("roberta-large") == "roberta"
        
        # Test models with special characters
        assert ModelRegistry.detect_model_type("llama-2-7b-chat") == "llama"
        assert ModelRegistry.detect_model_type("mistral-7b-instruct-v0.1") == "mistral"
    
    def test_model_info_consistency(self):
        """Test that model info is consistent across different access methods."""
        model_types = ["llama", "mistral", "bert", "roberta", "t5"]
        
        for model_type in model_types:
            # Get info directly
            direct_info = ModelRegistry.get_model_info(model_type)
            
            # Get info via detection
            detected = ModelRegistry.detect_model_type(model_type)
            if detected:
                detected_info = ModelRegistry.get_model_info(detected)
                
                # Should be the same
                assert direct_info == detected_info
            
            # Test individual components
            target_modules = ModelRegistry.get_target_modules(model_type)
            modules_to_save = ModelRegistry.get_modules_to_save(model_type)
            architecture_type = ModelRegistry.get_architecture_type(model_type)
            
            # Should match the full info
            assert direct_info["target_modules"] == target_modules
            assert direct_info["modules_to_save"] == modules_to_save
            assert direct_info["architecture_type"] == architecture_type


class TestModelRegistryIntegration:
    """Test model registry integration with other components."""
    
    def test_registry_with_model_wrapper(self):
        """Test that model registry works with model wrapper."""
        from llm_pipeline.core.model_wrapper import LoRAModelWrapper
        from llm_pipeline.core.config import LoRAConfig
        import torch.nn as nn
        
        # Create a mock model that matches expected patterns
        class MockLlamaModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.ModuleDict({
                        'self_attn': nn.ModuleDict({
                            'q_proj': nn.Linear(128, 128),
                            'k_proj': nn.Linear(128, 128),
                            'v_proj': nn.Linear(128, 128),
                            'o_proj': nn.Linear(128, 128),
                        }),
                        'mlp': nn.ModuleDict({
                            'gate_proj': nn.Linear(128, 256),
                            'up_proj': nn.Linear(128, 256),
                            'down_proj': nn.Linear(256, 128),
                        })
                    }) for _ in range(2)
                ])
        
        model = MockLlamaModel()
        config = LoRAConfig(target_modules=None)  # Should auto-detect
        
        # Should work without explicitly specifying model type
        wrapper = LoRAModelWrapper(model, config)
        
        # Should have detected Llama-like pattern
        assert wrapper.model_type in ["llama", "unknown"]  # Either detected or fallback
        
        # Should have applied LoRA to target modules
        assert len(wrapper.lora_modules) > 0
    
    def test_registry_with_validation(self):
        """Test that registry works with validation utilities."""
        from llm_pipeline.utils.validation import validate_model_compatibility
        
        # Test with known model types
        model_types = ["llama", "mistral", "bert"]
        
        for model_type in model_types:
            # Should not raise errors for valid model types
            try:
                result = validate_model_compatibility(
                    model=None,  # We're just testing the model type detection
                    config=None,
                    model_type=model_type
                )
                assert "model_type" in result
                assert result["model_type"] == model_type
            except Exception as e:
                # Some validation might fail due to None model, but model type should be recognized
                assert "model_type" in str(e) or "model" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__])
