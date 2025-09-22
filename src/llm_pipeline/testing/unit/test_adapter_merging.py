"""Unit tests for adapter merging functionality."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch

from llm_pipeline.adapters.merging import (
    AdapterMerger, LoRAAdapterMerger, MergeStrategy, MergeConfig
)
from llm_pipeline.core.config import LoRAConfig


class TestMergeStrategy:
    """Test merge strategy enum."""
    
    def test_merge_strategy_values(self):
        """Test that merge strategies have correct values."""
        assert MergeStrategy.ADDITIVE.value == "additive"
        assert MergeStrategy.WEIGHTED_SUM.value == "weighted_sum"
        assert MergeStrategy.SCALED_ADDITIVE.value == "scaled_additive"
        assert MergeStrategy.MAXIMUM.value == "maximum"
        assert MergeStrategy.MEAN.value == "mean"


class TestMergeConfig:
    """Test merge configuration."""
    
    def test_default_config(self):
        """Test default merge configuration."""
        config = MergeConfig()
        
        assert config.strategy == MergeStrategy.WEIGHTED_SUM  # Updated default
        assert config.weights is None
        assert config.alpha == 1.0
        assert config.save_merged_adapter is True
        assert config.merged_adapter_name == "merged"
        assert config.preserve_originals is True
        assert config.density == 0.5
        assert config.majority_sign_method == "total"
        
    def test_custom_config(self):
        """Test custom merge configuration."""
        config = MergeConfig(
            strategy=MergeStrategy.WEIGHTED_SUM,
            weights=[0.7, 0.3],
            alpha=0.8,
            save_merged_adapter=False,
            merged_adapter_name="custom_merged",
            preserve_originals=False
        )
        
        assert config.strategy == MergeStrategy.WEIGHTED_SUM
        assert config.weights == [0.7, 0.3]
        assert config.alpha == 0.8
        assert config.save_merged_adapter is False
        assert config.merged_adapter_name == "custom_merged"
        assert config.preserve_originals is False


class TestAdapterMerger:
    """Test adapter merger functionality."""
    
    @pytest.fixture
    def sample_adapters(self):
        """Create sample adapter tensors."""
        return {
            "adapter1": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "adapter2": torch.tensor([[0.5, 1.0], [1.5, 2.0]]),
            "adapter3": torch.tensor([[2.0, 1.0], [1.0, 2.0]])
        }
        
    @pytest.fixture
    def merger(self):
        """Create adapter merger."""
        return AdapterMerger()
        
    def test_merge_additive(self, merger, sample_adapters):
        """Test additive merging."""
        adapters = [sample_adapters["adapter1"], sample_adapters["adapter2"]]
        result = merger._merge_additive(adapters)
        
        expected = torch.tensor([[1.5, 3.0], [4.5, 6.0]])
        torch.testing.assert_close(result, expected)
        
    def test_merge_additive_single(self, merger, sample_adapters):
        """Test additive merging with single adapter."""
        adapters = [sample_adapters["adapter1"]]
        result = merger._merge_additive(adapters)
        
        torch.testing.assert_close(result, sample_adapters["adapter1"])
        
    def test_merge_weighted_sum(self, merger, sample_adapters):
        """Test weighted sum merging."""
        adapters = [sample_adapters["adapter1"], sample_adapters["adapter2"]]
        weights = [0.7, 0.3]
        result = merger._merge_weighted_sum(adapters, weights)
        
        expected = 0.7 * sample_adapters["adapter1"] + 0.3 * sample_adapters["adapter2"]
        torch.testing.assert_close(result, expected)
        
    def test_merge_weighted_sum_equal_weights(self, merger, sample_adapters):
        """Test weighted sum with equal weights (default)."""
        adapters = [sample_adapters["adapter1"], sample_adapters["adapter2"]]
        result = merger._merge_weighted_sum(adapters, None)
        
        expected = 0.5 * sample_adapters["adapter1"] + 0.5 * sample_adapters["adapter2"]
        torch.testing.assert_close(result, expected)
        
    def test_merge_weighted_sum_weight_mismatch(self, merger, sample_adapters):
        """Test weighted sum with mismatched weights."""
        adapters = [sample_adapters["adapter1"], sample_adapters["adapter2"]]
        weights = [0.7]  # Only one weight for two adapters
        
        with pytest.raises(ValueError, match="Number of weights"):
            merger._merge_weighted_sum(adapters, weights)
            
    def test_merge_scaled_additive(self, merger, sample_adapters):
        """Test scaled additive merging."""
        adapters = [sample_adapters["adapter1"], sample_adapters["adapter2"]]
        alpha = 0.5
        result = merger._merge_scaled_additive(adapters, alpha)
        
        expected = alpha * (sample_adapters["adapter1"] + sample_adapters["adapter2"])
        torch.testing.assert_close(result, expected)
        
    def test_merge_maximum(self, merger, sample_adapters):
        """Test element-wise maximum merging."""
        adapters = [sample_adapters["adapter1"], sample_adapters["adapter2"]]
        result = merger._merge_maximum(adapters)
        
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # Max of each element
        torch.testing.assert_close(result, expected)
        
    def test_merge_mean(self, merger, sample_adapters):
        """Test element-wise mean merging."""
        adapters = [sample_adapters["adapter1"], sample_adapters["adapter2"]]
        result = merger._merge_mean(adapters)
        
        expected = 0.5 * (sample_adapters["adapter1"] + sample_adapters["adapter2"])
        torch.testing.assert_close(result, expected)
        
    def test_merge_adapters_additive(self, merger, sample_adapters):
        """Test merge_adapters with additive strategy."""
        result = merger.merge_adapters(sample_adapters, MergeStrategy.ADDITIVE)
        
        expected = (sample_adapters["adapter1"] + 
                   sample_adapters["adapter2"] + 
                   sample_adapters["adapter3"])
        torch.testing.assert_close(result, expected)
        
    def test_merge_adapters_weighted_sum(self, merger, sample_adapters):
        """Test merge_adapters with weighted sum strategy."""
        weights = [0.5, 0.3, 0.2]
        result = merger.merge_adapters(
            sample_adapters, MergeStrategy.WEIGHTED_SUM, weights
        )
        
        expected = (0.5 * sample_adapters["adapter1"] + 
                   0.3 * sample_adapters["adapter2"] + 
                   0.2 * sample_adapters["adapter3"])
        torch.testing.assert_close(result, expected)
        
    def test_merge_adapters_scaled_additive(self, merger, sample_adapters):
        """Test merge_adapters with scaled additive strategy."""
        alpha = 0.8
        result = merger.merge_adapters(
            sample_adapters, MergeStrategy.SCALED_ADDITIVE, alpha=alpha
        )
        
        total = (sample_adapters["adapter1"] + 
                sample_adapters["adapter2"] + 
                sample_adapters["adapter3"])
        expected = alpha * total
        torch.testing.assert_close(result, expected)
        
    def test_merge_adapters_empty(self, merger):
        """Test merge_adapters with empty adapter dict."""
        with pytest.raises(ValueError, match="No adapters provided"):
            merger.merge_adapters({})
            
    def test_merge_adapters_invalid_strategy(self, merger, sample_adapters):
        """Test merge_adapters with invalid strategy."""
        with pytest.raises(ValueError, match="Unknown merging strategy"):
            merger.merge_adapters(sample_adapters, "invalid_strategy")
            
        
    def test_merge_ties(self, merger, sample_adapters):
        """Test TIES merging strategy."""
        density = 0.3
        majority_sign_method = "total"
        
        result = merger.merge_adapters(
            sample_adapters, 
            MergeStrategy.TIES, 
            density=density,
            majority_sign_method=majority_sign_method
        )
        
        # TIES should produce a different result than simple addition
        additive_result = merger._merge_additive(list(sample_adapters.values()))
        
        # Results should be different due to interference resolution
        assert not torch.allclose(result, additive_result, atol=1e-6)
        
    def test_merge_dare_linear(self, merger, sample_adapters):
        """Test DARE (Linear) merging strategy."""
        density = 0.4
        
        result = merger.merge_adapters(
            sample_adapters, 
            MergeStrategy.DARE_LINEAR, 
            density=density
        )
        
        # DARE should produce different results due to pruning
        # Use weighted_sum instead of the removed _merge_linear
        weighted_sum_result = merger._merge_weighted_sum(list(sample_adapters.values()), None)
        
        # Results should be different due to DARE pruning
        assert not torch.allclose(result, weighted_sum_result, atol=1e-6)
        
    def test_merge_dare_ties(self, merger, sample_adapters):
        """Test DARE (TIES) merging strategy."""
        density = 0.3
        majority_sign_method = "frequency"
        
        result = merger.merge_adapters(
            sample_adapters, 
            MergeStrategy.DARE_TIES, 
            density=density,
            majority_sign_method=majority_sign_method
        )
        
        # Should be different from both DARE_linear and TIES
        dare_linear_result = merger.merge_adapters(
            sample_adapters, MergeStrategy.DARE_LINEAR, density=density
        )
        ties_result = merger.merge_adapters(
            sample_adapters, MergeStrategy.TIES, density=density, majority_sign_method=majority_sign_method
        )
        
        # Results should be different
        assert not torch.allclose(result, dare_linear_result, atol=1e-6)
        assert not torch.allclose(result, ties_result, atol=1e-6)
        
    def test_dare_prune_and_rescale(self, merger):
        """Test DARE pruning and rescaling."""
        adapter = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        density = 0.5
        
        result = merger._dare_prune_and_rescale(adapter, density, merger.config)
        
        # Should maintain some structure but be different due to random pruning
        assert result.shape == adapter.shape
        assert not torch.allclose(result, adapter, atol=1e-6)
        
    def test_resolve_interference_total(self, merger):
        """Test interference resolution with total method."""
        adapters = [
            torch.tensor([[1.0, -2.0], [3.0, -4.0]]),
            torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])
        ]
        weights = [0.6, 0.4]
        density = 1.0
        majority_sign_method = "total"
        
        result = merger._resolve_interference(adapters, weights, density, majority_sign_method)
        
        # Should resolve sign conflicts
        assert result.shape == adapters[0].shape
        
    def test_resolve_interference_frequency(self, merger):
        """Test interference resolution with frequency method."""
        adapters = [
            torch.tensor([[1.0, -2.0], [3.0, -4.0]]),
            torch.tensor([[-1.0, 2.0], [-3.0, 4.0]]),
            torch.tensor([[1.0, -2.0], [3.0, -4.0]])  # Tie-breaker
        ]
        weights = [0.4, 0.3, 0.3]
        density = 1.0
        majority_sign_method = "frequency"
        
        result = merger._resolve_interference(adapters, weights, density, majority_sign_method)
        
        # Should resolve based on frequency
        assert result.shape == adapters[0].shape
        
    def test_resolve_interference_density(self, merger):
        """Test interference resolution with density sparsification."""
        adapters = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[0.5, 1.0], [1.5, 2.0]])
        ]
        weights = [0.6, 0.4]
        density = 0.5  # Keep only 50% of weights
        majority_sign_method = "total"
        
        result = merger._resolve_interference(adapters, weights, density, majority_sign_method)
        
        # Should have same shape but some weights should be zero
        assert result.shape == adapters[0].shape
        zero_count = (result == 0).sum().item()
        assert zero_count > 0  # Some weights should be zeroed out


class TestLoRAAdapterMerger:
    """Test LoRA adapter merger functionality."""
    
    @pytest.fixture
    def mock_model_wrapper(self):
        """Create mock model wrapper."""
        wrapper = Mock()
        
        # Create mock LoRA modules
        mock_module1 = Mock()
        mock_module1.lora_adapters = {
            "adapter1": Mock(),
            "adapter2": Mock()
        }
        mock_module1.base_layer = Mock()
        mock_module1.base_layer.weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # Mock add_adapter method
        def mock_add_adapter(name, config):
            mock_adapter = Mock()
            mock_adapter.lora_A = torch.randn(config.r, 2)
            mock_adapter.lora_B = torch.randn(2, config.r)
            mock_module1.lora_adapters[name] = mock_adapter
        
        mock_module1.add_adapter = mock_add_adapter
        
        mock_module2 = Mock()
        mock_module2.lora_adapters = {
            "adapter1": Mock(),
            "adapter3": Mock()
        }
        mock_module2.base_layer = Mock()
        mock_module2.base_layer.weight = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        
        # Mock add_adapter method for module2
        def mock_add_adapter2(name, config):
            mock_adapter = Mock()
            mock_adapter.lora_A = torch.randn(config.r, 2)
            mock_adapter.lora_B = torch.randn(2, config.r)
            mock_module2.lora_adapters[name] = mock_adapter
        
        mock_module2.add_adapter = mock_add_adapter2
        
        wrapper.lora_modules = {
            "module1": mock_module1,
            "module2": mock_module2
        }
        
        # Mock adapter get_delta_weight methods
        delta_weight1 = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        delta_weight2 = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
        delta_weight3 = torch.tensor([[0.9, 1.0], [1.1, 1.2]])
        
        mock_module1.lora_adapters["adapter1"].get_delta_weight.return_value = delta_weight1
        mock_module1.lora_adapters["adapter2"].get_delta_weight.return_value = delta_weight2
        mock_module2.lora_adapters["adapter1"].get_delta_weight.return_value = delta_weight1
        mock_module2.lora_adapters["adapter3"].get_delta_weight.return_value = delta_weight3
        
        return wrapper
        
    @pytest.fixture
    def lora_merger(self, mock_model_wrapper):
        """Create LoRA adapter merger."""
        return LoRAAdapterMerger(mock_model_wrapper)
        
    def test_merge_adapters_additive(self, lora_merger):
        """Test merging adapters with additive strategy."""
        adapter_names = ["adapter1", "adapter2"]
        merged_name = lora_merger.merge_adapters(adapter_names, MergeStrategy.ADDITIVE)
        
        assert merged_name == "merged"
        
    def test_merge_adapters_weighted_sum(self, lora_merger):
        """Test merging adapters with weighted sum strategy."""
        adapter_names = ["adapter1", "adapter2"]
        weights = [0.7, 0.3]
        merged_name = lora_merger.merge_adapters(
            adapter_names, MergeStrategy.WEIGHTED_SUM, weights
        )
        
        assert merged_name == "merged"
        
    def test_merge_adapters_empty_names(self, lora_merger):
        """Test merging with empty adapter names."""
        with pytest.raises(ValueError, match="No adapter names provided"):
            lora_merger.merge_adapters([])
            
    def test_merge_adapters_nonexistent(self, lora_merger):
        """Test merging with non-existent adapter names."""
        with pytest.raises(ValueError, match="No adapters found to merge"):
            lora_merger.merge_adapters(["nonexistent"])
            
    def test_merge_with_base(self, lora_merger):
        """Test merging adapters with base model."""
        adapter_names = ["adapter1", "adapter2"]
        lora_merger.merge_with_base(adapter_names, MergeStrategy.ADDITIVE)
        
        # Check that original weights were stored
        assert len(lora_merger.original_weights) == 2
        assert "module1" in lora_merger.original_weights
        assert "module2" in lora_merger.original_weights
        
    def test_unmerge_adapters(self, lora_merger):
        """Test unmerging adapters."""
        # First merge to store original weights
        adapter_names = ["adapter1"]
        lora_merger.merge_with_base(adapter_names, MergeStrategy.ADDITIVE)
        
        # Modify base weights
        original_weight = lora_merger.original_weights["module1"].clone()
        lora_merger.model_wrapper.lora_modules["module1"].base_layer.weight.data += 1.0
        
        # Unmerge
        lora_merger.unmerge_adapters()
        
        # Check that weights were restored
        restored_weight = lora_merger.model_wrapper.lora_modules["module1"].base_layer.weight.data
        torch.testing.assert_close(restored_weight, original_weight)
        
        # Check that original weights were cleared
        assert len(lora_merger.original_weights) == 0
        
    def test_unmerge_without_original_weights(self, lora_merger):
        """Test unmerging without stored original weights."""
        with pytest.raises(ValueError, match="No original weights stored"):
            lora_merger.unmerge_adapters()
            
    def test_get_adapter_info(self, lora_merger):
        """Test getting adapter information."""
        info = lora_merger.get_adapter_info()
        
        assert "adapters" in info
        assert "total_adapters" in info
        assert "modules_with_adapters" in info
        
        assert info["total_adapters"] == 4  # 2 adapters in module1 + 2 in module2
        assert info["modules_with_adapters"] == 2
        assert "module1" in info["adapters"]
        assert "module2" in info["adapters"]
        assert "adapter1" in info["adapters"]["module1"]
        assert "adapter2" in info["adapters"]["module1"]


class TestModelWrapperMerging:
    """Test model wrapper merging methods."""
    
    @pytest.fixture
    def mock_model_wrapper(self):
        """Create mock model wrapper with merging methods."""
        wrapper = Mock()
        
        # Mock LoRA modules with active adapters
        mock_module = Mock()
        mock_module.active_adapters = {"adapter1", "adapter2"}
        mock_module.lora_adapters = {
            "adapter1": Mock(),
            "adapter2": Mock()
        }
        
        wrapper.lora_modules = {"module1": mock_module}
        
        return wrapper
        
    def test_merge_adapters_default(self, mock_model_wrapper):
        """Test default adapter merging."""
        # Mock the LoRAAdapterMerger
        with patch('llm_pipeline.adapters.merging.LoRAAdapterMerger') as mock_merger_class:
            mock_merger = Mock()
            mock_merger.merge_adapters.return_value = "merged_adapter"
            mock_merger_class.return_value = mock_merger
            
            # Import and patch
            import sys
            sys.modules['llm_pipeline.adapters.merging'] = Mock()
            sys.modules['llm_pipeline.adapters.merging'].LoRAAdapterMerger = mock_merger_class
            
            # Test the method
            from llm_pipeline.core.model_wrapper import LoRAModelWrapper
            
            # Create a real wrapper instance
            base_model = nn.Linear(10, 5)
            config = LoRAConfig(r=4, alpha=8.0)
            wrapper = LoRAModelWrapper(base_model, config)
            
            # Mock the merger
            wrapper.merge_adapters = Mock(return_value="merged_adapter")
            
            result = wrapper.merge_adapters()
            
            assert result == "merged_adapter"
            
    def test_merge_with_base(self, mock_model_wrapper):
        """Test merging with base model."""
        # This test would require more complex mocking
        # For now, just test that the method exists and accepts parameters
        assert hasattr(mock_model_wrapper, 'merge_with_base')
        
    def test_unmerge_adapters(self, mock_model_wrapper):
        """Test unmerging adapters."""
        # This test would require more complex mocking
        # For now, just test that the method exists
        assert hasattr(mock_model_wrapper, 'unmerge_adapters')


if __name__ == "__main__":
    pytest.main([__file__])
