"""Unit tests for LoRA core components."""

import pytest
import torch
import torch.nn as nn
from llm_pipeline.core.config import LoRAConfig, DoRAConfig, RSLoRAConfig
from llm_pipeline.core.base_module import BaseLoRAModule
from llm_pipeline.core.model_wrapper import LoRAModelWrapper
from llm_pipeline.adapters.lora import LoRALinear
from llm_pipeline.adapters.dora import DoRAModule, DoRALinear


class TestLoRAConfig:
    """Test LoRA configuration classes."""
    
    def test_lora_config_defaults(self):
        """Test default LoRA configuration values."""
        config = LoRAConfig()
        
        assert config.r == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.1
        assert config.bias == "none"
        assert config.task_type == "CAUSAL_LM"
        assert config.init_lora_weights == "gaussian"
        assert config.use_rslora is False
        assert config.use_dora is False
        assert config.target_modules is None  # Should be None by default, set by model wrapper
    
    def test_lora_config_scaling(self):
        """Test LoRA scaling factor computation."""
        config = LoRAConfig(r=16, alpha=32.0)
        assert config.scaling == 2.0  # alpha / r
        
        rslora_config = RSLoRAConfig(r=16, alpha=32.0)
        import math
        expected_rslora_scaling = 32.0 / math.sqrt(16)  # alpha / sqrt(r)
        assert abs(rslora_config.scaling - expected_rslora_scaling) < 1e-6
    
    def test_lora_config_validation(self):
        """Test LoRA configuration validation."""
        # Valid configurations should not raise errors
        LoRAConfig(r=8, alpha=16.0, dropout=0.1)
        LoRAConfig(r=64, alpha=128.0, dropout=0.0)
        LoRAConfig(r=4, alpha=8.0, dropout=1.0)
        
        # Invalid bias setting should raise ValueError (__post_init__ validation)
        with pytest.raises(ValueError, match="Invalid bias setting"):
            LoRAConfig(bias="invalid")
        
        # Invalid initialization method should raise ValueError (__post_init__ validation)
        with pytest.raises(ValueError, match="Invalid init_lora_weights"):
            LoRAConfig(init_lora_weights="invalid")
    
    def test_dora_config(self):
        """Test DoRA configuration."""
        config = DoRAConfig(r=16, alpha=32.0, magnitude_init="ones")
        
        assert config.use_dora is True  # Should be set automatically
        assert config.magnitude_init == "ones"
        
        # Test invalid magnitude init should raise ValueError (__post_init__ validation)
        with pytest.raises(ValueError, match="Invalid magnitude_init"):
            DoRAConfig(magnitude_init="invalid")


class TestBaseLoRAModule:
    """Test base LoRA module implementation."""
    
    def test_base_lora_initialization(self):
        """Test base LoRA module initialization."""
        config = LoRAConfig(r=8, alpha=16.0)
        module = BaseLoRAModule(128, 256, config)
        
        assert module.in_features == 128
        assert module.out_features == 256
        assert module.rank == 8
        assert module.scaling == 2.0  # alpha / r
        assert module.lora_A.shape == (8, 128)
        assert module.lora_B.shape == (256, 8)
        assert module.merged is False
    
    def test_base_lora_forward_pass(self):
        """Test base LoRA forward pass."""
        config = LoRAConfig(r=4, alpha=8.0, dropout=0.0)  # No dropout for deterministic test
        module = BaseLoRAModule(32, 64, config)
        
        # Create test input
        x = torch.randn(2, 10, 32)
        
        # Forward pass
        output = module(x)
        
        # Check output shape
        assert output.shape == (2, 10, 64)
        
        # With standard LoRA initialization (B matrix zeros), output should be zeros initially
        # This is correct behavior - LoRA starts with zero contribution
        assert torch.allclose(output, torch.zeros_like(output))
    
    def test_base_lora_gradient_flow(self):
        """Test that gradients flow correctly through LoRA module."""
        config = LoRAConfig(r=4, alpha=8.0, dropout=0.0)
        module = BaseLoRAModule(32, 64, config)
        
        # Initialize B matrix to non-zero for gradient testing
        nn.init.normal_(module.lora_B, mean=0.0, std=0.02)
        
        x = torch.randn(2, 10, 32, requires_grad=True)
        
        # Forward pass
        output = module(x)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert module.lora_A.grad is not None
        assert module.lora_B.grad is not None
        
        # Check that gradients are not all zeros
        assert not torch.allclose(module.lora_A.grad, torch.zeros_like(module.lora_A.grad))
        assert not torch.allclose(module.lora_B.grad, torch.zeros_like(module.lora_B.grad))


class TestLoRALinear:
    """Test LoRA-enhanced linear layer."""
    
    def test_lora_linear_initialization(self):
        """Test LoRA linear layer initialization."""
        base_layer = nn.Linear(128, 256)
        config = LoRAConfig(r=8, alpha=16.0)
        lora_linear = LoRALinear(base_layer, config)
        
        assert lora_linear.in_features == 128
        assert lora_linear.out_features == 256
        assert "default" in lora_linear.lora_adapters
        assert "default" in lora_linear.active_adapters
    
    def test_lora_linear_forward_pass(self):
        """Test LoRA linear layer forward pass."""
        base_layer = nn.Linear(64, 128)
        config = LoRAConfig(r=4, alpha=8.0, dropout=0.0)
        lora_linear = LoRALinear(base_layer, config)
        
        x = torch.randn(2, 10, 64)
        
        # Forward pass
        output = lora_linear(x)
        
        # Check output shape
        assert output.shape == (2, 10, 128)
        
        # With standard LoRA initialization (B matrix zeros), output should be same as base layer
        # This is correct behavior - LoRA starts with zero contribution
        base_output = base_layer(x)
        assert torch.allclose(output, base_output)
    
    def test_lora_linear_adapter_management(self):
        """Test LoRA linear layer adapter management."""
        base_layer = nn.Linear(64, 128)
        config = LoRAConfig(r=4, alpha=8.0)
        lora_linear = LoRALinear(base_layer, config)
        
        # Add new adapter
        lora_linear.add_adapter("task1", LoRAConfig(r=8, alpha=16.0))
        
        assert "task1" in lora_linear.lora_adapters
        assert "task1" in lora_linear.active_adapters
        
        # Set active adapters
        lora_linear.set_active_adapters(["task1"])
        assert lora_linear.active_adapters == {"task1"}
        
        # Remove adapter
        lora_linear.remove_adapter("default")
        assert "default" not in lora_linear.lora_adapters
        assert "default" not in lora_linear.active_adapters


class TestDoRAModule:
    """Test DoRA (Weight-Decomposed LoRA) implementation."""
    
    def test_dora_initialization(self):
        """Test DoRA module initialization."""
        config = DoRAConfig(r=8, alpha=16.0, magnitude_init="ones")
        module = DoRAModule(128, 256, config)
        
        assert module.in_features == 128
        assert module.out_features == 256
        assert module.magnitude.shape == (256,)  # One magnitude per output dimension
        assert torch.allclose(module.magnitude, torch.ones(256))  # Initialized to ones
    
    def test_dora_forward_pass(self):
        """Test DoRA forward pass."""
        config = DoRAConfig(r=4, alpha=8.0, magnitude_init="ones")
        module = DoRAModule(32, 64, config)
        
        x = torch.randn(2, 10, 32)
        base_weight = torch.randn(64, 32)  # DoRA requires base weight
        
        # Forward pass
        output = module(x, base_weight)
        
        # Check output shape
        assert output.shape == (2, 10, 64)
        
        # Output should not be all zeros
        assert not torch.allclose(output, torch.zeros_like(output))
    
    def test_dora_magnitude_stats(self):
        """Test DoRA magnitude statistics."""
        config = DoRAConfig(r=4, alpha=8.0, magnitude_init="random")
        module = DoRAModule(32, 64, config)
        
        stats = module.get_magnitude_stats()
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        
        # For random initialization, stats should be reasonable
        assert 0.5 < stats["mean"] < 1.5  # Around 1.0
        assert stats["std"] > 0  # Should have some variance
        assert stats["min"] > 0  # Magnitudes should be positive
        assert stats["max"] > stats["min"]  # Should have range


class TestParameterCounting:
    """Test parameter counting and memory estimation."""
    
    def test_lora_parameter_count(self):
        """Test LoRA parameter counting."""
        config = LoRAConfig(r=8, alpha=16.0)
        module = BaseLoRAModule(128, 256, config)
        
        # LoRA parameters: A (8, 128) + B (256, 8) = 1024 + 2048 = 3072
        expected_params = 8 * 128 + 256 * 8
        actual_params = sum(p.numel() for p in module.parameters())
        
        assert actual_params == expected_params
    
    def test_memory_estimation(self):
        """Test memory estimation utilities."""
        from llm_pipeline.utils.memory import estimate_memory_usage
        
        num_params = 1000000  # 1M parameters
        memory_info = estimate_memory_usage(num_parameters=num_params)
        
        assert "model_memory_mb" in memory_info
        assert "gradient_memory_mb" in memory_info
        assert "optimizer_memory_mb" in memory_info
        assert "total_memory_mb" in memory_info
        
        # Basic sanity checks
        assert memory_info["model_memory_mb"] > 0
        assert memory_info["gradient_memory_mb"] > 0
        assert memory_info["optimizer_memory_mb"] > 0


class TestDoRARankZero:
    """Test DoRA Rank=0 behavior."""
    
    @pytest.fixture
    def dora_config(self):
        """Create DoRA config with rank=0."""
        return DoRAConfig(r=0, alpha=1.0, magnitude_init="ones")
    
    @pytest.fixture
    def dora_module(self, dora_config):
        """Create DoRA module with rank=0."""
        return DoRAModule(128, 256, dora_config)
    
    def test_dora_rank_zero_initialization(self, dora_module):
        """Test DoRA Rank=0 initialization."""
        assert dora_module.rank == 0
        assert dora_module.lora_A is None
        assert dora_module.lora_B is None
        assert dora_module.magnitude is not None
        assert dora_module.magnitude.shape == (256,)
    
    def test_dora_rank_zero_parameter_count(self, dora_module):
        """Test parameter count for DoRA Rank=0."""
        # Should only count magnitude parameters
        param_count = dora_module.get_parameter_count()
        expected_count = dora_module.magnitude.numel()  # 256
        assert param_count == expected_count
        assert param_count == 256
    
    def test_dora_rank_zero_forward_pass(self, dora_module):
        """Test forward pass for DoRA Rank=0."""
        batch_size, seq_len, in_features = 2, 16, 128
        out_features = 256
        
        x = torch.randn(batch_size, seq_len, in_features)
        base_weight = torch.randn(out_features, in_features)
        
        # Forward pass
        output = dora_module(x, base_weight)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, out_features)
        
        # Should not be all zeros (magnitude scaling applied)
        assert not torch.allclose(output, torch.zeros_like(output))
        
        # Should be different from base weight computation
        base_output = x @ base_weight.T
        assert not torch.allclose(output, base_output)
    
    def test_dora_rank_zero_gradient_flow(self, dora_module):
        """Test gradient flow for DoRA Rank=0."""
        batch_size, seq_len, in_features = 2, 16, 128
        out_features = 256
        
        x = torch.randn(batch_size, seq_len, in_features, requires_grad=True)
        base_weight = torch.randn(out_features, in_features, requires_grad=True)
        dora_module.magnitude.requires_grad_(True)
        
        # Forward pass
        output = dora_module(x, base_weight)
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        assert dora_module.magnitude.grad is not None
        assert dora_module.magnitude.grad.norm() > 0
        
        # LoRA parameters should not exist
        assert dora_module.lora_A is None
        assert dora_module.lora_B is None
    
    def test_dora_rank_zero_decomposed_weight(self, dora_module):
        """Test decomposed weight calculation for DoRA Rank=0."""
        out_features, in_features = 256, 128
        base_weight = torch.randn(out_features, in_features)
        
        decomposed_weight = dora_module.get_decomposed_weight(base_weight)
        
        # Check shape
        assert decomposed_weight.shape == (out_features, in_features)
        
        # Should be different from base weight due to magnitude scaling
        assert not torch.allclose(decomposed_weight, base_weight)
        
        # Verify mathematical relationship: W = m * W0 / ||W0||
        weight_norm = torch.norm(base_weight, dim=1, keepdim=True)
        weight_norm = torch.clamp(weight_norm, min=1e-8)
        normalized_weight = base_weight / weight_norm
        expected_weight = dora_module.magnitude.unsqueeze(1) * normalized_weight
        
        assert torch.allclose(decomposed_weight, expected_weight, atol=1e-6)
    
    def test_dora_rank_zero_delta_weight(self, dora_module):
        """Test delta weight calculation for DoRA Rank=0."""
        out_features, in_features = 256, 128
        base_weight = torch.randn(out_features, in_features)
        
        delta_weight = dora_module.get_delta_weight(base_weight)
        
        # Check shape
        assert delta_weight.shape == (out_features, in_features)
        
        # Delta should be non-zero (magnitude scaling creates difference)
        assert delta_weight.norm() > 0
        
        # Verify delta calculation: delta = scaled_weight - base_weight
        decomposed_weight = dora_module.get_decomposed_weight(base_weight)
        expected_delta = decomposed_weight - base_weight
        assert torch.allclose(delta_weight, expected_delta, atol=1e-6)
    
    def test_dora_rank_zero_magnitude_statistics(self, dora_module):
        """Test magnitude statistics for DoRA Rank=0."""
        stats = dora_module.get_magnitude_stats()
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        
        # Initialized with ones, so all should be 1.0
        assert abs(stats["mean"] - 1.0) < 1e-6
        assert abs(stats["std"] - 0.0) < 1e-6
        assert abs(stats["min"] - 1.0) < 1e-6
        assert abs(stats["max"] - 1.0) < 1e-6
    
    def test_dora_rank_zero_magnitude_initialization(self):
        """Test different magnitude initialization methods."""
        # Test "ones" initialization
        config_ones = DoRAConfig(r=0, magnitude_init="ones")
        dora_ones = DoRAModule(64, 128, config_ones)
        assert torch.allclose(dora_ones.magnitude, torch.ones(128))
        
        # Test "random" initialization
        config_random = DoRAConfig(r=0, magnitude_init="random")
        dora_random = DoRAModule(64, 128, config_random)
        # Random init should be around 1.0 with some variance
        assert 0.5 < dora_random.magnitude.mean().item() < 1.5
        
        # Test "kaiming" initialization
        config_kaiming = DoRAConfig(r=0, magnitude_init="kaiming")
        dora_kaiming = DoRAModule(64, 128, config_kaiming)
        # Kaiming init should have some variance
        assert dora_kaiming.magnitude.std().item() > 0
    
    def test_dora_rank_zero_vs_standard_lora(self):
        """Test that DoRA Rank=0 is different from standard LoRA Rank=0."""
        # Create DoRA Rank=0
        dora_config = DoRAConfig(r=0, alpha=1.0)
        dora_module = DoRAModule(128, 256, dora_config)
        
        # Create standard LoRA Rank=0
        lora_config = LoRAConfig(r=0, alpha=1.0)
        lora_module = BaseLoRAModule(128, 256, lora_config)
        
        # Test input
        x = torch.randn(2, 16, 128)
        
        # DoRA should produce non-zero output (magnitude scaling)
        dora_output = dora_module(x, torch.randn(256, 128))
        assert not torch.allclose(dora_output, torch.zeros_like(dora_output))
        
        # Standard LoRA should produce zero output (no adaptation)
        lora_output = lora_module(x)
        assert torch.allclose(lora_output, torch.zeros_like(lora_output))
        
        # Parameter counts should be different
        dora_params = dora_module.get_parameter_count()
        lora_params = lora_module.get_parameter_count()
        
        assert dora_params > lora_params  # DoRA has magnitude params
        assert dora_params == 256  # Only magnitude
        assert lora_params == 0    # No parameters
    
    def test_dora_rank_zero_with_different_magnitudes(self):
        """Test DoRA Rank=0 with different magnitude values."""
        config = DoRAConfig(r=0, magnitude_init="ones")
        dora_module = DoRAModule(64, 128, config)
        
        # Manually set different magnitude values
        dora_module.magnitude.data = torch.tensor([0.5, 1.0, 2.0, 0.1] * 32)  # Repeat pattern
        
        x = torch.randn(2, 8, 64)
        base_weight = torch.randn(128, 64)
        
        output = dora_module(x, base_weight)
        
        # Should produce different outputs based on magnitude scaling
        assert not torch.allclose(output, torch.zeros_like(output))
        
        # Check that magnitude scaling is applied correctly
        decomposed_weight = dora_module.get_decomposed_weight(base_weight)
        weight_norm = torch.norm(base_weight, dim=1, keepdim=True)
        weight_norm = torch.clamp(weight_norm, min=1e-8)
        normalized_weight = base_weight / weight_norm
        expected_weight = dora_module.magnitude.unsqueeze(1) * normalized_weight
        
        assert torch.allclose(decomposed_weight, expected_weight, atol=1e-6)


class TestDoRALinearRankZero:
    """Test DoRALinear with Rank=0."""
    
    @pytest.fixture
    def base_layer(self):
        """Create base linear layer."""
        return nn.Linear(128, 256)
    
    @pytest.fixture
    def dora_config(self):
        """Create DoRA config with rank=0."""
        return DoRAConfig(r=0, alpha=1.0, magnitude_init="ones")
    
    @pytest.fixture
    def dora_linear(self, base_layer, dora_config):
        """Create DoRALinear with rank=0."""
        return DoRALinear(base_layer, dora_config)
    
    def test_dora_linear_rank_zero_forward(self, dora_linear):
        """Test DoRALinear forward pass with rank=0."""
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, 128)
        
        # Forward pass
        output = dora_linear(x)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, 256)
        
        # Should be different from base layer output (magnitude scaling)
        base_output = dora_linear.base_layer(x)
        assert not torch.allclose(output, base_output)
    
    def test_dora_linear_rank_zero_disable_enable(self, dora_linear):
        """Test DoRALinear adapter disable/enable with rank=0."""
        x = torch.randn(2, 16, 128)
        
        # With adapter enabled (default)
        output_enabled = dora_linear(x)
        
        # Disable adapter
        dora_linear.disable_adapter()
        output_disabled = dora_linear(x)
        
        # Re-enable adapter
        dora_linear.enable_adapter()
        output_reenabled = dora_linear(x)
        
        # Disabled should match base layer
        base_output = dora_linear.base_layer(x)
        assert torch.allclose(output_disabled, base_output)
        
        # Enabled should be different from disabled
        assert not torch.allclose(output_enabled, output_disabled)
        
        # Re-enabled should match originally enabled
        assert torch.allclose(output_enabled, output_reenabled)
    
    def test_dora_linear_rank_zero_magnitude_stats(self, dora_linear):
        """Test DoRALinear magnitude statistics with rank=0."""
        stats = dora_linear.get_magnitude_stats()
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        
        # Should be initialized with ones
        assert abs(stats["mean"] - 1.0) < 1e-6


class TestOptimizerMemoryCalculator:
    """Test optimizer memory calculator functionality."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing."""
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        return model
    
    @pytest.fixture
    def sample_optimizers(self, sample_model):
        """Create sample optimizers for testing."""
        return {
            "adam": torch.optim.Adam(sample_model.parameters(), lr=1e-4),
            "adamw": torch.optim.AdamW(sample_model.parameters(), lr=1e-4),
            "sgd": torch.optim.SGD(sample_model.parameters(), lr=1e-3),
            "rmsprop": torch.optim.RMSprop(sample_model.parameters(), lr=1e-3),
            "adagrad": torch.optim.Adagrad(sample_model.parameters(), lr=1e-3)
        }
    
    def test_calculate_optimizer_memory_empty_state(self, sample_optimizers):
        """Test optimizer memory calculation with empty state."""
        from llm_pipeline.utils.optimizer_memory import OptimizerMemoryCalculator
        
        for name, optimizer in sample_optimizers.items():
            memory = OptimizerMemoryCalculator.calculate_optimizer_memory(optimizer)
            # Some optimizers may have initial state, so we just check it's non-negative
            assert memory >= 0
    
    def test_calculate_optimizer_memory_with_state(self, sample_optimizers):
        """Test optimizer memory calculation after optimization step."""
        from llm_pipeline.utils.optimizer_memory import OptimizerMemoryCalculator
        
        # Create dummy loss and do backward pass to populate optimizer states
        dummy_input = torch.randn(10, 100)
        dummy_target = torch.randint(0, 10, (10,))
        
        for name, optimizer in sample_optimizers.items():
            # Create a simple model and optimizer for this test
            test_model = nn.Linear(100, 10)
            test_optimizer = torch.optim.Adam(test_model.parameters(), lr=1e-4)
            
            # Forward and backward pass to populate optimizer state
            output = test_model(dummy_input)
            loss = nn.CrossEntropyLoss()(output, dummy_target)
            loss.backward()
            test_optimizer.step()
            
            memory = OptimizerMemoryCalculator.calculate_optimizer_memory(test_optimizer)
            assert memory > 0  # Should have memory after optimization step
    
    def test_estimate_optimizer_memory_by_type(self):
        """Test memory estimation by optimizer type."""
        from llm_pipeline.utils.optimizer_memory import OptimizerMemoryCalculator
        
        trainable_params = 1000
        
        # Test different optimizer types
        test_cases = [
            ("adam", 8000),  # 2 * 4 bytes per param
            ("adamw", 8000),  # 2 * 4 bytes per param
            ("sgd", 0),      # No additional states
            ("rmsprop", 4000),  # 1 * 4 bytes per param
            ("adagrad", 4000),  # 1 * 4 bytes per param
            ("adamax", 8000),   # 2 * 4 bytes per param
            ("unknown", 8000)   # Default to Adam
        ]
        
        for optimizer_type, expected_memory in test_cases:
            memory = OptimizerMemoryCalculator.estimate_optimizer_memory_by_type(
                trainable_params, optimizer_type, "fp32"
            )
            assert memory == expected_memory
    
    def test_estimate_optimizer_memory_fp16(self):
        """Test memory estimation with FP16 precision."""
        from llm_pipeline.utils.optimizer_memory import OptimizerMemoryCalculator
        
        trainable_params = 1000
        
        memory_fp32 = OptimizerMemoryCalculator.estimate_optimizer_memory_by_type(
            trainable_params, "adam", "fp32"
        )
        memory_fp16 = OptimizerMemoryCalculator.estimate_optimizer_memory_by_type(
            trainable_params, "adam", "fp16"
        )
        
        assert memory_fp16 == memory_fp32 // 2  # FP16 uses half the bytes
    
    def test_get_optimizer_type_from_instance(self, sample_optimizers):
        """Test getting optimizer type from instance."""
        from llm_pipeline.utils.optimizer_memory import OptimizerMemoryCalculator
        
        type_mapping = {
            "adam": "adam",
            "adamw": "adamw",
            "sgd": "sgd",
            "rmsprop": "rmsprop",
            "adagrad": "adagrad"
        }
        
        for name, optimizer in sample_optimizers.items():
            optimizer_type = OptimizerMemoryCalculator.get_optimizer_type_from_instance(optimizer)
            assert optimizer_type == type_mapping[name]
    
    def test_get_optimizer_info_empty_state(self, sample_optimizers):
        """Test getting optimizer info with empty state."""
        from llm_pipeline.utils.optimizer_memory import OptimizerMemoryCalculator
        
        optimizer = sample_optimizers["adam"]
        info = OptimizerMemoryCalculator.get_optimizer_info(optimizer)
        
        assert "optimizer_type" in info
        assert "total_memory_bytes" in info
        assert "total_memory_mb" in info
        assert "total_memory_gb" in info
        assert "parameter_count" in info
        assert "state_count" in info
        assert "memory_per_param_bytes" in info
        
        assert info["optimizer_type"] == "adam"
        assert info["total_memory_bytes"] == 0  # No state
        assert info["parameter_count"] > 0
        assert info["state_count"] == 0


class TestMemoryFootprintWithOptimizer:
    """Test memory footprint calculation with optimizer."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing."""
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        return model
    
    def test_get_memory_footprint_without_optimizer(self, sample_model):
        """Test memory footprint without optimizer."""
        from llm_pipeline.utils.optimizer_memory import get_memory_footprint_with_optimizer
        
        footprint = get_memory_footprint_with_optimizer(sample_model)
        
        assert "total_parameters" in footprint
        assert "trainable_parameters" in footprint
        assert "parameter_memory_mb" in footprint
        assert "gradient_memory_mb" in footprint
        assert "optimizer_memory_mb" in footprint
        assert "estimated_total_mb" in footprint
        assert "optimizer_type" in footprint
        assert "precision" in footprint
        
        assert footprint["optimizer_type"] == "estimated_adam"
        assert footprint["precision"] == "fp32"
        assert footprint["optimizer_memory_mb"] > 0
    
    def test_get_memory_footprint_with_optimizer(self, sample_model):
        """Test memory footprint with optimizer."""
        from llm_pipeline.utils.optimizer_memory import get_memory_footprint_with_optimizer
        
        optimizer = torch.optim.Adam(sample_model.parameters(), lr=1e-4)
        footprint = get_memory_footprint_with_optimizer(sample_model, optimizer)
        
        assert footprint["optimizer_type"] == "adam"
        assert "optimizer_info" in footprint
        
        optimizer_info = footprint["optimizer_info"]
        assert optimizer_info["optimizer_type"] == "adam"
        assert optimizer_info["total_memory_bytes"] == 0  # No state yet
    
    def test_get_memory_footprint_fp16(self, sample_model):
        """Test memory footprint with FP16 precision."""
        from llm_pipeline.utils.optimizer_memory import get_memory_footprint_with_optimizer
        
        footprint_fp32 = get_memory_footprint_with_optimizer(sample_model, precision="fp32")
        footprint_fp16 = get_memory_footprint_with_optimizer(sample_model, precision="fp16")
        
        # FP16 should use roughly half the memory
        assert footprint_fp16["parameter_memory_mb"] < footprint_fp32["parameter_memory_mb"]
        assert footprint_fp16["gradient_memory_mb"] < footprint_fp32["gradient_memory_mb"]
        assert footprint_fp16["optimizer_memory_mb"] < footprint_fp32["optimizer_memory_mb"]


class TestTrainingMemoryEstimation:
    """Test training memory estimation with optimizer types."""

    def test_estimate_training_memory_public_helper_preserves_optimizer_type(self):
        """The top-level helper should expose optimizer-sensitive estimates."""
        from llm_pipeline.utils.memory import estimate_training_memory

        model = nn.Linear(16, 16)

        sgd_memory = estimate_training_memory(
            model,
            batch_size=2,
            sequence_length=8,
            hidden_size=16,
            optimizer_type="sgd",
        )
        adam_memory = estimate_training_memory(
            model,
            batch_size=2,
            sequence_length=8,
            hidden_size=16,
            optimizer_type="adam",
        )

        assert sgd_memory["optimizer_type"] == "sgd"
        assert adam_memory["optimizer_type"] == "adam"
        assert sgd_memory["optimizer_memory_mb"] < adam_memory["optimizer_memory_mb"]
    
    def test_estimate_training_memory_different_optimizers(self):
        """Test training memory estimation with different optimizer types."""
        from llm_pipeline.utils.optimizer_memory import estimate_training_memory_with_optimizer
        
        trainable_params = 1000
        batch_size = 2
        sequence_length = 512
        
        optimizers = ["adam", "adamw", "sgd", "rmsprop", "adagrad"]
        
        for optimizer_type in optimizers:
            memory = estimate_training_memory_with_optimizer(
                trainable_params, optimizer_type, batch_size, sequence_length
            )
            
            assert "model_memory_mb" in memory
            assert "gradient_memory_mb" in memory
            assert "optimizer_memory_mb" in memory
            assert "activation_memory_mb" in memory
            assert "total_memory_mb" in memory
            assert "total_memory_gb" in memory
            assert memory["optimizer_type"] == optimizer_type
            assert memory["precision"] == "fp32"
            assert memory["batch_size"] == batch_size
            assert memory["sequence_length"] == sequence_length
    
    def test_estimate_training_memory_sgd_vs_adam(self):
        """Test that SGD uses less memory than Adam."""
        from llm_pipeline.utils.optimizer_memory import estimate_training_memory_with_optimizer
        
        trainable_params = 1000
        
        adam_memory = estimate_training_memory_with_optimizer(
            trainable_params, "adam", batch_size=1, sequence_length=512
        )
        sgd_memory = estimate_training_memory_with_optimizer(
            trainable_params, "sgd", batch_size=1, sequence_length=512
        )
        
        assert sgd_memory["optimizer_memory_mb"] < adam_memory["optimizer_memory_mb"]
    
    def test_estimate_training_memory_with_activations(self):
        """Test training memory estimation with and without activations."""
        from llm_pipeline.utils.optimizer_memory import estimate_training_memory_with_optimizer
        
        trainable_params = 1000
        
        with_activations = estimate_training_memory_with_optimizer(
            trainable_params, "adam", batch_size=2, sequence_length=512, include_activations=True
        )
        without_activations = estimate_training_memory_with_optimizer(
            trainable_params, "adam", batch_size=2, sequence_length=512, include_activations=False
        )
        
        assert with_activations["activation_memory_mb"] > 0
        assert without_activations["activation_memory_mb"] == 0
        assert with_activations["total_memory_mb"] > without_activations["total_memory_mb"]
    
    def test_estimate_training_memory_fp16(self):
        """Test training memory estimation with FP16 precision."""
        from llm_pipeline.utils.optimizer_memory import estimate_training_memory_with_optimizer
        
        trainable_params = 1000
        
        fp32_memory = estimate_training_memory_with_optimizer(
            trainable_params, "adam", precision="fp32"
        )
        fp16_memory = estimate_training_memory_with_optimizer(
            trainable_params, "adam", precision="fp16"
        )
        
        assert fp16_memory["model_memory_mb"] < fp32_memory["model_memory_mb"]
        assert fp16_memory["gradient_memory_mb"] < fp32_memory["gradient_memory_mb"]
        assert fp16_memory["optimizer_memory_mb"] < fp32_memory["optimizer_memory_mb"]


class TestOptimizerMemoryComparison:
    """Test optimizer memory comparison functionality."""
    
    def test_compare_optimizer_memory_default(self):
        """Test comparing default optimizer types."""
        from llm_pipeline.utils.optimizer_memory import compare_optimizer_memory_usage
        
        trainable_params = 1000
        
        comparison = compare_optimizer_memory_usage(trainable_params)
        
        expected_optimizers = ["adam", "adamw", "sgd", "rmsprop", "adagrad", "adamax"]
        
        for optimizer_type in expected_optimizers:
            assert optimizer_type in comparison
            assert "memory_bytes" in comparison[optimizer_type]
            assert "memory_mb" in comparison[optimizer_type]
            assert "memory_gb" in comparison[optimizer_type]
            assert "memory_per_param_bytes" in comparison[optimizer_type]
    
    def test_compare_optimizer_memory_custom(self):
        """Test comparing custom optimizer types."""
        from llm_pipeline.utils.optimizer_memory import compare_optimizer_memory_usage
        
        trainable_params = 1000
        custom_optimizers = ["sgd", "adam", "rmsprop"]
        
        comparison = compare_optimizer_memory_usage(trainable_params, custom_optimizers)
        
        assert len(comparison) == 3
        for optimizer_type in custom_optimizers:
            assert optimizer_type in comparison
    
    def test_compare_optimizer_memory_ordering(self):
        """Test that SGD uses least memory, LBFGS uses most."""
        from llm_pipeline.utils.optimizer_memory import compare_optimizer_memory_usage
        
        trainable_params = 1000
        optimizers = ["sgd", "adam", "lbfgs"]
        
        comparison = compare_optimizer_memory_usage(trainable_params, optimizers)
        
        sgd_memory = comparison["sgd"]["memory_bytes"]
        adam_memory = comparison["adam"]["memory_bytes"]
        lbfgs_memory = comparison["lbfgs"]["memory_bytes"]
        
        assert sgd_memory < adam_memory < lbfgs_memory
    
    def test_compare_optimizer_memory_fp16(self):
        """Test optimizer memory comparison with FP16 precision."""
        from llm_pipeline.utils.optimizer_memory import compare_optimizer_memory_usage
        
        trainable_params = 1000
        
        fp32_comparison = compare_optimizer_memory_usage(trainable_params, precision="fp32")
        fp16_comparison = compare_optimizer_memory_usage(trainable_params, precision="fp16")
        
        for optimizer_type in fp32_comparison:
            assert optimizer_type in fp16_comparison
            # Only check if FP32 memory is > 0 (to avoid issues with SGD having 0 memory)
            if fp32_comparison[optimizer_type]["memory_bytes"] > 0:
                assert fp16_comparison[optimizer_type]["memory_bytes"] < fp32_comparison[optimizer_type]["memory_bytes"]
            else:
                # If FP32 is 0, FP16 should also be 0
                assert fp16_comparison[optimizer_type]["memory_bytes"] == 0


class TestOptimizerMemoryIntegration:
    """Test integration with LoRA model wrapper."""
    
    @pytest.fixture
    def sample_model_wrapper(self):
        """Create a sample LoRA model wrapper."""
        model = nn.Linear(100, 10)
        config = LoRAConfig(r=4, alpha=8.0)
        wrapper = LoRAModelWrapper(model, config)
        return wrapper
    
    def test_model_wrapper_memory_footprint(self, sample_model_wrapper):
        """Test model wrapper memory footprint with optimizer."""
        optimizer = torch.optim.AdamW(sample_model_wrapper.parameters(), lr=1e-4)
        
        footprint = sample_model_wrapper.get_memory_footprint(optimizer=optimizer)
        
        assert "optimizer_type" in footprint
        assert footprint["optimizer_type"] == "adamw"
    
    def test_model_wrapper_training_memory_estimation(self, sample_model_wrapper):
        """Test model wrapper training memory estimation."""
        memory = sample_model_wrapper.estimate_training_memory(
            optimizer_type="sgd",
            batch_size=4,
            sequence_length=256
        )
        
        assert memory["optimizer_type"] == "sgd"
        assert memory["batch_size"] == 4
        assert memory["sequence_length"] == 256
    
    def test_model_wrapper_optimizer_comparison(self, sample_model_wrapper):
        """Test model wrapper optimizer comparison."""
        comparison = sample_model_wrapper.compare_optimizer_memory(
            optimizer_types=["sgd", "adam", "rmsprop"]
        )
        
        assert len(comparison) == 3
        assert "sgd" in comparison
        assert "adam" in comparison
        assert "rmsprop" in comparison


if __name__ == "__main__":
    pytest.main([__file__])
