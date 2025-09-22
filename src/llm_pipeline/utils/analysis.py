"""Analysis and comparison utilities."""

import torch
import torch.nn as nn
import time
import psutil
import os
from typing import Dict, List, Any, Optional, Tuple
from ..core.config import LoRAConfig
from .factory import create_adapter_linear


def compare_adapters(
    base_layer: nn.Linear,
    configs: Dict[str, LoRAConfig],
    input_tensor: torch.Tensor,
    num_runs: int = 100
) -> Dict[str, Dict[str, Any]]:
    """Compare different adapter types on the same input"""
    results = {}
    
    for adapter_type, config in configs.items():
        print(f"🔍 Analyzing {adapter_type} adapter...")
        
        # Create adapter
        adapter_layer = create_adapter_linear(base_layer, adapter_type, config)
        adapter_layer.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = adapter_layer(input_tensor)
        
        # Timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                output = adapter_layer(input_tensor)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        # Collect metrics
        param_count = sum(p.numel() for p in adapter_layer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in adapter_layer.parameters())
        
        results[adapter_type] = {
            "output_norm": output.norm().item(),
            "output_mean": output.mean().item(),
            "output_std": output.std().item(),
            "trainable_parameters": param_count,
            "total_parameters": total_params,
            "parameter_efficiency": param_count / total_params * 100,
            "forward_time_ms": (end_time - start_time) / num_runs * 1000,
            "output_shape": list(output.shape),
            "config": config
        }
    
    return results


def benchmark_adapters(
    base_layer: nn.Linear,
    adapter_configs: Dict[str, LoRAConfig],
    input_shapes: List[Tuple[int, ...]],
    device: str = "cpu"
) -> Dict[str, Dict[str, Any]]:
    """Comprehensive benchmark of adapter performance"""
    results = {}
    
    for adapter_type, config in adapter_configs.items():
        print(f"🚀 Benchmarking {adapter_type}...")
        
        adapter_results = {}
        adapter_layer = create_adapter_linear(base_layer, adapter_type, config)
        adapter_layer = adapter_layer.to(device)
        adapter_layer.eval()
        
        for i, shape in enumerate(input_shapes):
            input_tensor = torch.randn(*shape, base_layer.in_features).to(device)
            
            # Memory before
            if device == "cuda":
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()
            else:
                memory_before = psutil.Process().memory_info().rss
            
            # Forward pass timing
            torch.cuda.synchronize() if device == "cuda" else None
            start_time = time.time()
            
            with torch.no_grad():
                output = adapter_layer(input_tensor)
            
            torch.cuda.synchronize() if device == "cuda" else None
            end_time = time.time()
            
            # Memory after
            if device == "cuda":
                memory_after = torch.cuda.memory_allocated()
                memory_used = (memory_after - memory_before) / 1024**2  # MB
            else:
                memory_after = psutil.Process().memory_info().rss
                memory_used = (memory_after - memory_before) / 1024**2  # MB
            
            adapter_results[f"shape_{i}"] = {
                "input_shape": shape,
                "forward_time_ms": (end_time - start_time) * 1000,
                "memory_used_mb": memory_used,
                "throughput_samples_per_sec": shape[0] / (end_time - start_time),
                "output_norm": output.norm().item()
            }
        
        results[adapter_type] = adapter_results
    
    return results


def analyze_parameter_efficiency(
    base_layer: nn.Linear,
    adapter_configs: Dict[str, LoRAConfig]
) -> Dict[str, Dict[str, Any]]:
    """Analyze parameter efficiency of different adapters"""
    results = {}
    base_params = sum(p.numel() for p in base_layer.parameters())
    
    print(f"📊 Base layer parameters: {base_params:,}")
    
    for adapter_type, config in adapter_configs.items():
        adapter_layer = create_adapter_linear(base_layer, adapter_type, config)
        
        # Parameter counts
        total_params = sum(p.numel() for p in adapter_layer.parameters())
        trainable_params = sum(p.numel() for p in adapter_layer.parameters() if p.requires_grad)
        adapter_params = total_params - base_params
        
        # Efficiency metrics
        compression_ratio = base_params / adapter_params if adapter_params > 0 else float('inf')
        parameter_overhead = adapter_params / base_params * 100
        trainable_percentage = trainable_params / total_params * 100
        
        results[adapter_type] = {
            "base_parameters": base_params,
            "adapter_parameters": adapter_params,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "compression_ratio": compression_ratio,
            "parameter_overhead_percent": parameter_overhead,
            "trainable_percentage": trainable_percentage,
            "efficiency_score": compression_ratio * (100 - parameter_overhead),
            "config": {
                "rank": config.r,
                "alpha": config.alpha,
                "scaling": config.scaling
            }
        }
    
    return results


def analyze_adapter_interference(
    base_layer: nn.Linear,
    adapter_configs: Dict[str, LoRAConfig],
    input_tensor: torch.Tensor,
    noise_levels: List[float] = [0.0, 0.1, 0.2, 0.5]
) -> Dict[str, Dict[str, Any]]:
    """Analyze how adapters respond to input noise (robustness)"""
    results = {}
    
    for adapter_type, config in adapter_configs.items():
        adapter_layer = create_adapter_linear(base_layer, adapter_type, config)
        adapter_layer.eval()
        
        # Get clean output
        with torch.no_grad():
            clean_output = adapter_layer(input_tensor)
        
        noise_results = {}
        for noise_level in noise_levels:
            # Add noise to input
            noise = torch.randn_like(input_tensor) * noise_level
            noisy_input = input_tensor + noise
            
            with torch.no_grad():
                noisy_output = adapter_layer(noisy_input)
            
            # Measure robustness
            output_diff = (noisy_output - clean_output).norm()
            relative_change = output_diff / clean_output.norm()
            
            noise_results[f"noise_{noise_level}"] = {
                "output_difference": output_diff.item(),
                "relative_change": relative_change.item(),
                "robustness_score": 1.0 / (1.0 + relative_change.item())
            }
        
        results[adapter_type] = noise_results
    
    return results


def generate_analysis_report(
    comparison_results: Dict[str, Dict[str, Any]],
    efficiency_results: Dict[str, Dict[str, Any]],
    benchmark_results: Optional[Dict[str, Dict[str, Any]]] = None
) -> str:
    """Generate a comprehensive analysis report"""
    
    report = []
    report.append("# 📊 Adapter Analysis Report")
    report.append("=" * 50)
    
    # Summary table
    report.append("\n## 📋 Summary Comparison")
    report.append("| Adapter | Params | Efficiency | Speed | Robustness |")
    report.append("|---------|--------|------------|-------|------------|")
    
    for adapter_type in comparison_results.keys():
        comp = comparison_results[adapter_type]
        eff = efficiency_results[adapter_type]
        
        params = f"{comp['trainable_parameters']:,}"
        efficiency = f"{eff['compression_ratio']:.1f}x"
        speed = f"{comp['forward_time_ms']:.2f}ms"
        robustness = f"{comp['output_std']:.4f}"
        
        report.append(f"| {adapter_type} | {params} | {efficiency} | {speed} | {robustness} |")
    
    # Detailed analysis
    report.append("\n## 🔍 Detailed Analysis")
    
    for adapter_type in comparison_results.keys():
        report.append(f"\n### {adapter_type.upper()} Adapter")
        comp = comparison_results[adapter_type]
        eff = efficiency_results[adapter_type]
        
        report.append(f"- **Parameters**: {comp['trainable_parameters']:,} trainable")
        report.append(f"- **Compression**: {eff['compression_ratio']:.1f}x reduction")
        report.append(f"- **Overhead**: {eff['parameter_overhead_percent']:.1f}%")
        report.append(f"- **Forward Time**: {comp['forward_time_ms']:.2f}ms")
        report.append(f"- **Output Stability**: {comp['output_std']:.4f}")
    
    # Recommendations
    report.append("\n## 🎯 Recommendations")
    
    # Find best adapter for each metric
    best_efficiency = min(efficiency_results.items(), key=lambda x: x[1]['parameter_overhead_percent'])
    best_speed = min(comparison_results.items(), key=lambda x: x[1]['forward_time_ms'])
    best_stability = min(comparison_results.items(), key=lambda x: x[1]['output_std'])
    
    report.append(f"- **Most Efficient**: {best_efficiency[0]} ({best_efficiency[1]['compression_ratio']:.1f}x compression)")
    report.append(f"- **Fastest**: {best_speed[0]} ({best_speed[1]['forward_time_ms']:.2f}ms)")
    report.append(f"- **Most Stable**: {best_stability[0]} (std: {best_stability[1]['output_std']:.4f})")
    
    return "\n".join(report)