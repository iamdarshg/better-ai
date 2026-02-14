#!/usr/bin/env python3
"""
RAM Usage Analyzer for Better AI

Empirically measures VRAM usage for small model variants and extrapolates
to production configurations to provide more accurate estimates.
"""

import torch
import torch.nn as nn
import gc
import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from better_ai.models.core import DeepSeekModel
from better_ai.config import ModelConfig

def measure_peak_memory(config: ModelConfig, batch_size: int, seq_len: int) -> Dict[str, float]:
    """Instantiates a model and measures its memory footprint"""
    # Force CPU for estimation if no GPU available, but we want to measure peak memory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reset memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Base memory
    base_mem = 0

    try:
        model = DeepSeekModel(config).to(device)

        # Measure weights memory (static)
        param_mem = sum(p.numel() * p.element_size() for p in model.parameters())

        # Forward pass to measure activation memory
        model.eval()
        with torch.no_grad():
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            # Use small seq_len for measurement if needed
            outputs = model(input_ids, use_cache=True)

        if device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated()
        else:
            # CPU estimation is harder, using param_mem + heuristic for activations
            # In a real scenario, we'd use psutil
            import psutil
            process = psutil.Process()
            peak_vram = process.memory_info().rss

        # Cleanup
        del model
        del input_ids
        del outputs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "peak_memory_bytes": peak_vram,
            "parameter_memory_bytes": param_mem,
            "activation_memory_bytes": peak_vram - param_mem if peak_vram > param_mem else 0
        }
    except Exception as e:
        print(f"Error measuring memory: {e}")
        return {}

def extrapolate_memory(small_result: Dict[str, Any], small_config: ModelConfig, prod_config: ModelConfig) -> float:
    """Extrapolates from small model measurements to production config"""
    # 1. Parameter Scaling (linear with total params)
    # We can calculate this exactly
    prod_param_count = sum(p.numel() for p in DeepSeekModel(prod_config).parameters()) # Might be too slow
    # Alternative: use the existing calculate_parameters function
    from tools.update_readme_estimates import calculate_parameters
    prod_params = calculate_parameters(prod_config)["total_params"]
    small_params = calculate_parameters(small_config)["total_params"]

    param_scaling = prod_params / small_params

    # 2. Activation Scaling
    # Activations scale with: hidden_dim * num_layers * batch_size * seq_len
    prod_act_factor = prod_config.hidden_dim * prod_config.num_layers
    small_act_factor = small_config.hidden_dim * small_config.num_layers
    act_scaling = prod_act_factor / small_act_factor

    # 3. KV Cache Scaling
    # Scales with: num_layers * num_kv_heads * head_dim * seq_len
    prod_kv_factor = prod_config.num_layers * (prod_config.num_key_value_heads or (prod_config.num_attention_heads // 2)) * (prod_config.hidden_dim // prod_config.num_attention_heads)
    small_kv_factor = small_config.num_layers * (small_config.num_key_value_heads or (small_config.num_attention_heads // 2)) * (small_config.hidden_dim // small_config.num_attention_heads)
    kv_scaling = prod_kv_factor / small_kv_factor

    # Extrapolated memory
    extrapolated_params = small_result["parameter_memory_bytes"] * param_scaling
    extrapolated_act = small_result["activation_memory_bytes"] * act_scaling

    # Use a safer multiplier for overhead and fragmentation
    total = (extrapolated_params + extrapolated_act) * 1.15

    return total

if __name__ == "__main__":
    small_config = ModelConfig.get_small_model_config()
    prod_config = ModelConfig.get_production_config()

    print(f"Analyzing RAM usage by measuring small variant...")
    # Measure with batch=1, seq=128
    result = measure_peak_memory(small_config, batch_size=1, seq_len=128)

    if result:
        print(f"Small model peak memory: {result['peak_memory_bytes'] / (1024**2):.2f} MB")

        # Extrapolate
        extrapolated = extrapolate_memory(result, small_config, prod_config)
        print(f"Extrapolated Production RAM: {extrapolated / (1024**3):.2f} GB")

        # Output for script consumption
        with open(".ram_estimate", "w") as f:
            f.write(str(extrapolated))
