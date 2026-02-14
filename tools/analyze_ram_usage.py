#!/usr/bin/env python3
"""
RAM Usage Analyzer for Better AI

Empirically measures VRAM usage for small model variants across different
batch sizes and sequence lengths to determine scaling overheads and
extrapolate to production configurations.
"""

import torch
import torch.nn as nn
import gc
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from better_ai.models.core import DeepSeekModel
from better_ai.config import ModelConfig
from tools.update_readme_estimates import calculate_parameters

def measure_memory_footprint(config: ModelConfig, batch_size: int, seq_len: int, precision: str = "bf16") -> Dict[str, float]:
    """Instantiates a model and measures its memory footprint empirically"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import psutil
    process = psutil.Process()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    dtype = torch.float8_e4m3fn if precision == "fp8" and hasattr(torch, "float8_e4m3fn") else torch.bfloat16

    try:
        # Measure baseline (system + python overhead)
        baseline_cuda = torch.cuda.memory_allocated() if device.type == "cuda" else 0
        baseline_rss = process.memory_info().rss

        model = DeepSeekModel(config).to(device)
        if precision == "fp8":
            # Use the implemented FP8 conversion
            from better_ai.optimizers.fp8 import FP8Linear
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    parent_name = '.'.join(name.split('.')[:-1])
                    module_name = name.split('.')[-1]
                    parent = model
                    if parent_name:
                        parent = dict(model.named_modules())[parent_name]

                    fp8_linear = FP8Linear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        use_fp8=True
                    ).to(device)
                    setattr(parent, module_name, fp8_linear)
        else:
            model = model.to(dtype)

        param_mem = (torch.cuda.memory_allocated() - baseline_cuda) if device.type == "cuda" else sum(p.numel() * p.element_size() for p in model.parameters())

        # Forward pass to measure activation memory
        model.eval()
        with torch.no_grad():
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            # Simple forward pass
            outputs = model(input_ids, use_cache=True)

        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() - baseline_cuda
        else:
            peak_mem = process.memory_info().rss - baseline_rss

        # Cleanup
        del model
        del input_ids
        del outputs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "peak_bytes": peak_mem,
            "param_bytes": param_mem,
            "overhead_bytes": peak_mem - param_mem,
            "batch_size": batch_size,
            "seq_len": seq_len
        }
    except Exception as e:
        print(f"Error measuring memory (B={batch_size}, S={seq_len}): {e}")
        return {}

def run_analysis():
    small_config = ModelConfig.get_small_model_config()

    results = {"bf16": [], "fp8": []}

    # Measure across different seq lengths to find scaling
    seq_lengths = [128, 256, 512, 1024]
    batch_sizes = [1, 2, 4]

    for precision in ["bf16", "fp8"]:
        print(f"Analyzing {precision} scaling...")
        for b in batch_sizes:
            for s in seq_lengths:
                res = measure_memory_footprint(small_config, b, s, precision)
                if res:
                    results[precision].append(res)

    # Analyze and save results
    output_path = Path(__file__).parent.parent / ".ram_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Analysis complete. Data saved to {output_path}")

if __name__ == "__main__":
    run_analysis()
