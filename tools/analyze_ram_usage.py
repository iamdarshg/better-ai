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
from tools.update_readme_estimates import calculate_parameters

from better_ai.models.core import DeepSeekModel
from better_ai.config import ModelConfig


def measure_memory_footprint(
    config: ModelConfig, batch_size: int, seq_len: int, precision: str = "bf16"
) -> Dict[str, float]:
    """Instantiates a model and measures its memory footprint empirically"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import psutil

    process = psutil.Process()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    dtype = (
        torch.float8_e4m3fn
        if precision == "fp8" and hasattr(torch, "float8_e4m3fn")
        else torch.bfloat16
    )
    torch.set_default_dtype(dtype)
    try:
        # Measure baseline (system + python overhead)
        baseline_cuda = torch.cuda.memory_allocated() if device.type == "cuda" else 0
        baseline_rss = process.memory_info().rss

        if precision == "fp8":
            config.fp8_e4m3 = True
            config.use_fp8 = True
            model = DeepSeekModel(config).to(device)
            # Use the implemented FP8 conversion
            from better_ai.optimizers.fp8 import FP8Linear

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    parent_name = ".".join(name.split(".")[:-1])
                    module_name = name.split(".")[-1]
                    parent = model
                    if parent_name:
                        parent = dict(model.named_modules())[parent_name]

                    fp8_linear = FP8Linear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        use_fp8=True,
                    ).to(device)
                    setattr(parent, module_name, fp8_linear)
        else:
            config.fp8_e4m3 = False
            config.use_fp8 = False
            model = DeepSeekModel(config).to(device)
            # Apply dtype conversion module by module to avoid breaking the model
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    parent_name = ".".join(name.split(".")[:-1])
                    module_name = name.split(".")[-1]
                    parent = model
                    if parent_name:
                        parent = dict(model.named_modules())[parent_name]

                    module.to(dtype)
                    setattr(parent, module_name, module)

        param_mem = (
            (torch.cuda.memory_allocated() - baseline_cuda)
            if device.type == "cuda"
            else sum(p.numel() * p.element_size() for p in model.parameters())
        )

        # Forward pass to measure activation memory
        model.eval()
        with torch.no_grad():
            input_ids = torch.randint(
                0,
                config.vocab_size,
                (batch_size, seq_len),
                device=device,
                dtype=torch.long,
            )
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
            "seq_len": seq_len,
        }
    except Exception as e:
        print(f"Error measuring memory (B={batch_size}, S={seq_len}): {e}")
        return {}


def run_analysis(precisions=None, batch_sizes=None, seq_lengths=None):
    """
    Run RAM analysis for specified precisions and configurations.

    Args:
        precisions: List of precisions to analyze (default: ["bf16", "fp8"])
        batch_sizes: List of batch sizes to test (default: [1, 2, 4])
        seq_lengths: List of sequence lengths to test (default: [128, 256, 512, 1024])
    """
    if precisions is None:
        precisions = ["bf16", "fp8"]
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16]
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]

    small_config = ModelConfig.get_small_model_config()
    small_config.use_cot_specialization = True
    small_config.use_flash_attention = True
    small_config.use_star = True
    small_config.use_tool_heads = True
    small_config.use_json_db_ops_head = True
    small_config.use_math_reasoning_head = True
    small_config.use_algorithm_head = True
    small_config.use_grammar_constraints = True
    small_config.use_entropic_steering = True
    small_config.use_recursive_scratchpad = True
    small_config.use_tidar = True
    small_config.use_reward_models = True
    small_config.use_reasoning_rewards = True
    small_config.use_value_head = True
    small_config.use_fp8 = False  # Will be set per precision
    small_config.fp8_e4m3 = False  # Will be set per precision
    small_config.num_layers = 8
    small_config.use_moe_every_n_layers = 2
    small_config.vocab_size = 15260  # Half of standard BERT vocab size for consistency
    small_config.hidden_dim = 256
    small_config.num_attention_heads = 8
    small_config.num_key_value_heads = 4
    small_config.intermediate_dim = 768

    results = {"bf16": [], "fp8": []}

    for precision in precisions:
        if precision not in results:
            results[precision] = []

        print(f"\n{'=' * 60}")
        print(f"Analyzing {precision.upper()} scaling...")
        print(f"{'=' * 60}")

        for b in batch_sizes:
            for s in seq_lengths:
                print(f"  Testing: Batch={b}, Seq={s}...", end=" ", flush=True)
                res = measure_memory_footprint(small_config, b, s, precision)
                if res:
                    results[precision].append(res)
                    peak_gb = res["peak_bytes"] / (1024**3)
                    param_mb = res["param_bytes"] / (1024**2)
                    overhead_mb = res["overhead_bytes"] / (1024**2)
                    print(
                        f"OK Peak: {peak_gb:.2f}GB (Params: {param_mb:.2f}MB, Overhead: {overhead_mb:.2f}MB)"
                    )
                else:
                    print("FAIL")

    # Analyze and save results
    output_path = Path(__file__).parent.parent / ".ram_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Analysis complete. Data saved to {output_path}")
    print(f"{'=' * 60}\n")

    # Print summary
    total_measurements = sum(len(v) for v in results.values())
    print(f"Total measurements: {total_measurements}")
    for precision, data in results.items():
        if data:
            print(f"  {precision.upper()}: {len(data)} measurements")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze RAM usage for small model variant across different configurations"
    )
    parser.add_argument(
        "--precision",
        nargs="+",
        choices=["bf16", "fp8"],
        help="Precisions to analyze (default: both)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        help="Batch sizes to test (default: 1 2 4)",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        help="Sequence lengths to test (default: 128 256 512 1024)",
    )

    args = parser.parse_args()

    run_analysis(
        precisions=args.precision or ["bf16", "fp8"],
        batch_sizes=args.batch_sizes or [1, 2, 4],
        seq_lengths=args.seq_lengths or [128, 256, 512, 1024],
    )
