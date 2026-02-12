#!/usr/bin/env python3
"""
Comprehensive Resource Estimator for Better AI

Calculates accurate VRAM requirements by accounting for ALL model components.
Uses binary search to find maximum batch sizes that fit in each GPU.
"""

import os
import sys
import yaml
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from better_ai.config import ModelConfig, TrainingConfig, InferenceConfig


@dataclass
class GPUConfig:
    """GPU specifications"""

    name: str
    vram_gb: float
    fp16_tflops: float
    fp8_tflops: float


# GPU specifications
GPU_SPECS = {
    "RTX 2070": GPUConfig("RTX 2070", 8.0, 7.5, 0.0),
    "RTX 5090": GPUConfig("RTX 5090", 32.0, 100.0, 200.0),
    "H300e": GPUConfig("H300e", 80.0, 2000.0, 4000.0),
    "H200": GPUConfig("H200", 141.0, 1000.0, 2000.0),
}


def calculate_parameters(config: ModelConfig) -> Dict:
    """Calculate total and active parameters for the model."""
    vocab_size = config.vocab_size
    hidden_dim = config.hidden_dim
    num_layers = config.num_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads or (num_heads // 2)
    intermediate_dim = config.intermediate_dim
    num_experts = config.num_experts
    num_experts_per_token = config.num_experts_per_token
    shared_experts = config.shared_experts
    head_dim = hidden_dim // num_heads

    # Embeddings
    embedding_params = vocab_size * hidden_dim
    lm_head_params = embedding_params

    # Attention per layer
    q_proj = hidden_dim * (num_heads * head_dim)
    k_proj = hidden_dim * (num_kv_heads * head_dim)
    v_proj = hidden_dim * (num_kv_heads * head_dim)
    o_proj = (num_heads * head_dim) * hidden_dim
    attention_per_layer = q_proj + k_proj + v_proj + o_proj

    # Layer norms per layer (2 norms, 1 param each for RMSNorm)
    norms_per_layer = 2 * hidden_dim

    # FFN/SwiGLU per layer
    ffn_per_layer = 3 * hidden_dim * intermediate_dim

    # MoE
    router_params = hidden_dim * num_experts
    expert_params = ffn_per_layer
    shared_expert_params = shared_experts * expert_params
    all_experts_params = num_experts * expert_params
    active_experts_params = num_experts_per_token * expert_params + shared_expert_params

    # Layer distribution (MoE every 2nd layer)
    num_moe_layers = num_layers // 2
    num_standard_layers = num_layers - num_moe_layers

    standard_layer = attention_per_layer + norms_per_layer + ffn_per_layer
    moe_layer_all = (
        attention_per_layer
        + norms_per_layer
        + router_params
        + all_experts_params
        + shared_expert_params
    )
    moe_layer_active = (
        attention_per_layer + norms_per_layer + router_params + active_experts_params
    )

    total_standard = num_standard_layers * standard_layer
    total_moe_all = num_moe_layers * moe_layer_all
    total_moe_active = num_moe_layers * moe_layer_active

    # Features
    feature_params = {}
    total_feature = 0

    if config.use_tidar:
        # TiDAR: transformer + projections
        d = config.tidar_diffusion_dim
        transformer = config.tidar_num_layers * (
            4 * d * d + 2 * d * 4 * d
        )  # Simplified
        projections = (
            3 * hidden_dim * d + d * d * 2
        )  # input, prompt, output, time embed
        tidar_total = transformer + projections
        feature_params["tidar"] = tidar_total
        total_feature += tidar_total

    if config.use_cot_specialization:
        cot = config.cot_num_heads * (
            hidden_dim * config.cot_hidden_dim * 2
        ) + hidden_dim * (config.cot_hidden_dim // 2)
        feature_params["cot"] = cot
        total_feature += cot

    if config.use_inner_monologue:
        mono = hidden_dim * config.private_subspace_dim * 2 + hidden_dim * (
            hidden_dim // 2
        )
        feature_params["monologue"] = mono
        total_feature += mono

    if config.use_tool_heads:
        tool = (
            hidden_dim * config.tool_hidden_dim * 4
            + config.tool_hidden_dim * config.tool_vocab_size
            + config.tool_hidden_dim * hidden_dim
        )
        feature_params["tool_heads"] = tool
        total_feature += tool

    if config.use_recursive_scratchpad:
        scratch = (
            hidden_dim * config.scratchpad_hidden_dim * 2
            + hidden_dim * (hidden_dim // 2) * 2
        )
        feature_params["scratchpad"] = scratch
        total_feature += scratch

    # Totals
    total_params = (
        embedding_params
        + lm_head_params
        + total_standard
        + total_moe_all
        + total_feature
    )
    active_params = (
        embedding_params
        + lm_head_params
        + total_standard
        + total_moe_active
        + total_feature
    )

    return {
        "total_params": total_params,
        "active_params": active_params,
        "embedding_params": embedding_params,
        "lm_head_params": lm_head_params,
        "attention_total": num_layers * attention_per_layer,
        "ffn_standard_total": num_standard_layers * ffn_per_layer,
        "ffn_moe_total": num_moe_layers * (all_experts_params + shared_expert_params),
        "ffn_moe_active": num_moe_layers * active_experts_params,
        "router_total": num_moe_layers * router_params,
        "feature_params": feature_params,
        "total_feature_params": total_feature,
        "num_standard_layers": num_standard_layers,
        "num_moe_layers": num_moe_layers,
    }


def calculate_inference_memory(
    config: ModelConfig, batch_size: int, seq_len: int, precision: str
) -> float:
    """Calculate VRAM needed for inference in bytes."""
    params = calculate_parameters(config)

    bytes_per_param = 1 if precision == "fp8" else 2

    # Model weights
    model_mem = params["total_params"] * bytes_per_param

    # KV cache: 2 x batch x seq x layers x kv_heads x head_dim x bytes
    num_kv_heads = config.num_key_value_heads or (config.num_attention_heads // 2)
    head_dim = config.hidden_dim // config.num_attention_heads
    kv_cache_mem = (
        2
        * batch_size
        * seq_len
        * config.num_layers
        * num_kv_heads
        * head_dim
        * bytes_per_param
    )

    # Activations (conservative estimate: ~0.5x model memory for working memory)
    activation_mem = model_mem * 0.3

    # CUDA overhead
    overhead_mem = 0.5 * (1024**3)  # 500MB

    # Add 10% fragmentation
    total = (model_mem + kv_cache_mem + activation_mem + overhead_mem) * 1.10

    return total


def calculate_training_memory(
    config: ModelConfig,
    training_config: TrainingConfig,
    batch_size: int,
    seq_len: int,
    precision: str,
    use_8bit_optimizer: bool = True,  # Default to 8-bit optimizer for memory savings
) -> float:
    """Calculate VRAM needed for training in bytes."""
    params = calculate_parameters(config)

    # Memory optimization: Use BF16 for gradients even with FP8 weights
    # This is a common optimization that saves VRAM without hurting convergence
    weight_bytes = 1 if precision == "fp8" else 2  # FP8 or BF16
    grad_bytes = 2  # BF16 for gradients (optimization)
    master_bytes = 4  # FP32 master weights (always needed for stability)

    # Model weights
    model_mem = params["total_params"] * weight_bytes
    master_weights_mem = params["total_params"] * master_bytes

    # Gradients (stored in BF16 to save memory)
    grad_mem = params["total_params"] * grad_bytes

    # Optimizer states (AdamW: momentum + variance)
    # 8-bit optimizer: ~2 bytes per param (quantized states + block scales)
    # Standard FP32: 8 bytes per param (2 x FP32)
    if use_8bit_optimizer:
        optimizer_mem = params["total_params"] * 2  # 8-bit quantized + scales
    else:
        optimizer_mem = params["total_params"] * 2 * 4  # FP32 momentum + variance

    # Activations
    # With gradient checkpointing: store only layer inputs
    # Without: store all intermediate activations
    tokens = batch_size * seq_len
    activation_per_token = config.hidden_dim * config.num_layers * 4  # 4 bytes (FP32)

    if config.use_gradient_checkpointing:
        # Only store checkpoints + current layer activations
        activation_mem = tokens * activation_per_token * 0.25
    else:
        activation_mem = tokens * activation_per_token

    # Communication buffers (DDP)
    comm_buffer_mem = params["total_params"] * 4  # FP32

    # CUDA overhead
    overhead_mem = 1.0 * (1024**3)  # 1GB

    # Add 15% fragmentation for training
    subtotal = (
        model_mem
        + master_weights_mem
        + grad_mem
        + optimizer_mem
        + activation_mem
        + comm_buffer_mem
        + overhead_mem
    )
    total = subtotal * 1.15

    return total


def find_max_inference_batch(config: ModelConfig, gpu_name: str, precision: str) -> int:
    """Find max inference batch size for GPU."""
    gpu = GPU_SPECS[gpu_name]
    available_vram = gpu.vram_gb * (1024**3) * 0.90  # 90% of VRAM

    # Use full sequence length for worst-case scenario
    seq_len = config.max_seq_length

    # Binary search
    low, high = 1, 2048
    max_batch = 0

    while low <= high:
        mid = (low + high) // 2
        mem = calculate_inference_memory(config, mid, seq_len, precision)

        if mem <= available_vram:
            max_batch = mid
            low = mid + 1
        else:
            high = mid - 1

    return max_batch


def find_max_training_batch(
    config: ModelConfig,
    training_config: TrainingConfig,
    gpu_name: str,
    precision: str,
    seq_len: Optional[int] = None,
) -> int:
    """Find max training batch size for GPU."""
    gpu = GPU_SPECS[gpu_name]
    available_vram = gpu.vram_gb * (1024**3) * 0.85  # 85% of VRAM for training

    # Use provided sequence length or fall back to config
    if seq_len is None:
        seq_len = config.max_seq_length

    # Binary search
    low, high = 1, 1024
    max_batch = 0

    while low <= high:
        mid = (low + high) // 2
        mem = calculate_training_memory(
            config, training_config, mid, seq_len, precision, use_8bit_optimizer=True
        )

        if mem <= available_vram:
            max_batch = mid
            low = mid + 1
        else:
            high = mid - 1

    return max_batch


def format_params(num: int) -> str:
    """Format parameter count."""
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    return str(num)


def calculate_training_steps() -> Tuple[int, Dict, float]:
    """Read datasets.yml and sum training steps, plus calculate weighted avg seq length."""
    datasets_path = Path(__file__).parent.parent / "datasets.yml"

    if not datasets_path.exists():
        return 0, {}, 8192  # Default to 8k if no datasets.yml

    with open(datasets_path, "r") as f:
        datasets = yaml.safe_load(f)

    total_steps = 0
    stage_steps = {}
    weighted_seq_sum = 0

    for dataset in datasets.get("datasets", []):
        steps = dataset.get("num_training_steps", 0)
        stage = dataset.get("stage", "unknown")
        seq_len = dataset.get("max_seq_length", 8192)

        if steps > 0:
            total_steps += steps
            if stage not in stage_steps:
                stage_steps[stage] = 0
            stage_steps[stage] += steps
            # Weight sequence length by number of steps
            weighted_seq_sum += seq_len * steps

    # Calculate weighted average sequence length
    avg_seq_length = weighted_seq_sum / total_steps if total_steps > 0 else 8192

    return total_steps, stage_steps, avg_seq_length


def generate_estimate_section() -> str:
    """Generate markdown section for README."""
    model_config = ModelConfig.get_production_config()
    training_config = TrainingConfig()
    inference_config = InferenceConfig()

    params = calculate_parameters(model_config)
    total_steps, stage_steps, avg_seq_length = calculate_training_steps()

    # Calculate batch sizes for each GPU
    results = {}
    for gpu_name in GPU_SPECS.keys():
        results[gpu_name] = {
            "inf_bf16": find_max_inference_batch(model_config, gpu_name, "bf16"),
            "inf_fp8": find_max_inference_batch(model_config, gpu_name, "fp8"),
            "train_bf16": find_max_training_batch(
                model_config, training_config, gpu_name, "bf16", int(avg_seq_length)
            ),
            "train_fp8": find_max_training_batch(
                model_config, training_config, gpu_name, "fp8", int(avg_seq_length)
            ),
        }

    section = f"""## Resource Estimates

*Auto-generated from config using production settings*
*Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

### Model Architecture
- **Total Parameters**: {format_params(params["total_params"])}
- **Active Parameters**: {format_params(params["active_params"])} (per token with MoE)
- **Sparsity**: {(1 - params["active_params"] / params["total_params"]) * 100:.1f}%
- **Layers**: {model_config.num_layers} ({params["num_standard_layers"]} standard + {params["num_moe_layers"]} MoE)
- **Hidden Dimension**: {model_config.hidden_dim}
- **Intermediate Dimension**: {model_config.intermediate_dim}
- **Max Sequence Length**: {model_config.max_seq_length:,} tokens
- **Average Sequence Length**: {avg_seq_length:,.0f} tokens (step-weighted from datasets)

**Parameter Breakdown:**
| Component | Parameters | % of Total |
|-----------|------------|------------|
| Embeddings | {format_params(params["embedding_params"])} | {params["embedding_params"] / params["total_params"] * 100:.1f}% |
| LM Head | {format_params(params["lm_head_params"])} | {params["lm_head_params"] / params["total_params"] * 100:.1f}% |
| Attention | {format_params(params["attention_total"])} | {params["attention_total"] / params["total_params"] * 100:.1f}% |
| FFN (Standard) | {format_params(params["ffn_standard_total"])} | {params["ffn_standard_total"] / params["total_params"] * 100:.1f}% |
| FFN (MoE - All) | {format_params(params["ffn_moe_total"])} | {params["ffn_moe_total"] / params["total_params"] * 100:.1f}% |
| FFN (MoE - Active) | {format_params(params["ffn_moe_active"])} | {params["ffn_moe_active"] / params["active_params"] * 100:.1f}% of active |
| Routers | {format_params(params["router_total"])} | {params["router_total"] / params["total_params"] * 100:.1f}% |
"""

    if params["total_feature_params"] > 0:
        section += "\n**Advanced Features:**\n| Feature | Parameters |\n|---------|------------|\n"
        for name, count in params["feature_params"].items():
            section += f"| {name} | {format_params(count)} |\n"
        section += f"| **Total Features** | **{format_params(params['total_feature_params'])}** |\n"

    section += f"""\n### VRAM Requirements (Batch=1, Seq={model_config.max_seq_length:,})

**Inference VRAM per GPU:**
| GPU | Available | BF16 Required | FP8 Required | BF16 Batch | FP8 Batch |
|-----|-----------|---------------|--------------|------------|-----------|
"""

    # Calculate VRAM for batch=1
    inf_bf16_mem = calculate_inference_memory(
        model_config, 1, model_config.max_seq_length, "bf16"
    ) / (1024**3)
    inf_fp8_mem = calculate_inference_memory(
        model_config, 1, model_config.max_seq_length, "fp8"
    ) / (1024**3)

    for gpu_name in ["RTX 2070", "RTX 5090", "H300e", "H200"]:
        gpu = GPU_SPECS[gpu_name]
        r = results[gpu_name]
        bf16_batch = (
            r["inf_bf16"] if r["inf_bf16"] > 0 else f"0 (need {inf_bf16_mem:.0f}GB)"
        )
        fp8_batch = (
            r["inf_fp8"] if r["inf_fp8"] > 0 else f"0 (need {inf_fp8_mem:.0f}GB)"
        )
        section += f"| {gpu_name} | {gpu.vram_gb:.0f} GB | {inf_bf16_mem:.0f} GB | {inf_fp8_mem:.0f} GB | {bf16_batch} | {fp8_batch} |\n"

    section += f"""\n**Training VRAM per GPU (with 8-bit optimizer, avg seq length):**
| GPU | Available | BF16 Required | FP8 Required | BF16 Batch | FP8 Batch |
|-----|-----------|---------------|--------------|------------|-----------|
"""

    # Calculate VRAM for batch=1 (using 8-bit optimizer and average seq length)
    train_bf16_mem = calculate_training_memory(
        model_config,
        training_config,
        1,
        int(avg_seq_length),
        "bf16",
        use_8bit_optimizer=True,
    ) / (1024**3)
    train_fp8_mem = calculate_training_memory(
        model_config,
        training_config,
        1,
        int(avg_seq_length),
        "fp8",
        use_8bit_optimizer=True,
    ) / (1024**3)

    for gpu_name in ["RTX 2070", "RTX 5090", "H300e", "H200"]:
        gpu = GPU_SPECS[gpu_name]
        r = results[gpu_name]
        bf16_batch = (
            r["train_bf16"]
            if r["train_bf16"] > 0
            else f"0 (need {train_bf16_mem:.0f}GB)"
        )
        fp8_batch = (
            r["train_fp8"] if r["train_fp8"] > 0 else f"0 (need {train_fp8_mem:.0f}GB)"
        )
        section += f"| {gpu_name} | {gpu.vram_gb:.0f} GB | {train_bf16_mem:.0f} GB | {train_fp8_mem:.0f} GB | {bf16_batch} | {fp8_batch} |\n"

    section += f"""\n### Training Pipeline

**Total Steps**: {total_steps:,}

| Stage | Steps |
|-------|-------|
"""

    for stage, steps in sorted(stage_steps.items()):
        section += f"| {stage} | {steps:,} |\n"

    section += f"""\n### Training Time Estimates (100% utilization)

| GPU | FP16 TFLOPS | Hours | Days |
|-----|-------------|-------|------|
"""

    for gpu_name in ["RTX 2070", "RTX 5090", "H300e", "H200"]:
        gpu = GPU_SPECS[gpu_name]
        # Use batch=1 for normalized time estimate
        tokens_per_step = 1 * min(32768, training_config.max_seq_length)
        flops_per_step = 6 * params["total_params"] * tokens_per_step
        total_flops = flops_per_step * total_steps
        seconds = total_flops / (gpu.fp16_tflops * 1e12)
        hours = seconds / 3600
        days = hours / 24
        section += (
            f"| {gpu_name} | {gpu.fp16_tflops:.0f} | {hours:,.0f} | {days:.1f} |\n"
        )

    section += "\n_Note: Training times assume 100% GPU utilization. Real-world training typically achieves 30-70%._\n"

    return section


def update_readme(force: bool = False) -> bool:
    """Update README.md with resource estimates."""
    readme_path = Path(__file__).parent.parent / "README.md"
    cache_path = Path(__file__).parent.parent / ".estimate_cache"

    if not readme_path.exists():
        print(f"Error: README.md not found")
        return False

    # Check for changes
    current_hash = hashlib.md5()
    config_path = Path(__file__).parent.parent / "better_ai" / "config.py"
    datasets_path = Path(__file__).parent.parent / "datasets.yml"

    for path in [config_path, datasets_path]:
        if path.exists():
            with open(path, "rb") as f:
                current_hash.update(f.read())
    current_hash = current_hash.hexdigest()

    if not force and cache_path.exists():
        with open(cache_path) as f:
            if f.read().strip() == current_hash:
                print("Configs unchanged, skipping update")
                return False

    # Read and update README
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_section = generate_estimate_section()
    marker = "## Resource Estimates"

    if marker in content:
        start = content.find(marker)
        end = content.find("\n## ", start + len(marker))
        if end == -1:
            content = content[:start] + new_section
        else:
            content = content[:start] + new_section + "\n" + content[end:]
    else:
        content = content.rstrip() + "\n\n" + new_section

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)

    with open(cache_path, "w") as f:
        f.write(current_hash)

    print("[OK] Updated README.md with resource estimates")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.show:
        print(generate_estimate_section())
    else:
        update_readme(force=args.force)
