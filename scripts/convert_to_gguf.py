"""GGUF export implementation for Better AI models.

Converts PyTorch models to GGUF format for use with Ollama, llama.cpp, and other tools.
Supports DeepSeek architecture including MoE layers and advanced features.
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import gguf
    from gguf import GGUFWriter, TensorType
except ImportError:
    gguf = None

try:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    HfApi = None

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from better_ai.config import ModelConfig
from better_ai.models.core import DeepSeekModel
from better_ai.models.moe import DeepSeekMoEModel


# GGUF architecture mappings for DeepSeek
ARCHITECTURE = "deepseek"

# Tensor name mappings from PyTorch to GGUF format
TENSOR_MAPPINGS = {
    # Embeddings
    "embed_tokens.weight": "token_embd.weight",
    # Attention
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    # MLP / Feed-forward
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    # MoE specific
    "mlp.router.router_linear.weight": "expert_router.weight",
    "mlp.router.router_linear.bias": "expert_router.bias",
    "mlp.experts.{i}.gate_up_proj.weight": "expert_{i}_gate_up.weight",
    "mlp.experts.{i}.down_proj.weight": "expert_{i}_down.weight",
    "mlp.shared_experts_layer.{i}.gate_up_proj.weight": "shared_expert_{i}_gate_up.weight",
    "mlp.shared_experts_layer.{i}.down_proj.weight": "shared_expert_{i}_down.weight",
    # Layer norms
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    # Final output
    "norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}

# Advanced features tensor mappings
ADVANCED_FEATURES_MAPPINGS = {
    # Recursive Scratchpad
    "scratchpad.iteration_projections.{i}.weight": "scratchpad_iter_{i}.weight",
    "scratchpad.iteration_projections.{i}.bias": "scratchpad_iter_{i}.bias",
    "scratchpad.scratchpad_transform.weight": "scratchpad_transform.weight",
    "scratchpad.output_projection.weight": "scratchpad_output.weight",
    # TiDAR
    "tidar.diffusion_blocks.{i}.weight": "tidar_diffusion_{i}.weight",
    "tidar.refinement_head.weight": "tidar_refinement.weight",
    # CoT Specialization
    "cot_heads.reasoning_projections.{i}.weight": "cot_reasoning_{i}.weight",
    "cot_heads.output_projection.weight": "cot_output.weight",
    # Inner Monologue
    "inner_monologue.private_projection.weight": "monologue_private.weight",
    "inner_monologue.thought_detector.weight": "monologue_detector.weight",
    # STaR
    "star.consistency_scorer.weight": "star_consistency.weight",
    # Tool Use
    "tool_heads.tool_classifier.weight": "tool_classifier.weight",
    "tool_heads.tool_params_head.weight": "tool_params.weight",
    # Specialized Heads
    "json_db_ops_head.gate.weight": "json_db_ops_gate.weight",
    "json_db_ops_head.expert.weight": "json_db_ops_expert.weight",
    "math_reasoning_head.gate.weight": "math_reasoning_gate.weight",
    "math_reasoning_head.expert.weight": "math_reasoning_expert.weight",
    "algorithm_head.gate.weight": "algorithm_gate.weight",
    "algorithm_head.expert.weight": "algorithm_expert.weight",
    # Reward Models
    "reward_model.classifier.weight": "reward_classifier.weight",
    "reward_model.value_head.weight": "reward_value.weight",
    "multi_attr_reward.attribute_embeddings.weight": "multi_attr_embeddings.weight",
    "hrm.hierarchical_scorer.weight": "hrm_scorer.weight",
    # Entropic Steering
    "entropic_steering.entropy_projection.weight": "entropy_proj.weight",
    # Value Head
    "value_head.weight": "value_head.weight",
}


def get_gguf_tensor_name(pytorch_name: str) -> str:
    """Convert PyTorch tensor name to GGUF format.

    Args:
        pytorch_name: Tensor name from PyTorch model

    Returns:
        GGUF-formatted tensor name
    """
    # Handle layer-specific weights (e.g., "layers.0.self_attn.q_proj.weight")
    parts = pytorch_name.split(".")

    if parts[0] == "layers":
        layer_num = parts[1]
        rest = ".".join(parts[2:])

        # Check if this is an MoE expert weight
        if "mlp.experts." in rest or "mlp.shared_experts_layer." in rest:
            for pattern, gguf_pattern in TENSOR_MAPPINGS.items():
                if "{i}" in pattern:
                    # Handle expert indices
                    import re

                    match = re.match(
                        pattern.replace(".", r"\.").replace("{i}", r"(\d+)"), rest
                    )
                    if match:
                        expert_idx = match.group(1)
                        gguf_name = gguf_pattern.format(i=expert_idx)
                        return f"blk.{layer_num}.{gguf_name}"

        # Standard layer mapping
        if rest in TENSOR_MAPPINGS:
            return f"blk.{layer_num}.{TENSOR_MAPPINGS[rest]}"

    # Check advanced features (not in layers)
    for pattern, gguf_pattern in ADVANCED_FEATURES_MAPPINGS.items():
        if "{i}" in pattern:
            import re

            match = re.match(
                pattern.replace(".", r"\.").replace("{i}", r"(\d+)"), pytorch_name
            )
            if match:
                idx = match.group(1)
                return gguf_pattern.format(i=idx)
        elif pytorch_name.startswith(pattern.split(".")[0]):
            if pytorch_name in ADVANCED_FEATURES_MAPPINGS:
                return ADVANCED_FEATURES_MAPPINGS[pytorch_name]

    # Non-layer weights (embeddings, final norm, lm_head)
    if pytorch_name in TENSOR_MAPPINGS:
        return TENSOR_MAPPINGS[pytorch_name]

    # Return original if no mapping found
    return pytorch_name


def quantize_tensor(tensor: np.ndarray, quantization: str) -> Tuple[np.ndarray, int]:
    """Quantize a tensor according to the specified format.

    Args:
        tensor: Input tensor as numpy array
        quantization: Quantization type (f16, q4_0, q4_1, q8_0)

    Returns:
        Tuple of (quantized_tensor, gguf_type)
    """
    if gguf is None:
        return tensor.astype(np.float32), None

    if quantization == "f16":
        return tensor.astype(np.float16), gguf.GGMLQuantizationType.F16
    elif quantization == "f32":
        return tensor.astype(np.float32), gguf.GGMLQuantizationType.F32
    elif quantization == "q8_0":
        # Simple 8-bit quantization (per-channel)
        scale = np.max(np.abs(tensor)) / 127.0
        if scale == 0:
            scale = 1.0
        quantized = np.clip(tensor / scale, -127, 127).astype(np.int8)
        return quantized, gguf.GGMLQuantizationType.Q8_0
    elif quantization in ["q4_0", "q4_1"]:
        # 4-bit quantization requires more complex block-wise processing
        # For now, use a simplified version
        scale = np.max(np.abs(tensor)) / 7.0
        if scale == 0:
            scale = 1.0
        quantized = np.clip(tensor / scale, -7, 7).astype(np.int8)
        # Pack two 4-bit values into one byte
        packed = np.zeros(tensor.shape[0] // 2 + (tensor.shape[0] % 2), dtype=np.uint8)
        for i in range(0, tensor.shape[0], 2):
            if i + 1 < tensor.shape[0]:
                packed[i // 2] = ((quantized[i] & 0x0F) << 4) | (
                    quantized[i + 1] & 0x0F
                )
            else:
                packed[i // 2] = (quantized[i] & 0x0F) << 4
        return (
            packed,
            gguf.GGMLQuantizationType.Q4_0
            if quantization == "q4_0"
            else gguf.GGMLQuantizationType.Q4_1,
        )
    else:
        raise ValueError(f"Unsupported quantization type: {quantization}")


def export_tokenizer_to_gguf(tokenizer_path: str, writer: GGUFWriter) -> None:
    """Export tokenizer information to GGUF.

    Args:
        tokenizer_path: Path to tokenizer directory or file
        writer: GGUFWriter instance
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

        # Add tokenizer metadata
        writer.add_tokenizer_model("deepseek")
        writer.add_tokenizer_pre("default")

        # Add vocabulary
        vocab = tokenizer.get_vocab()
        vocab_items = sorted(vocab.items(), key=lambda x: x[1])

        tokens = []
        scores = []
        token_types = []

        for token, idx in vocab_items:
            tokens.append(token.encode("utf-8"))
            scores.append(0.0)  # Default score
            token_types.append(gguf.TokenType.NORMAL)

        writer.add_token_list(tokens)
        writer.add_token_scores(scores)
        writer.add_token_types(token_types)

        # Add special tokens
        special_tokens = {}
        if hasattr(tokenizer, "bos_token") and tokenizer.bos_token:
            special_tokens["bos"] = tokenizer.bos_token_id
        if hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
            special_tokens["eos"] = tokenizer.eos_token_id
        if hasattr(tokenizer, "pad_token") and tokenizer.pad_token:
            special_tokens["pad"] = tokenizer.pad_token_id
        if hasattr(tokenizer, "unk_token") and tokenizer.unk_token:
            special_tokens["unk"] = tokenizer.unk_token_id

        for name, token_id in special_tokens.items():
            writer.add_token_special_token(name, token_id)

    except Exception as e:
        print(f"Warning: Could not load tokenizer from {tokenizer_path}: {e}")
        print("Continuing without tokenizer data...")


def add_advanced_features_metadata(writer: GGUFWriter, config: ModelConfig) -> None:
    """Add metadata for advanced features to GGUF.

    Args:
        writer: GGUFWriter instance
        config: ModelConfig with feature flags
    """
    # Recursive Scratchpad
    writer.add_bool(
        "better_ai.use_recursive_scratchpad",
        getattr(config, "use_recursive_scratchpad", False),
    )
    if getattr(config, "use_recursive_scratchpad", False):
        writer.add_uint32(
            "better_ai.scratchpad_max_iterations",
            getattr(config, "scratchpad_max_iterations", 6),
        )
        writer.add_uint32(
            "better_ai.scratchpad_hidden_dim",
            getattr(config, "scratchpad_hidden_dim", 4096),
        )

    # TiDAR
    writer.add_bool("better_ai.use_tidar", getattr(config, "use_tidar", False))
    if getattr(config, "use_tidar", False):
        writer.add_uint32(
            "better_ai.tidar_num_steps", getattr(config, "tidar_num_steps", 2)
        )
        writer.add_uint32(
            "better_ai.tidar_diffusion_dim",
            getattr(config, "tidar_diffusion_dim", 4096),
        )

    # CoT Specialization
    writer.add_bool(
        "better_ai.use_cot_specialization",
        getattr(config, "use_cot_specialization", False),
    )
    if getattr(config, "use_cot_specialization", False):
        writer.add_uint32(
            "better_ai.cot_num_heads", getattr(config, "cot_num_heads", 4)
        )
        writer.add_uint32(
            "better_ai.cot_hidden_dim", getattr(config, "cot_hidden_dim", 3072)
        )

    # Inner Monologue
    writer.add_bool(
        "better_ai.use_inner_monologue", getattr(config, "use_inner_monologue", False)
    )
    if getattr(config, "use_inner_monologue", False):
        writer.add_uint32(
            "better_ai.private_subspace_dim",
            getattr(config, "private_subspace_dim", 3072),
        )
        writer.add_uint32(
            "better_ai.thought_token_id", getattr(config, "thought_token_id", 100)
        )

    # STaR
    writer.add_bool("better_ai.use_star", getattr(config, "use_star", False))
    if getattr(config, "use_star", False):
        writer.add_uint32(
            "better_ai.star_bootstrap_rounds",
            getattr(config, "star_bootstrap_rounds", 3),
        )
        writer.add_uint32(
            "better_ai.star_consistency_samples",
            getattr(config, "star_consistency_samples", 8),
        )

    # Tool Use
    writer.add_bool(
        "better_ai.use_tool_heads", getattr(config, "use_tool_heads", False)
    )
    if getattr(config, "use_tool_heads", False):
        writer.add_uint32(
            "better_ai.tool_vocab_size", getattr(config, "tool_vocab_size", 6144)
        )
        writer.add_uint32(
            "better_ai.tool_hidden_dim", getattr(config, "tool_hidden_dim", 2048)
        )

    # Specialized Heads
    writer.add_bool(
        "better_ai.use_json_db_ops_head", getattr(config, "use_json_db_ops_head", False)
    )
    writer.add_bool(
        "better_ai.use_math_reasoning_head",
        getattr(config, "use_math_reasoning_head", False),
    )
    writer.add_bool(
        "better_ai.use_algorithm_head", getattr(config, "use_algorithm_head", False)
    )

    # Grammar Constraints
    writer.add_bool(
        "better_ai.use_grammar_constraints",
        getattr(config, "use_grammar_constraints", False),
    )
    if getattr(config, "use_grammar_constraints", False):
        writer.add_string(
            "better_ai.grammar_type", getattr(config, "grammar_type", "gbnf")
        )
    writer.add_bool(
        "better_ai.enforce_json_output", getattr(config, "enforce_json_output", False)
    )

    # Entropic Steering
    writer.add_bool(
        "better_ai.use_entropic_steering",
        getattr(config, "use_entropic_steering", False),
    )
    if getattr(config, "use_entropic_steering", False):
        writer.add_float32(
            "better_ai.entropy_threshold", getattr(config, "entropy_threshold", 2.5)
        )

    # Reward Models
    writer.add_bool(
        "better_ai.use_reward_models", getattr(config, "use_reward_models", False)
    )
    writer.add_bool(
        "better_ai.use_reasoning_rewards",
        getattr(config, "use_reasoning_rewards", False),
    )
    writer.add_bool(
        "better_ai.use_value_head", getattr(config, "use_value_head", False)
    )

    # Attention variants
    writer.add_bool(
        "better_ai.use_striped_attention",
        getattr(config, "use_striped_attention", False),
    )
    if getattr(config, "use_striped_attention", False):
        writer.add_uint32(
            "better_ai.striped_block_size", getattr(config, "striped_block_size", 1024)
        )
    writer.add_bool(
        "better_ai.use_linear_attention", getattr(config, "use_linear_attention", False)
    )


def convert_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "f16",
    tokenizer_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> None:
    """Convert Better AI PyTorch model to GGUF format.

    Args:
        model_path: Path to PyTorch model checkpoint (.pt or .bin)
        output_path: Path to save GGUF file
        quantization: Quantization type (f16, f32, q4_0, q4_1, q8_0)
        tokenizer_path: Optional path to tokenizer for embedding vocab
        config_path: Optional path to model config JSON
    """
    if gguf is None:
        print("Error: 'gguf' library not installed. Please run 'pip install gguf'.")
        return

    if torch is None:
        print("Error: 'torch' library not installed. Please run 'pip install torch'.")
        return

    print(f"Loading model from {model_path}...")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # Extract state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Load config if provided
    config = None
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = json.load(f)
        config = ModelConfig.from_dict(config_data)
    else:
        # Try to infer from checkpoint or use defaults
        config = ModelConfig()

    print(f"Converting to GGUF with quantization: {quantization}...")

    # Create GGUF writer
    writer = GGUFWriter(output_path, ARCHITECTURE)

    # Add model metadata
    writer.add_name("Better AI DeepSeek")
    writer.add_architecture(ARCHITECTURE)
    writer.add_description("Better AI DeepSeek model exported to GGUF format")

    # Add model hyperparameters
    writer.add_uint32("vocab_size", config.vocab_size)
    writer.add_uint32("hidden_size", config.hidden_dim)
    writer.add_uint32("num_layers", config.num_layers)
    writer.add_uint32("num_attention_heads", config.num_attention_heads)
    writer.add_uint32(
        "num_key_value_heads", config.num_key_value_heads or config.num_attention_heads
    )
    writer.add_uint32("intermediate_size", config.intermediate_dim)
    writer.add_uint32("max_position_embeddings", config.max_seq_length)
    writer.add_float32("rms_norm_eps", config.norm_eps)
    writer.add_float32("rope_theta", config.rope_theta)

    # Add MoE parameters if applicable
    if config.num_experts > 0:
        writer.add_uint32("num_experts", config.num_experts)
        writer.add_uint32("num_experts_per_token", config.num_experts_per_token)
        writer.add_uint32("shared_experts", config.shared_experts)
        writer.add_float32(
            "expert_capacity_factor", getattr(config, "expert_capacity_factor", 1.1)
        )

    # Add advanced features metadata
    print("Adding advanced features metadata...")
    add_advanced_features_metadata(writer, config)

    # Process and add tensors
    tensor_count = 0
    advanced_tensor_count = 0

    for pytorch_name, tensor in state_dict.items():
        # Convert tensor to numpy
        numpy_tensor = tensor.detach().cpu().numpy()

        # Get GGUF tensor name
        gguf_name = get_gguf_tensor_name(pytorch_name)

        # Track if this is an advanced feature tensor
        is_advanced = any(
            prefix in pytorch_name
            for prefix in [
                "scratchpad",
                "tidar",
                "cot_heads",
                "inner_monologue",
                "star",
                "tool_heads",
                "json_db_ops_head",
                "math_reasoning_head",
                "algorithm_head",
                "reward_model",
                "multi_attr_reward",
                "hrm",
                "entropic_steering",
                "value_head",
            ]
        )

        # Quantize tensor
        try:
            quantized_tensor, tensor_type = quantize_tensor(numpy_tensor, quantization)
            writer.add_tensor(gguf_name, quantized_tensor, raw_dtype=tensor_type)
            tensor_count += 1
            if is_advanced:
                advanced_tensor_count += 1

            if tensor_count % 10 == 0:
                print(f"  Processed {tensor_count} tensors...")

        except Exception as e:
            print(f"Warning: Failed to process tensor {pytorch_name}: {e}")
            # Fall back to F32
            writer.add_tensor(gguf_name, numpy_tensor.astype(np.float32))
            tensor_count += 1

    print(
        f"Exported {tensor_count} tensors ({advanced_tensor_count} advanced feature tensors)"
    )

    # Export tokenizer if path provided
    if tokenizer_path:
        print("Exporting tokenizer...")
        export_tokenizer_to_gguf(tokenizer_path, writer)

    # Write GGUF file
    print("Writing GGUF file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Successfully exported GGUF to {output_path}")

    # Print file size
    file_size = os.path.getsize(output_path) / (1024**3)  # GB
    print(f"File size: {file_size:.2f} GB")


def push_to_huggingface(
    gguf_path: str,
    repo_id: str,
    commit_message: str = "Upload GGUF model",
    private: bool = False,
    token: Optional[str] = None,
    config: Optional[Any] = None,
    repository_url: str = "https://github.com/anomalyco/better-ai",
) -> str:
    """Push GGUF model to HuggingFace Hub.

    Args:
        gguf_path: Path to GGUF file
        repo_id: HuggingFace Hub repository ID (e.g., "username/model-name")
        commit_message: Commit message for the upload
        private: Whether to create private repository
        token: HuggingFace Hub API token (optional, will use cached token if not provided)
        config: Model configuration for model card generation
        repository_url: URL to source code repository

    Returns:
        URL of the uploaded file
    """
    if HfApi is None:
        print(
            "Error: 'huggingface_hub' not installed. Please run 'pip install huggingface_hub'."
        )
        return ""

    print(f"Pushing GGUF model to HuggingFace Hub: {repo_id}...")

    api = HfApi(token=token)

    # Create or get repository
    try:
        repo_url = api.create_repo(
            repo_id=repo_id,
            private=private,
            repo_type="model",
            exist_ok=True,
        )
        print(f"Repository ready: {repo_url}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return ""

    # Generate and upload model card
    try:
        from model_card_utils import generate_model_card, save_model_card
        import tempfile

        model_name = os.path.basename(gguf_path).replace(".gguf", "")
        model_card = generate_model_card(
            model_name=model_name,
            export_format="GGUF",
            config=config,
            repository_url=repository_url,
            tags=["gguf", "ollama", "llama-cpp"],
        )

        # Save model card to temp file and upload
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(model_card)
            readme_path = f.name

        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message="Add model card",
        )
        os.unlink(readme_path)
        print("Model card (README.md) uploaded successfully")

    except Exception as e:
        print(f"Warning: Could not generate/upload model card: {e}")

    # Upload file
    try:
        filename = os.path.basename(gguf_path)
        api.upload_file(
            path_or_fileobj=gguf_path,
            path_in_repo=filename,
            repo_id=repo_id,
            commit_message=commit_message,
        )

        file_url = f"https://huggingface.co/{repo_id}/blob/main/{filename}"
        print(f"Successfully uploaded to: {file_url}")
        return file_url

    except Exception as e:
        print(f"Error uploading file: {e}")
        return ""


def main():
    """CLI entry point for GGUF conversion."""
    parser = argparse.ArgumentParser(
        description="Convert Better AI model to GGUF format for Ollama/llama.cpp"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to PyTorch model checkpoint (.pt or .bin)",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save GGUF file"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="f16",
        choices=["f16", "f32", "q4_0", "q4_1", "q8_0"],
        help="Quantization level",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer directory (optional)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to model config JSON (optional)",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the exported GGUF file to HuggingFace Hub",
        default=True
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="iamdarshg7/Blended-5x3B-gguf",
        help="HuggingFace Hub repository ID (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository on HuggingFace Hub",
        default=False
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload GGUF model",
        help="Commit message for HuggingFace Hub upload",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace Hub API token (optional)",
    )

    args = parser.parse_args()

    # Convert to GGUF
    convert_to_gguf(
        args.model_path,
        args.output_path,
        args.quantization,
        args.tokenizer_path,
        args.config_path,
    )

    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        if args.repo_id is None:
            print("Error: --repo_id is required when using --push_to_hub")
            return

        # Load config for model card generation
        config = None
        if args.config_path and os.path.exists(args.config_path):
            with open(args.config_path, "r") as f:
                config_data = json.load(f)
            config = ModelConfig.from_dict(config_data)

        push_to_huggingface(
            gguf_path=args.output_path,
            repo_id=args.repo_id,
            commit_message=args.commit_message,
            private=args.private,
            token=args.token,
            config=config,
        )


if __name__ == "__main__":
    main()
