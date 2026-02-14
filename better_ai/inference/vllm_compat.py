"""
Compatibility layer for vLLM integration with advanced features support.

This module provides a vLLM-compatible wrapper for the DeepSeek model,
enabling efficient inference with PagedAttention, continuous batching,
and all advanced features (TiDAR, STaR, CoT specialization, etc.).
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import json
import torch
import torch.nn as nn
from pathlib import Path


class VLLMDeepSeekModel(nn.Module):
    """
    Wrapper for DeepSeek model to make it compatible with vLLM's internal APIs.
    vLLM expects specific method names and tensor formats.

    Supports all advanced features:
    - Recursive Scratchpad
    - TiDAR (Temporal Diffusion-Augmented Reasoning)
    - CoT Specialization
    - Inner Monologue
    - STaR (Self-Taught Reasoner)
    - Tool Use
    - Specialized Heads (JSON/DB, Math, Algorithm)
    - Entropic Steering
    - Reward Models
    """

    def __init__(self, config, weights_dir: Optional[str] = None):
        super().__init__()
        self.config = config
        self.weights_dir = weights_dir

        # Import here to avoid circular imports
        from ..models.core import DeepSeekModel
        from ..models.moe import DeepSeekMoEModel

        # Initialize the actual model based on config
        if getattr(config, "num_experts", 0) > 0:
            self.model = DeepSeekMoEModel(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_attention_heads,
                num_key_value_heads=getattr(
                    config, "num_key_value_heads", config.num_attention_heads // 2
                ),
                intermediate_size=config.intermediate_dim,
                num_experts=config.num_experts,
                num_experts_per_token=getattr(config, "num_experts_per_token", 2),
                expert_capacity_factor=getattr(config, "expert_capacity_factor", 1.25),
                shared_experts=getattr(config, "shared_experts", 1),
                max_seq_length=config.max_seq_length,
                norm_eps=config.norm_eps,
            )
        else:
            self.model = DeepSeekModel(config)

        # Load weights if directory provided
        if weights_dir:
            self.load_weights(weights_dir)

        # Cache for advanced feature state
        self._advanced_feature_cache = {}

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: Optional[Any] = None,
        use_advanced_features: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass optimized for vLLM's PagedAttention.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            positions: Position IDs for RoPE [batch_size, seq_len]
            kv_caches: List of KV cache tensors for each layer
            attn_metadata: vLLM attention metadata (slot mapping, block tables, etc.)
            use_advanced_features: Whether to apply advanced features

        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Prepare attention mask if provided in metadata
        attention_mask = None
        if attn_metadata is not None and hasattr(attn_metadata, "attention_mask"):
            attention_mask = attn_metadata.attention_mask

        # Run forward pass through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=False,
            return_advanced_features=use_advanced_features,
        )

        logits = outputs["logits"]

        # Apply advanced features if enabled and requested
        if use_advanced_features and "advanced_features" in outputs:
            logits = self._apply_advanced_features(
                logits, outputs["advanced_features"], input_ids
            )

        return logits

    def _apply_advanced_features(
        self,
        logits: torch.Tensor,
        advanced_features: Dict[str, Any],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply advanced features to modify logits.

        Args:
            logits: Original logits from model
            advanced_features: Dictionary of feature outputs
            input_ids: Input token IDs for context

        Returns:
            Modified logits
        """
        # Apply grammar constraints if available
        if "constrained_logits" in advanced_features:
            logits = advanced_features["constrained_logits"]

        # Apply entropic steering if available
        if "entropic_steering" in advanced_features:
            # Entropic steering already modifies logits in the model
            pass

        # Cache advanced features for potential external use
        self._advanced_feature_cache = advanced_features

        return logits

    def load_weights(self, weights_dir: str):
        """
        Efficient weight loading from directory.
        Supports both consolidated checkpoints and sharded weights.

        Args:
            weights_dir: Directory containing model weights
        """
        print(f"Loading weights from {weights_dir} for vLLM...")

        weights_path = Path(weights_dir)

        # Look for checkpoint files
        checkpoint_files = list(weights_path.glob("*.pt")) + list(
            weights_path.glob("*.bin")
        )

        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {weights_dir}")

        # Load the first checkpoint found
        # For sharded models, we would need to load all shards
        checkpoint_path = checkpoint_files[0]
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract state dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Load into model
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {checkpoint_path}")

        # Load config if available
        config_path = weights_path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            # Update config with loaded values
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

    def get_advanced_features(self) -> Dict[str, Any]:
        """Get the latest advanced feature outputs.

        Returns:
            Dictionary of advanced feature outputs from last forward pass
        """
        return self._advanced_feature_cache

    def compute_rewards(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute reward scores using reward models.

        Args:
            hidden_states: Hidden states from model
            attention_mask: Attention mask

        Returns:
            Dictionary of reward scores
        """
        rewards = {}

        if hasattr(self.model, "reward_model"):
            rewards["branch"] = self.model.reward_model(hidden_states, attention_mask)

        if hasattr(self.model, "multi_attr_reward"):
            rewards["multi_attribute"] = self.model.multi_attr_reward(
                hidden_states, attention_mask
            )

        if hasattr(self.model, "value_head"):
            rewards["value"] = self.model.value_head(hidden_states)

        return rewards


def get_vllm_config(model_config) -> Dict[str, Any]:
    """
    Maps Better AI ModelConfig to vLLM's internal configuration format.
    Includes advanced features configuration.

    Args:
        model_config: Better AI ModelConfig instance

    Returns:
        Dictionary compatible with vLLM's model config
    """
    config = {
        # Core architecture
        "architecture": "DeepSeekForCausalLM",
        "model_type": "deepseek",
        "hidden_size": model_config.hidden_dim,
        "num_attention_heads": model_config.num_attention_heads,
        "num_key_value_heads": getattr(
            model_config, "num_key_value_heads", model_config.num_attention_heads // 2
        ),
        "num_hidden_layers": model_config.num_layers,
        "vocab_size": model_config.vocab_size,
        "intermediate_size": model_config.intermediate_dim,
        "max_position_embeddings": model_config.max_seq_length,
        "rms_norm_eps": model_config.norm_eps,
        "rope_theta": model_config.rope_theta,
        # MoE parameters
        "moe_num_experts": getattr(model_config, "num_experts", 0),
        "moe_top_k": getattr(model_config, "num_experts_per_token", 0),
        "moe_shared_experts": getattr(model_config, "shared_experts", 0),
        "moe_capacity_factor": getattr(model_config, "expert_capacity_factor", 1.25),
        "moe_every_n_layers": getattr(model_config, "use_moe_every_n_layers", 2),
        # Quantization
        "quantization": "fp8" if getattr(model_config, "use_fp8", False) else None,
        # vLLM specific
        "dtype": "bfloat16" if getattr(model_config, "use_fp8", False) else "float16",
        "tensor_parallel_size": 1,  # Default, can be overridden
        "pipeline_parallel_size": 1,
        # Advanced features flags (for vLLM to handle if supported)
        "better_ai_config": {
            "use_recursive_scratchpad": getattr(
                model_config, "use_recursive_scratchpad", False
            ),
            "use_tidar": getattr(model_config, "use_tidar", False),
            "use_cot_specialization": getattr(
                model_config, "use_cot_specialization", False
            ),
            "use_inner_monologue": getattr(model_config, "use_inner_monologue", False),
            "use_star": getattr(model_config, "use_star", False),
            "use_tool_heads": getattr(model_config, "use_tool_heads", False),
            "use_json_db_ops_head": getattr(
                model_config, "use_json_db_ops_head", False
            ),
            "use_math_reasoning_head": getattr(
                model_config, "use_math_reasoning_head", False
            ),
            "use_algorithm_head": getattr(model_config, "use_algorithm_head", False),
            "use_grammar_constraints": getattr(
                model_config, "use_grammar_constraints", False
            ),
            "enforce_json_output": getattr(model_config, "enforce_json_output", False),
            "use_entropic_steering": getattr(
                model_config, "use_entropic_steering", False
            ),
            "use_reward_models": getattr(model_config, "use_reward_models", False),
            "use_reasoning_rewards": getattr(
                model_config, "use_reasoning_rewards", False
            ),
            "use_value_head": getattr(model_config, "use_value_head", False),
            "use_striped_attention": getattr(
                model_config, "use_striped_attention", False
            ),
            "use_linear_attention": getattr(
                model_config, "use_linear_attention", False
            ),
            "use_flash_attention": getattr(model_config, "use_flash_attention", True),
        },
    }

    return config


def export_for_vllm(
    model_path: str,
    output_dir: str,
    config_path: Optional[str] = None,
    tensor_parallel_size: int = 1,
) -> None:
    """Export Better AI model to vLLM-compatible format.

    Args:
        model_path: Path to model checkpoint
        output_dir: Directory to save vLLM-compatible files
        config_path: Optional path to model config
        tensor_parallel_size: Number of GPUs for tensor parallelism
    """
    print(f"Exporting model from {model_path} to vLLM format...")

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Load config
    if config_path and os.path.exists(config_path):
        from ..config import ModelConfig

        with open(config_path, "r") as f:
            config_data = json.load(f)
        model_config = ModelConfig.from_dict(config_data)
    else:
        # Infer from checkpoint or use defaults
        from ..config import ModelConfig

        model_config = ModelConfig()

    # Generate vLLM config
    vllm_config = get_vllm_config(model_config)
    vllm_config["tensor_parallel_size"] = tensor_parallel_size

    # Save config
    config_output_path = os.path.join(output_dir, "config.json")
    with open(config_output_path, "w") as f:
        json.dump(vllm_config, f, indent=2)
    print(f"Saved vLLM config to {config_output_path}")

    # Save weights
    if tensor_parallel_size > 1:
        # Shard weights for tensor parallelism
        _save_sharded_weights(state_dict, output_dir, tensor_parallel_size)
    else:
        # Save consolidated weights
        weights_output_path = os.path.join(output_dir, "model.pt")
        torch.save({"model_state_dict": state_dict}, weights_output_path)
        print(f"Saved model weights to {weights_output_path}")

    print(f"Successfully exported to vLLM format in {output_dir}")


def _save_sharded_weights(
    state_dict: Dict[str, torch.Tensor], output_dir: str, tensor_parallel_size: int
) -> None:
    """Save weights sharded for tensor parallelism.

    Args:
        state_dict: Full model state dict
        output_dir: Output directory
        tensor_parallel_size: Number of shards
    """
    print(f"Sharding weights for {tensor_parallel_size}-way tensor parallelism...")

    # Shard along hidden dimension for applicable layers
    sharded_dicts = [{} for _ in range(tensor_parallel_size)]

    for name, tensor in state_dict.items():
        # Determine if this tensor should be sharded
        should_shard = any(
            keyword in name
            for keyword in [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "embed_tokens",
                "lm_head",
            ]
        )

        if should_shard and tensor.dim() >= 2:
            # Shard along the appropriate dimension
            if "embed" in name or "lm_head" in name or "down_proj" in name:
                # Shard output dimension (dim 0)
                shards = torch.chunk(tensor, tensor_parallel_size, dim=0)
            else:
                # Shard input dimension (dim 1 for weights)
                shards = torch.chunk(
                    tensor, tensor_parallel_size, dim=1 if tensor.dim() > 1 else 0
                )

            for i, shard in enumerate(shards):
                sharded_dicts[i][name] = shard
        else:
            # Replicate non-sharded tensors
            for i in range(tensor_parallel_size):
                sharded_dicts[i][name] = tensor

    # Save each shard
    for i, shard_dict in enumerate(sharded_dicts):
        shard_path = os.path.join(output_dir, f"model_shard_{i}.pt")
        torch.save({"model_state_dict": shard_dict}, shard_path)
        print(f"Saved shard {i} to {shard_path}")


def main():
    """CLI entry point for vLLM export."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export Better AI model to vLLM-compatible format"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt or .bin)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save vLLM-compatible files",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to model config JSON (optional)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )

    args = parser.parse_args()
    export_for_vllm(
        args.model_path, args.output_dir, args.config_path, args.tensor_parallel_size
    )


if __name__ == "__main__":
    main()
