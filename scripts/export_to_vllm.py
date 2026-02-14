#!/usr/bin/env python
"""
vLLM export script - CLI entry point for exporting Better AI models to vLLM format.

This script calls the VLLMDeepSeekModel wrapper from better_ai.inference.vllm_compat
to export models in a format compatible with vLLM inference engine.

Usage:
    python scripts/export_to_vllm.py --model_path checkpoint.pt --output_dir ./vllm_model
    python scripts/export_to_vllm.py --model_path checkpoint.pt --output_dir ./vllm_model --push_to_hub --repo_id username/model-name
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from better_ai.inference.vllm_compat import (
    VLLMDeepSeekModel,
    export_for_vllm,
    get_vllm_config,
)


def push_to_huggingface(
    output_dir: str,
    repo_id: str,
    commit_message: str = "Upload vLLM model",
    private: bool = False,
    token: str = None,
    config_path: str = None,
    repository_url: str = "https://github.com/anomalyco/better-ai",
) -> str:
    """Push vLLM model to HuggingFace Hub.

    Args:
        output_dir: Directory containing vLLM model files
        repo_id: HuggingFace Hub repository ID (e.g., "username/model-name")
        commit_message: Commit message for the upload
        private: Whether to create private repository
        token: HuggingFace Hub API token (optional)
        config_path: Path to model config for model card generation
        repository_url: URL to source code repository

    Returns:
        URL of the uploaded repository
    """
    try:
        from huggingface_hub import HfApi, upload_folder
    except ImportError:
        print(
            "Error: 'huggingface_hub' not installed. Please run 'pip install huggingface_hub'."
        )
        return ""

    print(f"Pushing vLLM model to HuggingFace Hub: {repo_id}...")

    # Generate model card
    try:
        from model_card_utils import generate_model_card, save_model_card
        from better_ai.config import ModelConfig
        import json

        # Load config for model card
        config = None
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = json.load(f)
            config = ModelConfig.from_dict(config_data)
        else:
            # Try to load from vllm config
            vllm_config_path = os.path.join(output_dir, "config.json")
            if os.path.exists(vllm_config_path):
                with open(vllm_config_path, "r") as f:
                    vllm_config = json.load(f)
                if "better_ai_config" in vllm_config:
                    config = ModelConfig.from_dict(vllm_config["better_ai_config"])

        model_name = repo_id.split("/")[-1]
        model_card = generate_model_card(
            model_name=model_name,
            export_format="vLLM",
            config=config,
            repository_url=repository_url,
            tags=["vllm", "deepseek"],
        )

        # Save model card to output directory
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(model_card)
        print("Model card (README.md) generated successfully")

    except Exception as e:
        print(f"Warning: Could not generate model card: {e}")

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

    # Upload folder contents
    try:
        upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            commit_message=commit_message,
            token=token,
        )

        print(f"Successfully uploaded to: https://huggingface.co/{repo_id}")
        return f"https://huggingface.co/{repo_id}"

    except Exception as e:
        print(f"Error uploading files: {e}")
        return ""


def main():
    """CLI entry point for vLLM export."""
    parser = argparse.ArgumentParser(
        description="Export Better AI model to vLLM-compatible format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to local directory
  python scripts/export_to_vllm.py --model_path model.pt --output_dir ./vllm_model

  # Export with tensor parallelism
  python scripts/export_to_vllm.py --model_path model.pt --output_dir ./vllm_model --tensor_parallel_size 2

  # Export and push to HuggingFace Hub
  python scripts/export_to_vllm.py --model_path model.pt --output_dir ./vllm_model --push_to_hub --repo_id username/model-name

  # Export with custom config and push privately
  python scripts/export_to_vllm.py --model_path model.pt --output_dir ./vllm_model --config_path config.json --push_to_hub --repo_id username/model-name --private
        """,
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
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the exported model to HuggingFace Hub",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="HuggingFace Hub repository ID (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository on HuggingFace Hub",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload vLLM model",
        help="Commit message for HuggingFace Hub upload",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace Hub API token (optional, will use cached token if not provided)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the exported model by loading it with VLLMDeepSeekModel",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.push_to_hub and args.repo_id is None:
        parser.error("--repo_id is required when using --push_to_hub")

    if not os.path.exists(args.model_path):
        parser.error(f"Model path does not exist: {args.model_path}")

    if args.config_path and not os.path.exists(args.config_path):
        parser.error(f"Config path does not exist: {args.config_path}")

    # Export model
    print(f"Exporting model from {args.model_path} to vLLM format...")
    print(f"Output directory: {args.output_dir}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")

    export_for_vllm(
        model_path=args.model_path,
        output_dir=args.output_dir,
        config_path=args.config_path,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # Verify export if requested
    if args.verify:
        print("\nVerifying exported model...")
        try:
            from better_ai.config import ModelConfig

            # Load config
            config_path = os.path.join(args.output_dir, "config.json")
            with open(config_path, "r") as f:
                vllm_config = json.load(f)

            # Create config object
            config = ModelConfig(
                vocab_size=vllm_config["vocab_size"],
                hidden_dim=vllm_config["hidden_size"],
                num_layers=vllm_config["num_hidden_layers"],
                num_attention_heads=vllm_config["num_attention_heads"],
                num_key_value_heads=vllm_config.get("num_key_value_heads"),
                intermediate_size=vllm_config["intermediate_size"],
                max_seq_length=vllm_config["max_position_embeddings"],
                norm_eps=vllm_config["rms_norm_eps"],
            )

            # Initialize wrapper
            wrapper = VLLMDeepSeekModel(config, weights_dir=args.output_dir)

            # Test forward pass
            import torch

            batch_size, seq_len = 1, 10
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            positions = torch.arange(seq_len).unsqueeze(0)

            with torch.no_grad():
                outputs = wrapper(input_ids, positions, [])

            print(f"✓ Model verification successful!")
            print(f"  Output shape: {outputs.shape}")
            print(f"  Expected: ({batch_size}, {seq_len}, {config.vocab_size})")

        except Exception as e:
            print(f"✗ Model verification failed: {e}")
            import traceback

            traceback.print_exc()

    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        push_to_huggingface(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            commit_message=args.commit_message,
            private=args.private,
            token=args.token,
            config_path=args.config_path,
        )


if __name__ == "__main__":
    main()
