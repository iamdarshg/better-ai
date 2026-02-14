"""Model card generation utilities for HuggingFace Hub uploads."""

import os
from typing import Optional, Dict, Any
from datetime import datetime


def generate_model_card(
    model_name: str,
    export_format: str,
    config: Optional[Any] = None,
    repository_url: str = "https://github.com/iamdarshg7/better-ai",
    tags: Optional[list] = None,
) -> str:
    """Generate a HuggingFace model card with repository links.

    Args:
        model_name: Name of the model
        export_format: Format of export (e.g., "GGUF", "vLLM")
        config: Model configuration object (optional)
        repository_url: URL to the source code repository
        tags: Additional tags for the model card

    Returns:
        Model card content as markdown string
    """
    if tags is None:
        tags = []

    # Base tags
    base_tags = ["deepseek", "better-ai", export_format.lower()]
    if config and getattr(config, "num_experts", 0) > 0:
        base_tags.append("mixture-of-experts")
        base_tags.append("moe")
    all_tags = base_tags + tags

    # Pipeline tag
    pipeline_tag = "text-generation"

    # Generate tags section
    tags_section = "\n".join([f"- {tag}" for tag in all_tags])

    # Advanced features section
    advanced_features = []
    if config:
        if getattr(config, "use_recursive_scratchpad", False):
            advanced_features.append("- Recursive Scratchpad for iterative reasoning")
        if getattr(config, "use_tidar", False):
            advanced_features.append("- TiDAR (Temporal Diffusion-Augmented Reasoning)")
        if getattr(config, "use_cot_specialization", False):
            advanced_features.append("- Chain-of-Thought Specialization")
        if getattr(config, "use_inner_monologue", False):
            advanced_features.append("- Inner Monologue for thought processes")
        if getattr(config, "use_star", False):
            advanced_features.append("- STaR (Self-Taught Reasoner)")
        if getattr(config, "use_tool_heads", False):
            advanced_features.append("- Tool Use capabilities")
        if getattr(config, "use_entropic_steering", False):
            advanced_features.append("- Entropic Steering")
        if getattr(config, "use_reward_models", False):
            advanced_features.append("- Reward Models for RLHF")

    advanced_section = (
        "\n".join(advanced_features) if advanced_features else "None configured"
    )

    # Model config summary
    config_summary = ""
    if config:
        config_summary = f"""
### Model Configuration

- **Architecture**: DeepSeek-inspired Transformer
- **Hidden Size**: {config.hidden_dim}
- **Layers**: {config.num_layers}
- **Attention Heads**: {config.num_attention_heads}
- **Vocabulary Size**: {config.vocab_size}
- **Max Sequence Length**: {config.max_seq_length}
- **MoE Experts**: {getattr(config, "num_experts", "N/A")}
"""

    model_card = f"""---
tags:
{tags_section}
pipeline_tag: {pipeline_tag}
---

# {model_name}

This model was exported from the [Better AI]({repository_url}) project using the {export_format} format.

## Model Description

{model_name} is a DeepSeek-inspired language model with advanced reasoning capabilities and Mixture of Experts (MoE) architecture.

{config_summary}

## Advanced Features

{advanced_section}

## Source Code

This model is part of the **Better AI** project:
- **Repository**: [{repository_url}]({repository_url})
- **Documentation**: See repository README and docs/
- **Issues**: Please report issues at the repository issue tracker

## Usage

### {export_format} Format

This model is distributed in {export_format} format, which is optimized for:
"""

    if export_format == "GGUF":
        model_card += """
- [Ollama](https://ollama.ai) - Local LLM runner
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - C++ inference engine
- [LM Studio](https://lmstudio.ai) - Desktop GUI for local models

```bash
# Using Ollama
ollama run <model-name>

# Using llama.cpp
./main -m model.gguf -p "Your prompt here"
```
"""
    elif export_format == "vLLM":
        model_card += """
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput inference engine
- [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)

```python
from better_ai.inference.vllm_compat import VLLMDeepSeekModel

model = VLLMDeepSeekModel(config, weights_dir="./model")
```
"""

    model_card += f"""
## Citation

If you use this model in your research, please cite the Better AI project:

```bibtex
@software{{better_ai,
  title = {{Better AI: Advanced Language Model Framework}},
  author = {{Darsh Gupta and the Better AI Team}},
  year = {{{datetime.now().year}}},
  url = {{{repository_url}}}
}}
```

## License

Please refer to the LICENSE file in the source repository for licensing information.

## Acknowledgments

This model was created using the Better AI framework, which builds upon DeepSeek architecture and incorporates various advanced features for enhanced reasoning and tool use.

---

*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

    return model_card


def save_model_card(
    model_card: str,
    output_path: str,
) -> None:
    """Save model card to file.

    Args:
        model_card: Model card content
        output_path: Path to save the model card
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(model_card)
    print(f"Model card saved to: {output_path}")
