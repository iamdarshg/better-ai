# Better AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Last Commit](https://img.shields.io/github/last-commit/iamdarshg/better-ai)](https://github.com/iamdarshg/better-ai/commits)
[![Language](https://img.shields.io/github/languages/top/iamdarshg/better-ai)](https://github.com/iamdarshg/better-ai)
[![Build Status](https://img.shields.io/github/actions/workflow/status/iamdarshg/better-ai/training_test.yml)](https://github.com/iamdarshg/better-ai/actions)


Production-grade RLHF training system for coding models. Built on DeepSeek architecture with Mixture of Experts, advanced reasoning mechanisms, and multi-stage reinforcement learning.

## What Makes It Different

Better AI combines proven techniques into a cohesive training pipeline:

- **Striped Attention** — Optimized long-context processing for edge and distributed systems
- **GRPO with KV-Cache Reuse** — Memory-optimized policy gradient training
- **Hierarchical Reward Models** — Multi-attribute scoring for code quality (correctness, efficiency, readability, robustness)
- **MCTS for Chain-of-Thought** — Tree search constructs high-quality reasoning data
- **Security Hardening** — Dedicated DPO phase for CVE repair and vulnerability mitigation

## Installation

```bash
git clone https://github.com/iamdarshg/better-ai.git
cd better-ai
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 2.0+, CUDA 11.8+, 20GB+ VRAM

## Cloud Deployment (DigitalOcean)

For high-performance training and inference, we provide automated provisioning scripts for DigitalOcean GPU Droplets.

### Recommended Specs
- **Minimum for Inference/SFT:** H100 80GB (~$10-15/hr)
- **Recommended for Training:** 2x H100 or 8x H100 (Multi-GPU)
- **Operating System:** Ubuntu 22.04 LTS

### Automated Provisioning
1.  **Configure:** Update `infra/droplet_config.yml` with your region, SSH keys (comma-separated), and preferred GPU size.
2.  **Authenticate:** Ensure `doctl` is installed and authenticated (`doctl auth init`).
3.  **Deploy:** Run the creation script to spin up and provision the Droplet (including volume creation/attachment):
    ```bash
    bash infra/do_create_droplet.sh
    ```
4.  **Login:** Once the script finishes, SSH into your Droplet. The environment is fully configured in `/app/better-ai`.

### Teardown
To avoid unnecessary costs, run the teardown script which will snapshot your volumes and destroy the Droplet:
```bash
bash infra/do_teardown.sh
```

> **Note:** Block storage volumes are preserved after Droplet destruction. You must manually delete them in the DigitalOcean console or via `doctl` to stop all billing.

## Quick Start

### Train from Scratch

```bash
# Full pipeline: Pretrain → SFT → RLHF → Security
python better_ai/scripts/main_workflow.py --stage full

# Or individual stages
python train_enhanced.py --stage pretrain --test
python train_enhanced.py --stage rlhf --test
```

### Python API

```python
from better_ai.config import ModelConfig, TrainingConfig
from better_ai.models import DeepSeekMoEModel
from better_ai.training import CurriculumMCTSTrainer

# Configure model
model_config = ModelConfig(
    vocab_size=64000,
    hidden_dim=1536,
    num_layers=16,
    num_experts=8,
    use_striped_attention=True,
)

# Configure training
training_config = TrainingConfig(
    batch_size=32,
    learning_rate=1e-4,
    max_steps=100000,
    use_mcts=True,
)

# Initialize and train
model = DeepSeekMoEModel(model_config)
trainer = CurriculumMCTSTrainer(model, training_config)
trainer.train_with_curriculum(train_loader)
```

## Architecture

```
DeepSeekMoEModel
├── Striped Attention (edge-optimized long context)
├── MoE Layers (8 experts, top-2 routing)
├── Reasoning Features
│   ├── Recursive Scratchpad (iterative refinement)
│   ├── CoT Specialization (isolated reasoning heads)
│   ├── MCTS Tree Search
│   └── STaR Self-Learning
└── Reward Stack
    ├── Branch Reward Model (4 attributes)
    └── Hierarchical Reward Model
```

**Specs:** 64k vocab | 1,536 hidden | 16 layers | 8 experts | GQA attention

## Training Pipeline

| Stage | Dataset | Duration | Purpose |
|-------|---------|----------|---------|
| Pretrain | The Stack v2 | 1-2 weeks | Foundation |
| SFT | Magicoder + Code-Feedback | 3-5 days | Instruction following |
| RLHF Stage 1 | CodeUltraFeedback | 5-7 days | Preference alignment |
| RLHF Stage 2 | RLVR Coding | 3-5 days | Advanced reasoning |
| Security | CVE datasets | 2-3 days | Vulnerability hardening |

## Key Features

### Implemented & Tested

- **CurriculumMCTSTrainer** — Combines cosine curriculum with Monte Carlo Tree Search for CoT data generation
- **ARPO** — Agentic Reinforced Policy Optimization with entropy-based adaptive rollouts
- **GRPO** — Group Reward Policy Optimization with KV-cache reuse (40% memory reduction)
- **Striped Attention** — Load-balanced causal attention for 524k+ contexts
- **BR-RM** — Branch Reward Model scores code on correctness, efficiency, readability, robustness
- **Security DPO** — Final training phase for memory safety and prompt injection resistance
- **Tool-Use Heads** — Specialized prediction for API calls
- **JSON Enforcement** — Grammar-based constraints ensure valid outputs
- **STaR Module** — Self-taught reasoning with bootstrap learning
- **FP8 Quantization** — E4M3/E5M2 mixed precision for 50% memory savings

## Configuration

```python
# Minimal (testing)
ModelConfig(vocab_size=32000, hidden_dim=768, num_layers=8)

# Production
ModelConfig(
    vocab_size=64000,
    hidden_dim=1536,
    num_layers=16,
    num_experts=8,
    use_striped_attention=True,
    use_recursive_scratchpad=True,
    use_cot_specialization=True,
)
```

## Testing

```bash
# All tests
pytest tests/

# Specific components
pytest tests/unit/test_grpo_implementation.py
pytest tests/integration/test_mf_rlhf.py
python tests/test_curriculum_mcts.py
```

## Project Structure

```
better-ai/
├── better_ai/                      # Main package
│   ├── __init__.py                # Package exports
│   ├── config.py                  # ModelConfig, TrainingConfig, InferenceConfig
│   ├── analysis_scripts/
│   │   └── plot_scaling.py        # Scaling analysis plots
│   ├── data/                      # Data loading & processing
│   │   ├── __init__.py
│   │   ├── curation.py            # Agent-FLAN style data curation
│   │   ├── dataset.py             # Base dataset classes
│   │   ├── dataset_config.py      # Dataset configuration management
│   │   ├── hf_datasets.py         # HuggingFace dataset integration
│   │   ├── unified_dataloader.py  # Unified data loading interface
│   │   └── datasets/              # Dataset implementations
│   │       ├── code_dataset.py
│   │       ├── expert_aware_dataset.py
│   │       ├── mixed_code_dataset.py
│   │       └── rolling_window_dataset.py
│   ├── inference/                 # Inference engine
│   │   ├── __init__.py
│   │   ├── engine.py              # InferenceEngine with KV-cache
│   │   └── generator.py           # TextGenerator, GenerationConfig
│   ├── models/                    # Model architectures
│   │   ├── __init__.py            # Model exports
│   │   ├── core.py                # DeepSeekModel, TransformerBlock
│   │   ├── moe.py                 # DeepSeekMoEModel, Expert, MoELayer
│   │   ├── moe_optimized.py       # Optimized MoE variants
│   │   ├── optimized_model.py     # OptimizedDeepSeekMoEModel
│   │   ├── attention.py           # FlashMultiHeadAttention
│   │   ├── attention_optimized.py # Optimized attention variants
│   │   ├── striped_attention.py   # Striped Attention for long contexts
│   │   ├── dsa.py                 # Distributed attention
│   │   ├── rope.py                # RoPE positional embeddings
│   │   ├── reward_model.py        # BranchRewardModel, HierarchicalRewardModel
│   │   ├── generation.py          # Generation utilities
│   │   ├── advanced_features.py   # Feature integrations
│   │   ├── enhanced_model.py      # Enhanced model variants
│   │   ├── tidar.py               # TiDAR diffusion module
│   │   ├── tot.py                 # Tree-of-Thought implementation
│   │   └── features/              # Specialized feature modules
│   │       ├── cot_specialization.py      # CoT attention heads
│   │       ├── recursive_scratchpad.py    # Iterative reasoning
│   │       ├── inner_monologue.py         # Private reasoning subspaces
│   │       ├── star_module.py             # Self-taught reasoning
│   │       ├── tool_use.py                # Tool-use prediction heads
│   │       ├── gbnf_constraint.py         # Grammar constraints
│   │       ├── json_enforcer.py           # JSON output enforcement
│   │       ├── entropic_steering.py       # Entropy-based steering
│   │       ├── specialized_head.py        # Specialized attention heads
│   │       └── reasoning_rewards.py       # Reasoning reward functions
│   ├── optimizers/                # Custom optimizers
│   │   ├── __init__.py
│   │   ├── fp8.py                 # FP8AdamW, FP8Optimizer
│   │   └── memory.py              # Memory-efficient optimizers
│   ├── scripts/                   # Training scripts
│   │   ├── __init__.py
│   │   └── main_workflow.py       # Main production training pipeline
│   ├── training/                  # Training algorithms
│   │   ├── __init__.py            # Training exports
│   │   ├── trainer.py             # Base Trainer class
│   │   ├── enhanced_trainer.py    # EnhancedMoETrainer
│   │   ├── integrated_trainer.py  # IntegratedAdvancedTrainer
│   │   ├── curriculum_mcts_trainer.py  # Curriculum + MCTS trainer
│   │   ├── grpo.py                # GRPOTrainer, GRPOLoss
│   │   ├── kv_cache_grpo.py       # Memory-optimized GRPO
│   │   ├── arpo.py                # ARPOTrainer (Agentic RL)
│   │   ├── steca.py               # STeCa (Trajectory Calibration)
│   │   ├── cleaner.py             # CLEANER self-purification
│   │   ├── mcts_cot.py            # MCTSCoTSearcher
│   │   ├── cosine_curriculum.py   # CosineCurriculumScheduler
│   │   ├── machine_feedback.py    # MachineFeedbackTrainer
│   │   ├── fault_localization.py  # Fault localization pipeline
│   │   ├── rlvr_security.py       # Security DPO phase
│   │   ├── evaluation.py          # RLHFEvaluator, benchmarks
│   │   ├── checkpointing.py       # SelectiveCheckpointManager
│   │   ├── expert_manager.py      # ExpertSpecializationManager
│   │   ├── coherence_scheduler.py # CoherenceBasedScheduler
│   │   ├── adaptive_optimizations.py  # Dynamic capacity management
│   │   ├── diversity_metrics.py   # Reasoning diversity metrics
│   │   ├── pruning.py             # Model pruning utilities
│   │   ├── tui.py                 # Training TUI
│   │   └── trainer_utils/         # Training utilities
│   │       ├── callbacks.py
│   │       ├── data.py
│   │       ├── optimization.py
│   │       └── rl.py
│   ├── utils/                     # Utilities
│   │   ├── __init__.py
│   │   ├── exceptions.py          # Custom exceptions
│   │   ├── react_notebook.py      # ReAct notebook format
│   │   └── verification.py        # Math/code verification
│   ├── test_config_utils.py       # Test utilities
│   └── test_resource_tags.py      # Resource tagging
├── tests/                         # Test suite
│   ├── conftest.py                # Pytest configuration
│   ├── low_resource_test.py       # Low-resource testing
│   ├── test_advanced_features.py  # Feature tests
│   ├── test_curriculum_mcts.py    # MCTS + curriculum tests
│   ├── test_grpo_implementation.py # GRPO tests
│   ├── test_rl_stage2.py          # RLHF stage 2 tests
│   ├── test_striped_attn.py       # Striped attention tests
│   ├── test_tidar_init.py         # TiDAR tests
│   ├── e2e/                       # End-to-end tests
│   │   └── __init__.py
│   ├── integration/               # Integration tests
│   │   ├── __init__.py
│   │   ├── test_hrm.py            # HRM integration
│   │   ├── test_low_resource_integration.py
│   │   ├── test_mf_rlhf.py        # Machine feedback RLHF
│   │   ├── test_refactored_modules.py
│   │   ├── test_rlhf_components.py
│   │   └── test_tot.py            # ToT integration
│   └── unit/                      # Unit tests
│       ├── __init__.py
│       ├── test_advanced_features.py
│       ├── test_arpo.py
│       ├── test_cleaner.py
│       ├── test_config_validation.py
│       ├── test_dataloader.py
│       ├── test_dataset_config.py
│       ├── test_exceptions.py
│       ├── test_integrated_trainer.py
│       ├── test_kv_cache_grpo.py
│       ├── test_length_dpo.py
│       ├── test_memory_optimization.py
│       ├── test_model_enhancements.py
│       ├── test_new_phase7_features.py
│       ├── test_pruning.py
│       ├── test_react_notebook.py
│       ├── test_security_workflow.py
│       └── test_striped_attn.py
├── tools/                         # Development tools
│   ├── list_high_resource_test_ids.py
│   ├── profile_high_resource_tests.py
│   ├── run_low_resource_tests.py
│   ├── run_test_with_torch_profiler.py
│   ├── runtest.py                 # Test runner
│   └── trigger_profile_dispatch.py
├── checkpoints/                   # Model checkpoints
├── logs/                          # Training logs
├── example_advanced_features.py   # Feature examples
├── train_enhanced.py             # CLI training script
├── setup.py                      # Package setup
├── setup.sh                      # Setup script
├── requirements.txt              # Dependencies
├── datasets.yml                  # Dataset configuration
├── ARCHITECTURE.md              # Architecture docs
├── QUICKSTART.md                # Quick start guide
├── todo.md                      # Development roadmap
├── README.md                    # This file
└── LICENSE                      # MIT License
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — Detailed system design
- [QUICKSTART.md](QUICKSTART.md) — Step-by-step tutorials
- [todo.md](todo.md) — Implementation roadmap

## Performance

| Metric | Result |
|--------|--------|
| Memory (FP8) | 50% reduction |
| Context Length | 524,288 tokens |
| Expert Routing | Top-2 of 8 |
| KV-Cache GRPO | 40% memory saved |

Benchmarks: HumanEval, MBPP, SWE-bench

## Contributing

```bash
# Setup
pip install -e .

# Before submitting
black better_ai/
pytest tests/
```

Issues and PRs welcome at [github.com/iamdarshg/better-ai](https://github.com/iamdarshg/better-ai)

## License

MIT — see [LICENSE](LICENSE)

## Resource Estimates

*Auto-generated from config using production settings*
*Last updated: 2026-05-06 08:11:21*

### Model Architecture
- **Total Parameters**: 5.72B
- **Active Parameters**: 3.29B (per token with MoE)
- **Sparsity**: 42.5%
- **Layers**: 8 (5 standard + 3 MoE)
- **Hidden Dimension**: 4096
- **Intermediate Dimension**: 11000
- **Max Sequence Length**: 524,288 tokens
- **Average Sequence Length**: 142,426 tokens (step-weighted from datasets)

**Parameter Breakdown:**
| Component | Parameters | % of Total |
|-----------|------------|------------|
| Embeddings | 262.14M | 4.6% |
| LM Head | 262.14M | 4.6% |
| Attention | 301.99M | 5.3% |
| FFN (Standard) | 675.84M | 11.8% |
| FFN (MoE - All) | 3.65B | 63.8% |
| FFN (MoE - Active) | 1.22B | 37.0% of active |
| Routers | 98304 | 0.0% |

**Advanced Features:**
| Feature | Parameters |
|---------|------------|
| tidar | 486.54M |
| tool_heads | 54.53M |
| scratchpad | 30.54M |
| **Total Features** | **571.61M** |

### VRAM Requirements (Batch=1, Seq=524,288)

**Inference VRAM per GPU:**
| GPU | Available | BF16 Required | FP8 Required | BF16 Batch | FP8 Batch |
|-----|-----------|---------------|--------------|------------|-----------|
| RTX 2070 | 8 GB | 50 GB | 25 GB | 0 (need 50GB) | 0 (need 25GB) |
| RTX 5090 | 32 GB | 50 GB | 25 GB | 0 (need 50GB) | 1 |
| H300e | 80 GB | 50 GB | 25 GB | 1 | 3 |
| H200 | 141 GB | 50 GB | 25 GB | 3 | 6 |

**Training VRAM per GPU (with 8-bit optimizer, avg seq length):**
| GPU | Available | BF16 Required | FP8 Required | BF16 Batch | FP8 Batch |
|-----|-----------|---------------|--------------|------------|-----------|
| RTX 2070 | 8 GB | 93 GB | 87 GB | 0 (need 93GB) | 0 (need 87GB) |
| RTX 5090 | 32 GB | 93 GB | 87 GB | 0 (need 93GB) | 0 (need 87GB) |
| H300e | 80 GB | 93 GB | 87 GB | 0 (need 93GB) | 0 (need 87GB) |
| H200 | 141 GB | 93 GB | 87 GB | 6 | 7 |

### Training Pipeline

**Total Steps**: 2,272,750

| Stage | Steps |
|-------|-------|
| pretraining | 1,122,000 |
| rlhf | 299,250 |
| security_dpo | 129,000 |
| sft | 722,500 |

### Training Time Estimates (100% utilization)

| GPU | FP16 TFLOPS | Hours | Days |
|-----|-------------|-------|------|
| RTX 2070 | 8 | 2,960 | 123.3 |
| RTX 5090 | 100 | 222 | 9.3 |
| H300e | 2000 | 11 | 0.5 |
| H200 | 1000 | 22 | 0.9 |

_Note: Training times assume 100% GPU utilization. Real-world training typically achieves 30-70%._
