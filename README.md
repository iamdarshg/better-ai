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

- **Ring Attention** — Process 524k+ token contexts efficiently across multiple GPUs
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
    use_ring_attention=True,
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
├── Ring Attention (distributed, 524k context)
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
- **Ring Attention** — Distributed attention sharding for near-infinite contexts
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
    use_ring_attention=True,
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
│   │   ├── ring_attention.py      # RingAttention for long contexts
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
*Last updated: 2026-02-12 21:37:35*

### Model Architecture
- **Total Parameters**: 15.19B
- **Active Parameters**: 6.13B (per token with MoE)
- **Sparsity**: 59.6%
- **Layers**: 10 (5 standard + 5 MoE)
- **Hidden Dimension**: 3072
- **Intermediate Dimension**: 16384
- **Max Sequence Length**: 524,288 tokens
- **Average Sequence Length**: 142,426 tokens (step-weighted from datasets)

**Parameter Breakdown:**
| Component | Parameters | % of Total |
|-----------|------------|------------|
| Embeddings | 196.61M | 1.3% |
| LM Head | 196.61M | 1.3% |
| Attention | 220.20M | 1.4% |
| FFN (Standard) | 754.97M | 5.0% |
| FFN (MoE - All) | 13.59B | 89.5% |
| FFN (MoE - Active) | 4.53B | 73.9% of active |
| Routers | 245760 | 0.0% |

**Advanced Features:**
| Feature | Parameters |
|---------|------------|
| tidar | 127.93M |
| monologue | 23.59M |
| tool_heads | 44.04M |
| scratchpad | 34.60M |
| **Total Features** | **230.16M** |

### VRAM Requirements (Batch=1, Seq=524,288)

**Inference VRAM per GPU:**
| GPU | Available | BF16 Required | FP8 Required | BF16 Batch | FP8 Batch |
|-----|-----------|---------------|--------------|------------|-----------|
| RTX 2070 | 8 GB | 52 GB | 26 GB | 0 (need 52GB) | 0 (need 26GB) |
| RTX 5090 | 32 GB | 52 GB | 26 GB | 0 (need 52GB) | 1 |
| H300e | 80 GB | 52 GB | 26 GB | 2 | 9 |
| H200 | 141 GB | 52 GB | 26 GB | 7 | 19 |

**Training VRAM per GPU (with 8-bit optimizer, avg seq length):**
| GPU | Available | BF16 Required | FP8 Required | BF16 Batch | FP8 Batch |
|-----|-----------|---------------|--------------|------------|-----------|
| RTX 2070 | 8 GB | 234 GB | 217 GB | 0 (need 234GB) | 0 (need 217GB) |
| RTX 5090 | 32 GB | 234 GB | 217 GB | 0 (need 234GB) | 0 (need 217GB) |
| H300e | 80 GB | 234 GB | 217 GB | 0 (need 234GB) | 0 (need 217GB) |
| H200 | 141 GB | 234 GB | 217 GB | 0 (need 234GB) | 0 (need 217GB) |

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
| RTX 2070 | 8 | 7,855 | 327.3 |
| RTX 5090 | 100 | 589 | 24.5 |
| H300e | 2000 | 29 | 1.2 |
| H200 | 1000 | 59 | 2.5 |

_Note: Training times assume 100% GPU utilization. Real-world training typically achieves 30-70%._
