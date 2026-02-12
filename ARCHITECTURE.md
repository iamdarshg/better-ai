# Better AI Architecture

DeepSeek-inspired RLHF system for coding models with MoE architecture, Ring Attention, and advanced reasoning mechanisms.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Better AI Stack                         │
├─────────────────────────────────────────────────────────────┤
│  Application Layer                                          │
│  ├── InferenceEngine (KV-cache, streaming)                  │
│  ├── TextGenerator (generation config)                      │
│  └── RLHFEvaluator (benchmarks, metrics)                    │
├─────────────────────────────────────────────────────────────┤
│  Training Layer                                             │
│  ├── CurriculumMCTSTrainer (main production trainer)        │
│  │   ├── CosineCurriculumScheduler                          │
│  │   └── MCTSCoTSearcher                                    │
│  ├── GRPOTrainer (group policy optimization)                │
│  │   └── KVCacheManager (40% memory reduction)              │
│  ├── ARPOTrainer (entropy-based rollouts)                   │
│  └── EnhancedMoETrainer (base MoE training)                 │
├─────────────────────────────────────────────────────────────┤
│  Model Layer                                                │
│  ├── DeepSeekMoEModel (main model class)                    │
│  │   ├── MoELayer (8 experts, top-2 routing)                │
│  │   ├── RingAttention (524k context)                       │
│  │   └── TransformerBlock × 16                              │
│  ├── Reward Models                                          │
│  │   ├── BranchRewardModel (4 attributes)                   │
│  │   └── HierarchicalRewardModel                            │
│  └── Feature Modules                                        │
│      ├── CoTSpecializationHeads                             │
│      ├── RecursiveScratchpad                                │
│      ├── STaRModule                                         │
│      └── ToolUseHeads                                       │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                 │
│  ├── UnifiedDataLoader (stages: pretrain→sft→rlhf)          │
│  ├── CodeDataset (The Stack v2, Magicoder)                  │
│  └── DatasetConfig (datasets.yml)                           │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. DeepSeekMoEModel (`better_ai/models/moe.py`)

Main model architecture combining MoE with transformer blocks.

**Configuration (default production):**
```python
ModelConfig(
    vocab_size=64000,
    hidden_dim=1536,
    num_layers=16,
    num_attention_heads=24,
    num_key_value_heads=12,  # GQA 2:1 ratio
    num_experts=8,
    num_experts_per_token=2,
    intermediate_dim=16384,
    max_seq_length=524288,  # 524k with Ring Attention
)
```

**Key Classes:**
- `DeepSeekMoEModel` - Main model (wrapper around transformer stack)
- `MoELayer` - Mixture of Experts layer
- `Expert` - Individual expert network (FFN)
- `ExpertRouter` - Top-k routing with load balancing

### 2. Ring Attention (`better_ai/models/ring_attention.py`)

Distributed attention mechanism for processing sequences longer than single GPU memory allows.

**Implementation:**
```python
class RingAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, block_size=1024):
        # Shards attention computation across devices
        # Uses ring topology: device N talks to device (N+1) % world_size
```

**Usage:**
```python
config = ModelConfig(use_ring_attention=True, ring_block_size=1024)
model = DeepSeekMoEModel(config)  # Enables 524k context
```

### 3. Reward Models (`better_ai/models/reward_model.py`)

**BranchRewardModel:**
- 4 branch heads: Correctness, Efficiency, Readability, Robustness
- Rethinking module for refinement
- Output: scalar reward [0, 1]

```python
class BranchRewardModel(nn.Module):
    def forward(self, hidden_states, return_branch_scores=True):
        # Returns: (reward, {correctness, efficiency, readability, robustness})
```

**HierarchicalRewardModel:**
- Combines single-step (BR-RM) and end-to-end scoring
- Weighted combination for holistic quality assessment

### 4. Training Infrastructure

#### CurriculumMCTSTrainer (`better_ai/training/curriculum_mcts_trainer.py`)

Production trainer combining curriculum learning with Monte Carlo Tree Search.

```python
from better_ai.training import CurriculumMCTSTrainer

trainer = CurriculumMCTSTrainer(
    model=model,
    reward_model=reward_model,
    config=CurriculumMCTSConfig(
        enable_curriculum=True,
        enable_mcts=True,
        mcts_frequency=5,  # Every 5 steps
        mcts_data_ratio=0.3,
    )
)

trainer.train_with_curriculum(train_loader, num_epochs=3)
```

**Components:**
- `CosineCurriculumScheduler` - Smooth difficulty progression
- `MCTSCoTSearcher` - Tree search for CoT data generation
- `GRPOTrainer` - Integrated policy optimization

#### GRPO with KV-Cache (`better_ai/training/grpo.py`, `better_ai/training/kv_cache_grpo.py`)

Group Reward Policy Optimization with memory optimization.

```python
from better_ai.training import GRPOTrainer
from better_ai.training.kv_cache_grpo import KVCacheManager

grpo_trainer = GRPOTrainer(model, reward_model, config)

# With KV-cache optimization
cache_manager = KVCacheManager(max_cache_size=1000, cache_dim=1536)
```

**Memory Savings:** 40% reduction through sequential generation with cache reuse

### 5. Reasoning Features (`better_ai/models/features/`)

#### CoT Specialization (`cot_specialization.py`)
```python
config = ModelConfig(use_cot_specialization=True, cot_num_heads=5)
```
- Isolates reasoning tokens from final output
- 5 specialized attention heads
- Prevents reasoning pollution

#### Recursive Scratchpad (`recursive_scratchpad.py`)
```python
config = ModelConfig(
    use_recursive_scratchpad=True,
    scratchpad_max_iterations=8,
)
```
- Iterative refinement up to 8 iterations
- Automatic stopping based on confidence
- Attention-based scratchpad state

#### STaR Module (`star_module.py`)
```python
config = ModelConfig(use_star=True, star_bootstrap_rounds=3)
```
- Self-taught reasoning with bootstrap learning
- Consistency checking across multiple paths
- Learns from successful reasoning traces

### 6. Data Pipeline (`better_ai/data/`)

**UnifiedDataLoader** (`unified_dataloader.py`):
```python
from better_ai.data.unified_dataloader import create_dataloader
from better_ai.data.dataset_config import load_datasets_by_stage

# Load datasets for specific stage
pretraining_datasets = load_datasets_by_stage('pretraining')
train_loader = create_dataloader(pretraining_datasets, batch_size=32)
```

**Dataset Configuration** (`datasets.yml`):
```yaml
datasets:
  - name: "The Stack"
    path: "/data/the_stack"
    max_seq_length: 8192
    num_training_steps: 100000
```

## Training Pipeline

### 5-Stage Pipeline

```
Stage 1: Pretraining (1-2 weeks)
├── Dataset: The Stack v2 (multi-language code)
├── Objective: Next-token prediction
├── Output: Base model with code understanding
└── Implementation: main_workflow.py --stage pretrain

Stage 2: SFT (3-5 days)
├── Dataset: Magicoder + Code-Feedback
├── Mix: 75% single-turn, 25% multi-turn
├── Objective: Instruction following
└── Implementation: main_workflow.py --stage sft

Stage 3: RLHF Stage 1 (5-7 days)
├── Dataset: CodeUltraFeedback (10k pairs)
├── Algorithm: GRPO with BR-RM
├── Objective: Human preference alignment
└── Implementation: main_workflow.py --stage rlhf

Stage 4: RLHF Stage 2 (3-5 days)
├── Dataset: RLVR Coding (80k traces)
├── Features: Multi-attribute + STaR
├── Objective: Advanced reasoning
└── Implementation: main_workflow.py --stage rlhf2

Stage 5: Security DPO (2-3 days)
├── Dataset: CVE datasets
├── Focus: Vulnerability repair, memory safety
└── Implementation: main_workflow.py --stage security
```

### Running the Pipeline

```bash
# Full pipeline
python better_ai/scripts/main_workflow.py --stage full

# Individual stages with mock data for testing
python train_enhanced.py --stage pretrain --test
python train_enhanced.py --stage sft --test
python train_enhanced.py --stage rlhf --test
```

## Memory Optimizations

### FP8 Quantization (`better_ai/optimizers/fp8.py`)
```python
from better_ai.optimizers import FP8AdamW

optimizer = FP8AdamW(model.parameters(), lr=1e-4)
```
- E4M3 for forward pass, E5M2 for gradients
- 50% memory reduction

### Gradient Checkpointing
```python
config = ModelConfig(use_gradient_checkpointing=True)
```
- Selective checkpointing for MoE layers
- 25% memory savings

### Expert Management (`better_ai/training/expert_manager.py`)
```python
from better_ai.training import ExpertSpecializationManager

manager = ExpertSpecializationManager(num_experts=8)
manager.update_specialization(expert_loads, losses)
```
- Tracks expert utilization
- Prevents expert collapse
- Dynamic capacity adjustment

## Evaluation

### Benchmarks (`better_ai/training/evaluation.py`)

```python
from better_ai.training.evaluation import (
    RLHFEvaluator,
    CodingBenchmarkEvaluator,
    MetricsAggregator,
)

evaluator = RLHFEvaluator(model, reward_model, device)
metrics = evaluator.evaluate(test_data)

# Coding benchmarks
benchmark = CodingBenchmarkEvaluator()
results = benchmark.evaluate_humaneval(model)
```

**Supported Benchmarks:**
- HumanEval (Python code generation)
- MBPP (Python coding problems)
- SWE-bench (software engineering tasks)
- Custom coding tasks

## Configuration Reference

### Model Config (`better_ai/config.py`)

```python
@dataclass
class ModelConfig:
    # Architecture
    vocab_size: int = 64000
    hidden_dim: int = 1536
    num_layers: int = 16
    num_attention_heads: int = 24
    num_key_value_heads: int = 12  # GQA
    
    # MoE
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    
    # Attention
    use_ring_attention: bool = False
    use_flash_attention: bool = True
    max_seq_length: int = 524288
    
    # Features
    use_cot_specialization: bool = False
    use_recursive_scratchpad: bool = False
    use_star: bool = False
```

### Training Config

```python
@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_steps: int = 100000
    gradient_accumulation_steps: int = 4
    
    # Optimization
    use_fp8: bool = False
    bf16: bool = True
    
    # Curriculum + MCTS
    use_mcts: bool = False
    mcts_frequency: int = 5
```

## Implementation Files Reference

| Component | File | Key Classes |
|-----------|------|-------------|
| Model | `models/moe.py` | DeepSeekMoEModel, MoELayer |
| Attention | `models/ring_attention.py` | RingAttention |
| Rewards | `models/reward_model.py` | BranchRewardModel, HierarchicalRewardModel |
| Main Trainer | `training/curriculum_mcts_trainer.py` | CurriculumMCTSTrainer |
| GRPO | `training/grpo.py` | GRPOTrainer |
| ARPO | `training/arpo.py` | ARPOTrainer |
| MCTS | `training/mcts_cot.py` | MCTSCoTSearcher |
| Curriculum | `training/cosine_curriculum.py` | CosineCurriculumScheduler |
| Inference | `inference/engine.py` | InferenceEngine |
| FP8 Optimizer | `optimizers/fp8.py` | FP8AdamW |

## Hardware Requirements

| Setup | GPUs | Memory | Training Time |
|-------|------|--------|---------------|
| Minimum | 1× A100 | 40GB | ~3 weeks |
| Recommended | 8× A100 | 80GB | ~1 week |
| Production | 8× H100 | 80GB | ~5 days |

**Memory per component:**
- Base model (16 layers, 1536 dim): ~12GB
- With Ring Attention (524k context): +8GB
- With MoE (8 experts): +4GB
- With KV-cache for GRPO: +6GB
- FP8 quantization: -50% from above
