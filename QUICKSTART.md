# Quick Start Guide

Get started with Better AI in 5 minutes.

## Installation

```bash
git clone https://github.com/iamdarshg/better-ai.git
cd better-ai
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 2.0+, CUDA 11.8+

## Your First Training Run

### CLI Quick Start

```bash
# Test with mock data (no GPU required)
python train_enhanced.py --stage pretrain --test

# Full training pipeline
python better_ai/scripts/main_workflow.py --stage full

# Individual stages
python train_enhanced.py --stage sft --test
python train_enhanced.py --stage rlhf --test
```

### Python API

#### Basic Training

```python
import torch
from better_ai import DeepSeekMoEModel, ModelConfig, TrainingConfig
from better_ai.training import CurriculumMCTSTrainer
from better_ai.data.unified_dataloader import create_dataloader
from better_ai.data.dataset_config import load_datasets_by_stage

# 1. Configure model
model_config = ModelConfig(
    vocab_size=32000,
    hidden_dim=768,
    num_layers=8,
    num_experts=4,
)

# 2. Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepSeekMoEModel(model_config, device=device)

# 3. Load data
pretraining_datasets = load_datasets_by_stage('pretraining')
train_loader = create_dataloader(pretraining_datasets, batch_size=8)

# 4. Configure training
training_config = TrainingConfig(
    batch_size=8,
    learning_rate=1e-4,
    max_steps=1000,
)

# 5. Create trainer and train
trainer = CurriculumMCTSTrainer(
    model=model,
    training_config=training_config,
)
trainer.train_with_curriculum(train_loader, num_epochs=1)
```

#### Production Training with All Features

```python
from better_ai import DeepSeekMoEModel, ModelConfig
from better_ai.models import BranchRewardModel
from better_ai.training import CurriculumMCTSTrainer, CurriculumMCTSConfig
from better_ai.training.cosine_curriculum import CurriculumConfig
from better_ai.training.mcts_cot import MCTSConfig
from better_ai.optimizers import FP8AdamW

# Production model config
model_config = ModelConfig(
    vocab_size=64000,
    hidden_dim=1536,
    num_layers=16,
    num_experts=8,
    use_ring_attention=True,
    use_recursive_scratchpad=True,
    use_cot_specialization=True,
    use_star=True,
)

# Initialize model and reward model
model = DeepSeekMoEModel(model_config)
reward_model = BranchRewardModel(model_config)

# Configure integrated trainer
config = CurriculumMCTSConfig(
    enable_curriculum=True,
    enable_mcts=True,
    mcts_frequency=5,
    mcts_data_ratio=0.3,
    curriculum=CurriculumConfig(
        total_steps=100000,
        structural_weight_end=0.3,
        semantic_weight_end=0.7,
    ),
    mcts=MCTSConfig(
        num_simulations=50,
        exploration_constant=1.414,
    ),
)

# Create FP8 optimizer
optimizer = FP8AdamW(model.parameters(), lr=1e-4)

# Initialize trainer
trainer = CurriculumMCTSTrainer(
    model=model,
    reward_model=reward_model,
    optimizer=optimizer,
    config=config,
)

# Train with curriculum and MCTS
trainer.train_with_curriculum(train_loader, num_epochs=3)
```

## Common Tasks

### Inference

```python
from better_ai import DeepSeekMoEModel, ModelConfig, InferenceConfig
from better_ai.inference import InferenceEngine

# Load model
model = DeepSeekMoEModel(ModelConfig())
checkpoint = torch.load("checkpoints/model.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Create inference engine
inference_config = InferenceConfig(
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    use_kv_cache=True,
)
engine = InferenceEngine(model, inference_config)

# Generate
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
output_ids = engine.generate(input_ids)
```

### Reward Model Scoring

```python
from better_ai.models import BranchRewardModel

# Initialize reward model
reward_model = BranchRewardModel(model_config)

# Get hidden states from model
outputs = model(input_ids, output_hidden_states=True)
hidden_states = outputs.hidden_states[-1]

# Score response
reward, attributes = reward_model(
    hidden_states,
    return_branch_scores=True
)

print(f"Total reward: {reward:.4f}")
print(f"Correctness: {attributes['correctness']:.4f}")
print(f"Efficiency: {attributes['efficiency']:.4f}")
```

### GRPO Training

```python
from better_ai.training import GRPOTrainer
from better_ai.training.grpo import GRPOConfig

# Configure GRPO
grpo_config = GRPOConfig(
    group_size=8,
    clip_epsilon=0.2,
    kl_penalty=0.01,
)

# Create GRPO trainer
grpo_trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    reward_model=reward_model,
    config=grpo_config,
)

# Train on preference pairs
metrics = grpo_trainer.train_step(preference_batch)
print(f"Policy loss: {metrics['policy_loss']:.4f}")
```

### MCTS Reasoning Search

```python
from better_ai.training.mcts_cot import MCTSCoTSearcher, MCTSConfig

# Configure MCTS
config = MCTSConfig(
    num_simulations=100,
    exploration_constant=1.414,
    max_depth=10,
)

searcher = MCTSCoTSearcher(model, reward_model, config)

# Search for best reasoning path
result = searcher.search(
    prompt="def fibonacci(n):",
    max_iterations=100,
)

print(f"Best value: {result.best_value:.4f}")
```

### Evaluation

```python
from better_ai.training.evaluation import RLHFEvaluator, CodingBenchmarkEvaluator

# Initialize evaluators
rlhf_evaluator = RLHFEvaluator(model, reward_model, device)
benchmark_evaluator = CodingBenchmarkEvaluator()

# Evaluate on test set
metrics = rlhf_evaluator.evaluate(test_dataloader)
print(f"Preference accuracy: {metrics['preference_accuracy']:.4f}")

# Run HumanEval benchmark
results = benchmark_evaluator.evaluate_humaneval(model)
print(f"HumanEval pass@1: {results['pass@1']:.4f}")
```

## Configuration Examples

### Minimal (Testing)
```python
ModelConfig(
    vocab_size=32000,
    hidden_dim=768,
    num_layers=8,
    num_experts=4,
)
```

### Production
```python
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

## Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check GPU Memory
```python
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### Profile Training
```python
import torch
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
) as prof:
    output = model(x)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Troubleshooting

**CUDA Out of Memory:**
```python
training_config.batch_size = 4
model_config.use_gradient_checkpointing = True
model_config.use_fp8 = True
```

**Slow Training:**
```python
training_config.bf16 = True
model_config.use_flash_attention = True
```

## Links

- GitHub: https://github.com/iamdarshg/better-ai
- Architecture: See ARCHITECTURE.md
- Full API: Docstrings in all modules
