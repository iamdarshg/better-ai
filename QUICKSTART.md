# Quick Start Guide - Better AI RLHF System

## 5-Minute Setup

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-org/better-ai.git
cd better-ai

# Install dependencies
pip install -r requirements.txt

# Install optional optimizations
pip install flash-attn triton  # For CUDA 11.8+
```

### 2. Run Your First Training

```bash
# Full pipeline: Pretrain → SFT → RLHF
python train_enhanced.py --stage full --eval

# Or run individual stages
python train_enhanced.py --stage pretrain
python train_enhanced.py --stage sft
python train_enhanced.py --stage rlhf --eval
```

### 3. Custom Training Script

```python
from better_ai.config import ModelConfig, TrainingConfig
from better_ai.models.enhanced_model import EnhancedDeepSeekModel
from better_ai.training.enhanced_trainer import EnhancedTrainer
from better_ai.data.dataset_loaders import create_dataloader

# 1. Setup configuration
model_config = ModelConfig()
training_config = TrainingConfig(
    batch_size=8,
    learning_rate=1e-4,
    max_steps=10000,
)

# 2. Initialize trainer
trainer = EnhancedTrainer(model_config, training_config)

# 3. Load data
train_loader = create_dataloader(
    "stack_v2",
    batch_size=8,
    split="train",
)

# 4. Train
metrics = trainer.train_pretraining(train_loader, num_epochs=1)

# 5. Evaluate
trainer._save_checkpoint("my_model.pt")
```

## Common Tasks

### Evaluate a Model

```python
from better_ai.training.evaluation import RLHFEvaluator

evaluator = RLHFEvaluator(model, reward_model, device)

# Compute preference accuracy
accuracy = evaluator.compute_preference_accuracy(
    chosen_scores, rejected_scores
)
print(f"Preference Accuracy: {accuracy:.4f}")
```

### Use Reward Model

```python
from better_ai.models.reward_model import BranchRewardModel

reward_model = BranchRewardModel(config)

# Score a response
hidden_states = model(input_ids, output_hidden_states=True)['hidden_states']
score = reward_model(hidden_states[-1])
print(f"Response Score: {score:.4f}")

# Get attribute breakdown
score, attributes = reward_model(hidden_states[-1], return_branch_scores=True)
print(f"Correctness: {attributes['correctness']:.4f}")
print(f"Efficiency: {attributes['efficiency']:.4f}")
```

### Generate with Advanced Features

```python
# Enable all advanced features
config = ModelConfig(
    use_recursive_scratchpad=True,
    use_cot_specialization=True,
    use_tool_heads=True,
    use_entropic_steering=True,
)

model = EnhancedDeepSeekModel(config)

# Generate with reasoning
outputs = model(input_ids, return_advanced_features=True)
reasoning_traces = outputs['advanced_features']['scratchpad']['reasoning_traces']
print(f"Reasoning iterations: {len(reasoning_traces)}")
```

### Fine-tune on Custom Data

```python
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "input_ids": example["input_ids"],
            "labels": example["labels"],
        }

# Train on custom data
custom_dataset = CustomDataset(my_examples)
custom_loader = DataLoader(custom_dataset, batch_size=8)

trainer.train_sft(custom_loader, num_epochs=3)
```

## Configuration Presets

### Minimal (Testing)
```python
config = ModelConfig(
    vocab_size=32000,
    hidden_dim=512,
    num_layers=4,
)
```

### Standard (Default)
```python
config = ModelConfig()  # Full config
```

### Large (Production)
```python
config = ModelConfig(
    vocab_size=65536,
    hidden_dim=2048,
    num_layers=24,
    num_experts=32,
)
```

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all logs will show
trainer = EnhancedTrainer(model_config, training_config)
```

### Check GPU Memory

```python
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### Profile Training Step

```python
import torch

model = model.to(device)
x = torch.randn(1, 128, config.hidden_dim).to(device)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
) as prof:
    output = model(x)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Data Formats

### For Pretraining

```python
{
    "input_ids": torch.Tensor,      # (seq_len,)
    "attention_mask": torch.Tensor, # (seq_len,)
    "labels": torch.Tensor,         # (seq_len,)
}
```

### For SFT

```python
{
    "input_ids": torch.Tensor,
    "attention_mask": torch.Tensor,
    "labels": torch.Tensor,
}
```

### For RLHF

```python
{
    "chosen_input_ids": torch.Tensor,
    "chosen_attention_mask": torch.Tensor,
    "rejected_input_ids": torch.Tensor,
    "rejected_attention_mask": torch.Tensor,
}
```

## Performance Tips

1. **Use gradient accumulation** for larger effective batch sizes
2. **Enable FP8 quantization** for 50% memory savings
3. **Use Ring Attention** for longer sequences
4. **Profile your code** to find bottlenecks
5. **Use DDP/FSDP** for multi-GPU training

## Troubleshooting

### Issue: CUDA Out of Memory
```python
# Reduce batch size or sequence length
training_config.batch_size = 4
training_config.max_seq_length = 4096

# Or enable optimizations
model_config.use_ring_attention = True
model_config.use_gradient_checkpointing = True
```

### Issue: Slow Training
```python
# Check if GPU is being used
print(torch.cuda.is_available())  # Should be True

# Enable mixed precision
training_config.bf16 = True

# Use gradient accumulation
training_config.gradient_accumulation_steps = 2
```

### Issue: Poor Model Quality
```python
# Train longer
training_config.max_steps = 200000

# Use better learning rate
training_config.learning_rate = 2e-4

# Enable advanced features
model_config.use_recursive_scratchpad = True
model_config.use_star = True
```

## Next Steps

1. **Read Documentation**: See README_ENHANCED.md for full docs
2. **Study Architecture**: ARCHITECTURE.md explains all components
3. **Run Examples**: See `examples/` directory for runnable examples
4. **Run Tests**: `python -m pytest tests/test_rlhf_components.py`
5. **Try Evaluation**: Use the evaluation suite to benchmark your model

## Links

- **GitHub**: https://github.com/your-org/better-ai
- **Documentation**: See README_ENHANCED.md
- **Architecture**: See ARCHITECTURE.md
- **API Reference**: Docstrings in all modules
- **Issues**: GitHub Issues for bug reports

## Getting Help

1. Check the FAQ in README_ENHANCED.md
2. Search existing GitHub issues
3. Read relevant source code documentation
4. Post a new issue with details
5. Join community discussions

## Citation

If you use Better AI, please cite:

```bibtex
@software{better_ai_2024,
  title={Better AI: Advanced RLHF System for Coding},
  author={Your Organization},
  year={2024},
}
```

---

**Happy Training! 🚀**
