# Better AI: Advanced RLHF System for Coding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Last Commit](https://img.shields.io/github/last-commit/iamdarshg/better-ai)](https://github.com/iamdarshg/better-ai/commits)
[![Language](https://img.shields.io/github/languages/top/iamdarshg/better-ai)](https://github.com/iamdarshg/better-ai)
[![Build Status](https://img.shields.io/github/actions/workflow/status/iamdarshg/better-ai/training_test.yml)](https://github.com/iamdarshg/better-ai/actions)

**Better AI** is a cutting-edge RLHF (Reinforcement Learning from Human Feedback) system designed for training advanced coding models with superior reasoning, correctness, and alignment capabilities.

## 🚀 Features

- **Cutting-Edge Training Pipeline**: A multi-stage pipeline that takes the model from pretraining to supervised fine-tuning (SFT) and finally to Reinforcement Learning from Human Feedback (RLHF) with Group Reward Policy Optimization (GRPO). This ensures a robust and well-aligned model.
- **DeepSeek-Inspired Architecture**: At its core, Better AI features a powerful and efficient transformer model inspired by the DeepSeek V3.2 architecture. This includes innovations like Ring Attention for near-infinite context processing and a Mixture of Experts (MoE) for dynamic routing and specialization.
- **Advanced Reasoning Capabilities**: To tackle complex coding challenges, Better AI is equipped with a suite of advanced reasoning features, including a Recursive Scratchpad for iterative problem-solving, Chain-of-Thought (CoT) Specialization for improved reasoning, Self-Taught Reasoner (STaR) Bootstrapping for self-improvement, and a self-correction mechanism to identify and fix errors.
- **Sophisticated Reward Modeling**: The RLHF process is guided by a Hierarchical Reward Model (HRM), a dual-reward framework that scores both single-step soundness and end-to-end coherence. This ensures that the model generates code that is not only locally correct but also globally coherent and well-structured.
- **State-of-the-Art Training Optimizations**: The training process is enhanced with a range of advanced techniques, including expert specialization tracking to encourage expert diversity, selective gradient checkpointing to reduce memory usage, dynamic expert capacity adjustment to handle varying loads, and a coherence-based scheduler to dynamically adjust the learning rate.
- **Memory and Distribution**: The system is designed for large-scale training with memory optimization techniques like FP8 quantization and support for distributed training with DDP/FSDP.

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training Pipeline](#-training-pipeline)
- [Architecture Overview](#-architecture-overview)
- [Configuration](#-configuration)
- [Dataset Configuration](#-dataset-configuration)
- [Testing](#-testing)
- [GitHub Actions CI/CD](#-github-actions-cicd)
- [Contributing](#-contributing)
- [License](#-license)

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- 20GB+ GPU memory (recommended)

### Install Dependencies

```bash
git clone https://github.com/iamdarshg/better-ai.git
cd better-ai
pip install -r requirements.txt

# Optional: Install for better performance
pip install flash-attn triton
```

## 🚀 Quick Start

### Run Full Training Pipeline

```bash
# Full pipeline with evaluation (uses mock data for testing)
python train_enhanced.py --stage full --eval --test

# Individual stages
python train_enhanced.py --stage pretrain --test
python train_enhanced.py --stage sft --test
python train_enhanced.py --stage rlhf --test
```

### Python API Example

```python
from better_ai.config import ModelConfig, TrainingConfig
from better_ai.models.enhanced_model import EnhancedDeepSeekModel
from better_ai.training.enhanced_trainer import EnhancedMoETrainer

# Setup configuration
model_config = ModelConfig()
training_config = TrainingConfig(batch_size=2, learning_rate=1e-4, max_steps=100)

# Initialize model and trainer
model = EnhancedDeepSeekModel(model_config)
trainer = EnhancedMoETrainer(model, training_config)

# Train (uses mock data by default)
trainer.train()
```

## 🔄 The Better AI Training Pipeline

The Better AI training pipeline is a carefully designed multi-stage process that progressively refines the model's capabilities. Each stage builds upon the last, resulting in a highly capable and well-aligned coding model.

### Stage 1: Pretraining
The foundation of the Better AI model is built during the pretraining stage. Here, the model is exposed to a massive corpus of code from a variety of sources, such as "The Stack v2". The primary objective of this stage is to teach the model the fundamental syntax, structures, and patterns of programming languages. This is achieved through a self-supervised learning process where the model learns to predict the next token in a sequence of code.

### Stage 2: Supervised Fine-Tuning (SFT)
Once the model has a solid understanding of code, it moves to the supervised fine-tuning (SFT) stage. In this stage, the model is trained on a curated dataset of high-quality code and natural language instructions, such as the "Magicoder" and "Code-Feedback" datasets. The goal of SFT is to teach the model to follow instructions and generate high-quality code that is not only syntactically correct but also well-structured and easy to understand.

### Stage 3: Reinforcement Learning from Human Feedback (RLHF)
The final stage of the training pipeline is Reinforcement Learning from Human Feedback (RLHF). In this stage, the model is further refined using a reward model that has been trained to predict human preferences. The model is presented with a prompt and generates multiple responses, which are then evaluated by the reward model. The model is then updated using Group Reward Policy Optimization (GRPO) to generate responses that are more likely to be preferred by humans. This process is guided by a sophisticated Hierarchical Reward Model that scores both single-step soundness and end-to-end coherence.

## 🏗️ A DeepSeek-Inspired Architecture for a Cutting-Edge System

The Better AI architecture is a modular and extensible system designed for cutting-edge research and development in RLHF and advanced model training. It is built around a powerful and efficient transformer model inspired by the DeepSeek V3.2 architecture, and it incorporates a suite of advanced features for enhanced reasoning and performance.

At its core, the Better AI model is a decoder-only transformer with a number of key architectural innovations that are inspired by the DeepSeek V3.2 architecture. These include:

- **Ring Attention**: To handle the long sequences of code that are common in software development, Better AI uses Ring Attention, a novel attention mechanism that allows for near-infinite context processing with linear complexity.
- **Mixture of Experts (MoE)**: To increase the model's capacity without a proportional increase in computational cost, Better AI uses a Mixture of Experts (MoE) architecture. This allows the model to dynamically route different parts of the input to different "expert" sub-networks, resulting in a more efficient and effective model.
- **Grouped Query Attention (GQA)**: To further improve the efficiency of the attention mechanism, Better AI uses Grouped Query Attention (GQA), which groups queries together to reduce the number of attention computations.

In addition to these core architectural features, Better AI also incorporates a number of advanced features to enhance its reasoning capabilities, including a Recursive Scratchpad for iterative reasoning, CoT Specialization for improved chain-of-thought processing, and a self-correction mechanism to identify and fix errors. For a more detailed breakdown of the architecture, please refer to the `ARCHITECTURE.md` file.

## ⚙️ Configuration

### Minimal Configuration (Testing)

```python
from better_ai.config import ModelConfig, TrainingConfig

model_config = ModelConfig(
    vocab_size=32000,
    hidden_dim=512,
    num_layers=4,
    num_experts=4,
)

training_config = TrainingConfig(
    batch_size=2,
    learning_rate=5e-4,
    max_steps=100,
    use_mock_data=True  # Use mock data for testing
)
```

### Production Configuration

```python
model_config = ModelConfig()  # Uses all defaults
training_config = TrainingConfig(
    batch_size=32,
    learning_rate=1e-4,
    max_steps=100000,
    distributed_backend="fsdp",
    use_fp8=True,
)
```

## 🧪 Testing

### Run Unit Tests

```bash
python -m pytest tests/
```

### Test Training Pipeline

```bash
# Test with mock data (fast, no GPU required)
python train_enhanced.py --stage full --test

# Test individual components
python test_model.py
python test_enhanced_model.py
```

## 🤖 GitHub Actions CI/CD

The project includes a comprehensive GitHub Actions workflow that:

1. **Runs on every push to main branch**
2. **Tests the complete training pipeline**
3. **Uses mock data for fast execution**
4. **Validates all training stages**
5. **Checks for crashes or errors**

### Workflow File: `.github/workflows/training_test.yml`

```yaml
name: Training Pipeline Test

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test-training:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest

    - name: Test training pipeline with mock data
      run: |
        python train_enhanced.py --stage full --test --batch-size 2 --max-steps 10

    - name: Run unit tests
      run: |
        python -m pytest tests/ -v

    - name: Validate checkpoints
      run: |
        ls -la checkpoints/
        test -f checkpoints/pretrained_model.pt
        test -f checkpoints/sft_model.pt
        test -f checkpoints/rlhf_model.pt
```

### Key Features of the CI/CD Pipeline

- **Multi-Python Version Testing**: Tests on Python 3.8, 3.9, 3.10
- **Fast Execution**: Uses mock data and small batch sizes
- **Memory Efficient**: Small model configuration for CI
- **Comprehensive Validation**: Tests all training stages
- **Checkpoint Verification**: Ensures models are saved correctly

## 📊 Performance Metrics

### Training Metrics

- **Perplexity**: Language modeling quality
- **Loss**: Training and validation loss
- **Accuracy**: Instruction following accuracy
- **Reward Scores**: Multi-attribute quality scores

### Evaluation Metrics

- **Coding Accuracy**: Percentage of correct solutions
- **Reasoning Quality**: Trace coherence and diversity
- **Efficiency**: Time/space complexity scores
- **Alignment**: Correlation with human preferences

## 🔧 Configuration Options

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 1 | Batch size per device |
| `learning_rate` | 1e-4 | Initial learning rate |
| `max_steps` | 100000 | Total training steps |
| `use_mock_data` | False | Use mock data for testing |
| `distributed_backend` | "ddp" | "ddp" or "fsdp" |

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 64000 | Vocabulary size |
| `hidden_dim` | 1536 | Hidden dimension |
| `num_layers` | 12 | Number of layers |
| `num_experts` | 16 | Number of MoE experts |

## 🐛 Debugging

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size`
   - Enable `use_gradient_checkpointing`
   - Use `use_fp8=True`

2. **Slow Training**
   - Enable `use_flash_attention`
   - Use mixed precision (`bf16=True`)
   - Increase `gradient_accumulation_steps`

3. **Training Instability**
   - Reduce `learning_rate`
   - Increase `warmup_steps`
   - Enable gradient clipping

### Debug Commands

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python train_enhanced.py --stage pretrain --test

# Profile memory usage
python -m memory_profiler train_enhanced.py --stage pretrain --test

# Profile time
python -m cProfile -s time train_enhanced.py --stage pretrain --test
```

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Detailed architecture documentation
- **[QUICKSTART.md](QUICKSTART.md)**: Quick start guide with examples
- **API Documentation**: Comprehensive docstrings in all modules
- **Examples**: Runnable examples in the `examples/` directory

## 📝 Dataset Configuration

The `datasets.yml` file allows for detailed configuration of each dataset, including parameters like maximum sequence length and the number of training steps.

### Example `datasets.yml`

```yaml
datasets:
  - name: "The Stack"
    path: "/path/to/the_stack"
    max_seq_length: 8192
    num_training_steps: 100000
  - name: "Magicoder"
    path: "/path/to/magicoder"
    max_seq_length: 4096
    num_training_steps: 50000
```

### Configuration Options

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | The name of the dataset. |
| `path` | string | The path to the dataset directory. |
| `max_seq_length` | int | The maximum sequence length for this dataset. |
| `num_training_steps` | int | The number of training steps to perform on this dataset. |

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Create a pull request

### Development Setup

```bash
git clone https://github.com/iamdarshg/better-ai.git
cd better-ai
pip install -e .  # Install in development mode
pip install -r requirements-dev.txt
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write comprehensive docstrings
- Include unit tests for new features

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- DeepSeek team for the original architecture
- Hugging Face for transformers library
- PyTorch team for the deep learning framework
- All contributors and users of Better AI

## 📬 Contact

For questions, issues, or contributions:
- **GitHub Issues**: https://github.com/iamdarshg/better-ai/issues
- **Discussions**: https://github.com/iamdarshg/better-ai/discussions
- **Email**: darshgupta@example.com

---

**Happy Training! 🚀**
