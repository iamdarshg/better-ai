
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from better_ai.config import ModelConfig, TrainingConfig
from better_ai.models.enhanced_model import EnhancedDeepSeekModel
from better_ai.training.enhanced_trainer import EnhancedMoETrainer

def run_benchmark(hidden_dim=None, vocab_size=None):
    """Run a tiny benchmark to get loss values"""
    config = ModelConfig(
        vocab_size=vocab_size or 128,
        hidden_dim=hidden_dim or 64,
        num_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        max_seq_length=16
    )

    train_config = TrainingConfig(
        batch_size=1,
        max_steps=10,
        learning_rate=1e-3,
        use_mock_data=True
    )

    device = torch.device("cpu")
    model = EnhancedDeepSeekModel(config, device=device)

    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, vocab_size, seq_len, size=10):
            self.size = size
            self.vocab_size = vocab_size
            self.seq_len = seq_len
        def __len__(self): return self.size
        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
                "labels": torch.randint(0, self.vocab_size, (self.seq_len,))
            }

    train_ds = MockDataset(config.vocab_size, config.max_seq_length)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)

    trainer = EnhancedMoETrainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=train_loader,
        optimizer=optimizer,
        scheduler=None,
        config=train_config,
        device=device,
        use_enhanced_features=False
    )

    results = trainer.train()
    final_loss = results['final_metrics']['loss']
    perplexity = np.exp(final_loss)

    return final_loss, perplexity

def plot_scaling():
    print("Gathering data for Scaling Plots...")

    # 1. Hidden Dimension Scaling
    hidden_dims = [32, 64, 128, 256]
    losses_dim = []
    ppls_dim = []

    for h_dim in hidden_dims:
        print(f"Benchmarking hidden_dim={h_dim}...")
        loss, ppl = run_benchmark(hidden_dim=h_dim)
        losses_dim.append(loss)
        ppls_dim.append(ppl)

    # 2. Vocab Size Scaling
    vocab_sizes = [128, 256, 512, 1024]
    losses_vocab = []
    ppls_vocab = []

    for v_size in vocab_sizes:
        print(f"Benchmarking vocab_size={v_size}...")
        loss, ppl = run_benchmark(vocab_size=v_size)
        losses_vocab.append(loss)
        ppls_vocab.append(ppl)

    # Plotting
    os.makedirs("better_ai/analysis_scripts/plots", exist_ok=True)

    # Plot Hidden Dim Scaling
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hidden_dims, losses_dim, marker='o')
    plt.title('Loss vs Internal Dimension')
    plt.xlabel('Hidden Dimension')
    plt.ylabel('Eval Loss')

    plt.subplot(1, 2, 2)
    plt.plot(hidden_dims, ppls_dim, marker='o', color='orange')
    plt.title('Perplexity vs Internal Dimension')
    plt.xlabel('Hidden Dimension')
    plt.ylabel('Perplexity')
    plt.tight_layout()
    plt.savefig("better_ai/analysis_scripts/plots/hidden_dim_scaling.png")

    # Plot Vocab Size Scaling
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(vocab_sizes, losses_vocab, marker='o')
    plt.title('Loss vs Vocab Size')
    plt.xlabel('Vocab Size')
    plt.ylabel('Eval Loss')

    plt.subplot(1, 2, 2)
    plt.plot(vocab_sizes, ppls_vocab, marker='o', color='orange')
    plt.title('Perplexity vs Vocab Size')
    plt.xlabel('Vocab Size')
    plt.ylabel('Perplexity')
    plt.tight_layout()
    plt.savefig("better_ai/analysis_scripts/plots/vocab_size_scaling.png")

    print("Plots saved in better_ai/analysis_scripts/plots/")

if __name__ == "__main__":
    plot_scaling()
