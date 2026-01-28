
import torch
import os
import psutil
import time
from better_ai.config import ModelConfig, TrainingConfig
from better_ai.models.enhanced_model import EnhancedDeepSeekModel
from better_ai.training.enhanced_trainer import EnhancedMoETrainer

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # in MB

def test_low_resource_workflow():
    print(f"Starting memory: {get_memory_usage():.2f} MB")

    # Tiny model configuration
    model_config = ModelConfig(
        vocab_size=256,
        hidden_dim=32,
        num_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_dim=64,
        max_seq_length=32,
        use_ring_attention=False, # Ring attention might need distributed setup
        use_recursive_scratchpad=True,
        use_tidar=True,
        tidar_num_steps=2,
        tidar_diffusion_dim=32
    )

    # Tiny training configuration
    training_config = TrainingConfig(
        batch_size=1,
        max_steps=5,
        learning_rate=1e-4,
        bf16=False, # Use float32 for CPU
        use_mock_data=True
    )

    device = torch.device("cpu")

    print("Initializing model...")
    model = EnhancedDeepSeekModel(model_config, device=device)
    print(f"Memory after model init: {get_memory_usage():.2f} MB")

    # Create mock dataloaders
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

    train_ds = MockDataset(model_config.vocab_size, model_config.max_seq_length)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)

    print("Initializing trainer...")
    trainer = EnhancedMoETrainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=train_loader,
        optimizer=optimizer,
        scheduler=None,
        config=training_config,
        device=device,
        use_enhanced_features=True
    )
    print(f"Memory after trainer init: {get_memory_usage():.2f} MB")

    print("Starting training...")
    trainer.train()
    print(f"Memory after training: {get_memory_usage():.2f} MB")

    # Inference test
    print("Starting inference...")
    input_ids = torch.randint(0, model_config.vocab_size, (1, 8))
    generated = model.generate(input_ids, max_new_tokens=5)
    print(f"Generated shape: {generated.shape}")
    print(f"Final memory: {get_memory_usage():.2f} MB")

    final_mem = get_memory_usage()
    assert final_mem < 2048, f"Memory usage {final_mem} MB exceeds 2GB limit"
    print("Low-resource workflow test passed!")

if __name__ == "__main__":
    test_low_resource_workflow()
