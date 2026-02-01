#!/usr/bin/env python3
"""
Integration test for low resource workflow using unittest and test_config_utils
"""

import unittest
import torch
import os
import psutil
import time

from better_ai.test_config_utils import (
    get_small_model_config,
    get_small_training_config,
)
from better_ai.models.enhanced_model import EnhancedDeepSeekModel
from better_ai.training.enhanced_trainer import EnhancedMoETrainer


class TestLowResourceIntegration(unittest.TestCase):
    """Test low resource workflow integration"""

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # in MB

    def test_low_resource_workflow(self):
        """Test that low resource workflow stays within memory limits"""
        print(f"Starting memory: {self.get_memory_usage():.2f} MB")

        # Use small config from test_config_utils
        model_config = get_small_model_config()
        # Override some specific settings for low resource test
        model_config.use_ring_attention = (
            False  # Ring attention might need distributed setup
        )
        model_config.use_recursive_scratchpad = True
        model_config.use_tidar = True
        model_config.tidar_num_steps = 2
        model_config.tidar_diffusion_dim = 32

        training_config = get_small_training_config()
        # Override specific settings for low resource test
        training_config.max_steps = 5
        training_config.use_mock_data = True

        device = torch.device("cpu")

        print("Initializing model...")
        model = EnhancedDeepSeekModel(model_config, device=device)
        print(f"Memory after model init: {self.get_memory_usage():.2f} MB")

        # Create mock dataloaders
        class MockDataset(torch.utils.data.Dataset):
            def __init__(self, vocab_size, seq_len, size=10):
                self.size = size
                self.vocab_size = vocab_size
                self.seq_len = seq_len

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
                    "labels": torch.randint(0, self.vocab_size, (self.seq_len,)),
                }

        train_ds = MockDataset(model_config.vocab_size, model_config.max_seq_length)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=training_config.learning_rate
        )

        print("Initializing trainer...")
        trainer = EnhancedMoETrainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=train_loader,
            optimizer=optimizer,
            scheduler=None,
            config=training_config,
            device=device,
            use_enhanced_features=True,
        )
        print(f"Memory after trainer init: {self.get_memory_usage():.2f} MB")

        print("Starting training...")
        trainer.train()
        print(f"Memory after training: {self.get_memory_usage():.2f} MB")

        # Inference test
        print("Starting inference...")
        input_ids = torch.randint(0, model_config.vocab_size, (1, 8))
        generated = model.generate(input_ids, max_new_tokens=5)
        print(f"Generated shape: {generated.shape}")
        print(f"Final memory: {self.get_memory_usage():.2f} MB")

        final_mem = self.get_memory_usage()
        self.assertLess(
            final_mem, 2048, f"Memory usage {final_mem} MB exceeds 2GB limit"
        )
        print("Low-resource workflow test passed!")


if __name__ == "__main__":
    unittest.main()
