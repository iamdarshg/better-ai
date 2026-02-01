#!/usr/bin/env python3
"""
Unit test for RL Stage 2 forward pass using unittest and test_config_utils
"""

import unittest
import torch
from better_ai.test_config_utils import (
    get_small_model_config,
    get_small_training_config,
)
from better_ai.models.enhanced_model import EnhancedDeepSeekModel
from better_ai.training.enhanced_trainer import EnhancedMoETrainer


class TestRLStage2(unittest.TestCase):
    """Test RL Stage 2 forward pass"""

    def test_rl_stage2_forward(self):
        # Use small configs from test_config_utils
        config = get_small_model_config()
        # Override specific settings for RL stage 2 test
        config.vocab_size = 100
        config.hidden_dim = 128
        config.num_layers = 1
        config.num_attention_heads = 4
        config.num_key_value_heads = 2

        train_config = get_small_training_config()
        train_config.rl_stage = 2

        model = EnhancedDeepSeekModel(config)

        trainer = EnhancedMoETrainer(
            model=model,
            train_dataloader=None,
            eval_dataloader=None,
            optimizer=None,
            scheduler=None,
            config=train_config,
            device=torch.device("cpu"),
            use_enhanced_features=False,
        )

        batch = {
            "prompt": "Test prompt",
            "response": "Test response",
            "input_ids": torch.randint(0, 100, (2, 8)),
            "attention_mask": torch.ones((2, 8)),
        }

        loss, aux_loss, expert_ids = trainer._enhanced_forward_pass(batch)
        self.assertNotEqual(loss.item(), 0)


if __name__ == "__main__":
    unittest.main()
