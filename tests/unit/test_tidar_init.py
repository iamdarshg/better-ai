#!/usr/bin/env python3
"""
Unit test for TiDAR initialization using unittest and test_config_utils
"""

import unittest
import torch
from better_ai.test_config_utils import get_small_model_config
from better_ai.models.enhanced_model import EnhancedDeepSeekModel


class TestTiDARInitialization(unittest.TestCase):
    """Test TiDAR initialization and forward pass"""

    def test_tidar_initialization(self):
        # Use small config from test_config_utils
        config = get_small_model_config()
        # Override specific settings for TiDAR test
        config.use_tidar = True
        config.tidar_num_steps = 3
        config.tidar_diffusion_dim = 64
        config.hidden_dim = 128
        config.vocab_size = 100
        config.num_layers = 1
        config.num_attention_heads = 8  # Ensure divisible by hidden_dim (128/8=16)
        config.num_key_value_heads = 4

        model = EnhancedDeepSeekModel(config)

        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        outputs = model(input_ids, return_advanced_features=True)

        self.assertIn("tidar", outputs["advanced_features"])
        tidar_out = outputs["advanced_features"]["tidar"]
        self.assertIn("refined_scratchpad", tidar_out)
        self.assertEqual(
            tidar_out["refined_scratchpad"].shape,
            (batch_size, seq_len, config.hidden_dim),
        )


if __name__ == "__main__":
    unittest.main()
