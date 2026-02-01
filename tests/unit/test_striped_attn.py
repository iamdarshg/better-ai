#!/usr/bin/env python3
"""
Unit test for striped attention using unittest and test_config_utils
"""

import unittest
import torch
from better_ai.test_config_utils import get_small_model_config
from better_ai.models.enhanced_model import EnhancedDeepSeekModel
from better_ai.test_resource_tags import high_resource, low_resource
@low_resource
class TestStripedAttention(unittest.TestCase):
    """Test striped attention initialization and forward pass"""

    def test_striped_attention_init(self):
        # Use minimal config for speed - override test_config_utils defaults
        config = get_small_model_config()
        # Override with ultra-small settings for fast testing
        config.use_ring_attention = True
        config.use_striped_attention = True
        config.hidden_dim = 32  # Much smaller
        config.num_layers = 1  # Single layer
        config.vocab_size = 50  # Smaller vocab
        config.num_attention_heads = 8  # 32/8=4, good division
        config.num_key_value_heads = 2
        config.intermediate_dim = 64  # 2x hidden_dim
        config.max_seq_length = 16  # Very short sequences

        # DKept expensive features to test striped attention as a whole
        config.use_flash_attention = True
        config.use_gradient_checkpointing = True
        config.use_sparse_attention = True

        model = EnhancedDeepSeekModel(config)

        # Check if attention is StripedAttention
        from better_ai.models.ring_attention import StripedAttention

        self.assertIsInstance(model.layers[0].self_attn, StripedAttention)

        # Minimal test data
        batch_size = 2  # Single batch
        seq_len = 4  # Very short sequence
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Fast forward pass with minimal computation
        with torch.no_grad():  # Disable gradients for speed
            outputs = model(input_ids)

        self.assertEqual(
            outputs["logits"].shape, (batch_size, seq_len, config.vocab_size)
        )


if __name__ == "__main__":
    unittest.main()
