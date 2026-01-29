"""
Unit tests for length-aware DPO loss
"""

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from better_ai.training.trainer_utils.rl import compute_length_aware_dpo_loss
from better_ai.config import ModelConfig

class MockModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lm_head = torch.nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, input_ids, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.config.vocab_size)
        return {"logits": logits}

def test_length_aware_dpo_loss():
    config = ModelConfig()
    config.thought_token_id = 100
    config.thought_end_token_id = 101

    model = MockModel(config)
    ref_model = MockModel(config)

    # Create batch with chosen/rejected pairs
    # Chosen has thoughts (100 ... 101)
    chosen_input_ids = torch.tensor([
        [1, 2, 100, 3, 4, 101, 5, 6, 7, 0],
        [1, 2, 100, 3, 4, 5, 101, 6, 0, 0]
    ])
    rejected_input_ids = torch.tensor([
        [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
    ])

    batch = {
        'chosen_input_ids': chosen_input_ids,
        'rejected_input_ids': rejected_input_ids,
        'prompt_len': 2
    }

    loss = compute_length_aware_dpo_loss(model, ref_model, batch)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() != 0
    print(f"Loss: {loss.item()}")

if __name__ == "__main__":
    pytest.main([__file__])
