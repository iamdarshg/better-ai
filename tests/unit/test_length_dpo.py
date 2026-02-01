"""
Unit tests for length-aware DPO loss
"""

import unittest
import torch
from better_ai.training.trainer_utils.rl import compute_length_aware_dpo_loss
from better_ai.test_config_utils import get_small_model_config


class MockModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lm_head = torch.nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, input_ids, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.config.vocab_size).to(
            input_ids.device
        )
        return {"logits": logits}


class TestLengthAwareDPOLoss(unittest.TestCase):
    def test_length_aware_dpo_loss(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = get_small_model_config()
        config.thought_token_id = 100
        config.thought_end_token_id = 101

        model = MockModel(config).to(device)
        ref_model = MockModel(config).to(device)

        # Create batch with chosen/rejected pairs
        # Chosen has thoughts (100 ... 101)
        chosen_input_ids = torch.tensor(
            [[1, 2, 100, 3, 4, 101, 5, 6, 7, 0], [1, 2, 100, 3, 4, 5, 101, 6, 0, 0]]
        ).to(device)
        rejected_input_ids = torch.tensor(
            [[1, 2, 3, 0, 0, 0, 0, 0, 0, 0], [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]]
        ).to(device)

        batch = {
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
            "prompt_len": 2,
        }

        loss = compute_length_aware_dpo_loss(model, ref_model, batch)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertNotEqual(loss.item(), 0)
        print(f"Loss: {loss.item()}")


if __name__ == "__main__":
    unittest.main()
