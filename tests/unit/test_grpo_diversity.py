
import unittest
import torch
import torch.nn as nn
from better_ai.training.grpo import GRPOTrainer

class MockTokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        return " ".join([str(t.item()) for t in tokens])

class TestGRPODiversity(unittest.TestCase):
    def test_diversity_reward_integration(self):
        config = {
            "hidden_dim": 8,
            "vocab_size": 16,
            "group_size": 2,
            "device": torch.device("cpu"),
            "use_diversity_reward": True,
            "diversity_reward_weight": 1.0,
            "tokenizer": MockTokenizer()
        }

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(8, 16)
                self.tokenizer = config["tokenizer"]
            def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
                class Out:
                    def __init__(self):
                        self.logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 16)
                        self.hidden_states = [torch.randn(input_ids.shape[0], input_ids.shape[1], 8)]
                return Out()

        class MockRewardModel(nn.Module):
            def forward(self, hidden_states, attention_mask=None):
                return torch.zeros(hidden_states.shape[0])

        model = MockModel()
        reward_model = MockRewardModel()
        optimizer = torch.optim.Adam(model.parameters())
        trainer = GRPOTrainer(model, reward_model, optimizer, config)

        # Mock _generate_response_with_logprobs to return specific sequences
        def mock_gen(input_ids, attention_mask, group_idx):
            batch_size = input_ids.shape[0]
            # Group 0 and Group 1 will have different tokens
            tokens = torch.full((batch_size, input_ids.shape[1] + 2), group_idx, dtype=torch.long)
            logprobs = torch.zeros(batch_size)
            return tokens, logprobs

        trainer._generate_response_with_logprobs = mock_gen

        batch = {"input_ids": torch.zeros(2, 2, dtype=torch.long)}

        # This calls _compute_group_rewards_and_logprobs
        reward_scores, old_logprobs = trainer._compute_group_rewards_and_logprobs(batch)

        # Since MockRewardModel returns 0, the reward should be only from diversity
        # Diverse trajectories (Group 0: "0 0", Group 1: "1 1")
        # Diversity reward should be > 0
        self.assertTrue(torch.all(reward_scores > 0))

if __name__ == "__main__":
    unittest.main()
