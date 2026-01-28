
import torch
import unittest
from better_ai.models.reward_model import HierarchicalRewardModel
from better_ai.config import ModelConfig
from better_ai.test_config_utils import get_small_model_config

class TestHierarchicalRewardModel(unittest.TestCase):
    def test_forward_pass(self):
        config = get_small_model_config()
        model = HierarchicalRewardModel(config)
        hidden_states = torch.randn(1, 10, config.hidden_dim)
        attention_mask = torch.ones(1, 10)
        reward = model(hidden_states, attention_mask)
        self.assertEqual(reward.shape, (1,))

    def test_loss_computation(self):
        config = get_small_model_config()
        model = HierarchicalRewardModel(config)
        chosen_rewards = torch.randn(4)
        rejected_rewards = torch.randn(4)
        loss = model.loss(chosen_rewards, rejected_rewards)
        self.assertIsInstance(loss, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
