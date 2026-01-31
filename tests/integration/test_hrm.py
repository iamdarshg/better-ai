
import torch
import unittest
from better_ai.models.reward_model import HierarchicalRewardModel
from better_ai.config import ModelConfig
from better_ai.test_config_utils import get_small_model_config

class TestHierarchicalRewardModel(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_forward_pass(self):
        config = get_small_model_config()
        model = HierarchicalRewardModel(config).to(self.device)
        hidden_states = torch.randn(1, 10, config.hidden_dim).to(self.device)
        attention_mask = torch.ones(1, 10).to(self.device)
        reward = model(hidden_states, attention_mask)
        self.assertEqual(reward.shape, (1,))

    def test_loss_computation(self):
        config = get_small_model_config()
        model = HierarchicalRewardModel(config).to(self.device)
        chosen_rewards = torch.randn(4).to(self.device)
        rejected_rewards = torch.randn(4).to(self.device)
        loss = model.loss(chosen_rewards, rejected_rewards)
        self.assertIsInstance(loss, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
