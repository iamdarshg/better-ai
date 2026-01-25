"""
Unit tests for new features in the Better AI model
"""

import unittest
import torch
from better_ai.data.unified_dataloader import parse_xml_tags

class TestNewFeatures(unittest.TestCase):

    def test_parse_xml_tags(self):
        """Test the parsing of XML-style tags"""

        # Test case 1: Simple case with tags
        text_with_tags = "<problem>This is the problem</problem>"
        expected_text = "[PROBLEM]This is the problem[/PROBLEM]"
        self.assertEqual(parse_xml_tags(text_with_tags), expected_text)

        # Test case 2: Multiple tags
        text_with_tags = "<problem>This is the problem</problem><constraints>These are the constraints</constraints>"
        expected_text = "[PROBLEM]This is the problem[/PROBLEM][CONSTRAINTS]These are the constraints[/CONSTRAINTS]"
        self.assertEqual(parse_xml_tags(text_with_tags), expected_text)

        # Test case 3: No tags
        text_without_tags = "This is a plain text"
        self.assertEqual(parse_xml_tags(text_without_tags), text_without_tags)

        # Test case 4: Empty string
        self.assertEqual(parse_xml_tags(""), "")

    def test_branch_reward_model(self):
        """Test the BranchRewardModel with multi-turn analysis"""

        from better_ai.models.reward_model import BranchRewardModel
        from better_ai.config import ModelConfig

        config = ModelConfig()
        model = BranchRewardModel(config, num_turns=3)

        # Create a dummy hidden state
        hidden_states = torch.randn(1, 10, config.hidden_dim)

        # Test the forward pass
        reward, branch_scores = model.forward(hidden_states, return_branch_scores=True)
        self.assertEqual(reward.shape, (1,))
        self.assertIn("correctness", branch_scores)
        self.assertIn("efficiency", branch_scores)
        self.assertIn("readability", branch_scores)
        self.assertIn("robustness", branch_scores)

    def test_expert_router(self):
        """Test the ExpertRouter with a pre-router network"""

        from better_ai.models.moe import ExpertRouter
        from better_ai.config import ModelConfig

        config = ModelConfig()
        router = ExpertRouter(config.hidden_dim, num_experts=8, pre_router_dim=64)

        # Create a dummy hidden state
        hidden_states = torch.randn(1, 10, config.hidden_dim)

        # Test the forward pass
        routing_weights, selected_experts, router_logits = router.forward(hidden_states)
        self.assertEqual(routing_weights.shape, (1, 10, 2))
        self.assertEqual(selected_experts.shape, (1, 10, 2))
        self.assertEqual(router_logits.shape, (1, 10, 8))
        self.assertTrue(torch.all(routing_weights > 0))
        self.assertTrue(torch.all(routing_weights <= 1))

    def test_trajectory_reward(self):
        """Test the trajectory reward computation"""

        from better_ai.models.enhanced_model import EnhancedDeepSeekModel
        from better_ai.config import ModelConfig

        config = ModelConfig()
        model = EnhancedDeepSeekModel(config)

        # Create dummy data with batch size 2
        final_answer = torch.tensor([[1, 2, 3], [1, 2, 3]])
        ground_truth_answer = torch.tensor([[1, 2, 3], [1, 2, 4]])

        # Test the reward computation
        reward = model.compute_trajectory_reward(final_answer, ground_truth_answer)
        self.assertTrue(torch.equal(reward, torch.tensor([1.0, 0.0])))

if __name__ == '__main__':
    unittest.main()
