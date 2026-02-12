"""
Unit tests for ARPO (Agentic Reinforced Policy Optimization)
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
from better_ai.test_resource_tags import high_resource
# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from better_ai.training.arpo import (
    EntropyMonitor,
    AdaptiveRolloutManager,
    StepLevelAdvantageAttributor,
    ARPOTrainer,
)

@high_resource
class TestEntropyMonitor(unittest.TestCase):
    """Test entropy monitoring functionality"""

    def test_entropy_monitor_initialization(self):
        monitor = EntropyMonitor(window_size=5, threshold_multiplier=1.5)
        self.assertEqual(monitor.window_size, 5)
        self.assertEqual(monitor.threshold_multiplier, 1.5)
        self.assertEqual(monitor.entropy_history, [])
        self.assertIsNone(monitor.baseline_entropy)

    def test_token_entropy_computation(self):
        monitor = EntropyMonitor()
        # Create test logits
        logits = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])

        entropy = monitor.compute_token_entropy(logits)
        self.assertEqual(entropy.shape, (2,))
        self.assertGreater(entropy[0], entropy[1])  # First is more uncertain

    def test_entropy_spike_detection(self):
        monitor = EntropyMonitor(window_size=3, threshold_multiplier=1.0)

        # Add baseline entropy values
        baseline_logits = torch.tensor([[0.5, 0.3, 0.2]])
        for _ in range(3):
            monitor.update(baseline_logits)

        # Add spike
        spike_logits = torch.tensor([[0.33, 0.33, 0.34]])  # High entropy
        analysis = monitor.update(spike_logits)

        self.assertTrue(analysis["is_spike"])
        self.assertGreater(analysis["current_entropy"], analysis["baseline_entropy"])

@high_resource
class TestAdaptiveRolloutManager(unittest.TestCase):
    """Test adaptive rollout management"""

    def test_initialization(self):
        manager = AdaptiveRolloutManager(base_branch_factor=2, max_branch_factor=6)
        self.assertEqual(manager.base_branch_factor, 2)
        self.assertEqual(manager.max_branch_factor, 6)
        self.assertEqual(manager.current_branch_factor, 2)

    def test_branch_factor_adjustment(self):
        manager = AdaptiveRolloutManager(base_branch_factor=1, max_branch_factor=3)

        # No spike - should remain at baseline
        normal_analysis = {"is_spike": False}
        factor = manager.get_branch_factor(normal_analysis)
        self.assertEqual(factor, 1)

        # Spike detected - should increase
        spike_analysis = {"is_spike": True}
        factor = manager.get_branch_factor(spike_analysis)
        self.assertEqual(factor, 2)

        # Another spike - should increase again
        factor = manager.get_branch_factor(spike_analysis)
        self.assertEqual(factor, 3)

        # No more spikes - should gradually decrease
        normal_analysis2 = {"is_spike": False}
        factor = manager.get_branch_factor(normal_analysis2)
        self.assertEqual(factor, 2)

@high_resource
class TestStepLevelAdvantageAttributor(unittest.TestCase):
    """Test step-level advantage attribution"""

    def test_tool_states_value_estimation(self):
        attributor = StepLevelAdvantageAttributor()

        # Test successful tool use
        success_state = {
            "tool_success": True,
            "has_error": False,
            "progress_score": 0.5,
        }
        value = attributor._estimate_state_value(success_state)
        self.assertEqual(value, 1.3)  # 0.5 + 0.3 + 0.5

        # Test failed tool use
        failed_state = {"tool_success": False, "has_error": True, "progress_score": 0.2}
        value = attributor._estimate_state_value(failed_state)
        self.assertEqual(value, 0.2)  # Only progress score

@high_resource
class TestARPOTrainer(unittest.TestCase):
    """Test ARPO trainer integration"""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mock_model = nn.Linear(10, 100).to(self.device)

        class MockRewardModel:
            def score(self, prompt, response):
                return 0.5  # Mock score

        self.mock_reward_model = MockRewardModel()
        self.mock_config = {
            "entropy_window": 5,
            "entropy_threshold": 2.0,
            "base_branch_factor": 1,
            "max_branch_factor": 3,
            "enable_adaptive_rollouts": True,
            "device": self.device,
        }
        self.trainer = ARPOTrainer(self.mock_model, self.mock_reward_model, None, self.mock_config)

    def test_trainer_initialization(self):
        self.assertEqual(self.trainer.entropy_monitor.window_size, 5)
        self.assertEqual(self.trainer.rollout_manager.base_branch_factor, 1)
        self.assertEqual(self.trainer.rollout_manager.max_branch_factor, 3)
        self.assertEqual(self.trainer.config["enable_adaptive_rollouts"], True)

    def test_entropy_analysis_during_generation(self):
        # Mock generation outputs
        mock_outputs = {"scores": [torch.randn(1, 100).to(self.device) for _ in range(10)]}

        analysis = self.trainer._analyze_generation_entropy(mock_outputs)

        self.assertIn("current_entropy", analysis)
        self.assertIn("is_spike", analysis)
        self.assertIn("baseline_entropy", analysis)

    def test_adaptive_branch_factor_determination(self):
        # Test normal conditions
        normal_analysis = {"is_spike": False}
        self.trainer.rollout_manager.current_branch_factor = 2
        factor = self.trainer.rollout_manager.get_branch_factor(normal_analysis)
        self.assertEqual(factor, 1)  # Should return to baseline

        # Test spike conditions
        spike_analysis = {"is_spike": True}
        factor = self.trainer.rollout_manager.get_branch_factor(spike_analysis)
        self.assertEqual(factor, 2)  # Should increase from baseline


if __name__ == "__main__":
    unittest.main()
