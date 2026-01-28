"""
Unit tests for ARPO (Agentic Reinforced Policy Optimization)
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from better_ai.training.arpo import (
    EntropyMonitor,
    AdaptiveRolloutManager,
    StepLevelAdvantageAttributor,
    ARPOTrainer,
)


class TestEntropyMonitor:
    """Test entropy monitoring functionality"""

    def test_entropy_monitor_initialization(self):
        monitor = EntropyMonitor(window_size=5, threshold_multiplier=1.5)
        assert monitor.window_size == 5
        assert monitor.threshold_multiplier == 1.5
        assert monitor.entropy_history == []
        assert monitor.baseline_entropy is None

    def test_token_entropy_computation(self):
        monitor = EntropyMonitor()
        # Create test logits
        logits = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])

        entropy = monitor.compute_token_entropy(logits)
        assert entropy.shape == (2,)
        assert entropy[0] > entropy[1]  # First is more uncertain

    def test_entropy_spike_detection(self):
        monitor = EntropyMonitor(window_size=3, threshold_multiplier=1.0)

        # Add baseline entropy values
        baseline_logits = torch.tensor([[0.5, 0.3, 0.2]])
        for _ in range(3):
            monitor.update(baseline_logits)

        # Add spike
        spike_logits = torch.tensor([[0.33, 0.33, 0.34]])  # High entropy
        analysis = monitor.update(spike_logits)

        assert analysis["is_spike"] == True
        assert analysis["current_entropy"] > analysis["baseline_entropy"]


class TestAdaptiveRolloutManager:
    """Test adaptive rollout management"""

    def test_initialization(self):
        manager = AdaptiveRolloutManager(base_branch_factor=2, max_branch_factor=6)
        assert manager.base_branch_factor == 2
        assert manager.max_branch_factor == 6
        assert manager.current_branch_factor == 2

    def test_branch_factor_adjustment(self):
        manager = AdaptiveRolloutManager(base_branch_factor=1, max_branch_factor=3)

        # No spike - should remain at baseline
        normal_analysis = {"is_spike": False}
        factor = manager.get_branch_factor(normal_analysis)
        assert factor == 1

        # Spike detected - should increase
        spike_analysis = {"is_spike": True}
        factor = manager.get_branch_factor(spike_analysis)
        assert factor == 2

        # Another spike - should increase again
        factor = manager.get_branch_factor(spike_analysis)
        assert factor == 3

        # No more spikes - should gradually decrease
        normal_analysis2 = {"is_spike": False}
        factor = manager.get_branch_factor(normal_analysis2)
        assert factor == 2


class TestStepLevelAdvantageAttributor:
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
        assert value == 0.8  # 0.5 + 0.3

        # Test failed tool use
        failed_state = {"tool_success": False, "has_error": True, "progress_score": 0.2}
        value = attributor._estimate_state_value(failed_state)
        assert value == 0.2  # Only progress score


class TestARPOTrainer:
    """Test ARPO trainer integration"""

    @pytest.fixture
    def mock_model(self):
        return nn.Linear(10, 100)  # Simple model for testing

    @pytest.fixture
    def mock_reward_model(self):
        class MockRewardModel:
            def score(self, prompt, response):
                return 0.5  # Mock score

        return MockRewardModel()

    @pytest.fixture
    def mock_config(self):
        return {
            "entropy_window": 5,
            "entropy_threshold": 2.0,
            "base_branch_factor": 1,
            "max_branch_factor": 3,
            "enable_adaptive_rollouts": True,
            "device": torch.device("cpu"),
        }

    def test_trainer_initialization(self, mock_model, mock_reward_model, mock_config):
        trainer = ARPOTrainer(mock_model, mock_reward_model, None, mock_config)

        assert trainer.entropy_monitor.window_size == 5
        assert trainer.rollout_manager.base_branch_factor == 1
        assert trainer.rollout_manager.max_branch_factor == 3
        assert trainer.config["enable_adaptive_rollouts"] == True

    def test_entropy_analysis_during_generation(
        self, mock_model, mock_reward_model, mock_config
    ):
        trainer = ARPOTrainer(mock_model, mock_reward_model, None, mock_config)

        # Mock generation outputs
        mock_outputs = {"scores": [torch.randn(1, 100) for _ in range(10)]}

        analysis = trainer._analyze_generation_entropy(mock_outputs)

        assert "current_entropy" in analysis
        assert "is_spike" in analysis
        assert "baseline_entropy" in analysis

    def test_adaptive_branch_factor_determination(
        self, mock_model, mock_reward_model, mock_config
    ):
        trainer = ARPOTrainer(mock_model, mock_reward_model, None, mock_config)

        # Test normal conditions
        normal_analysis = {"is_spike": False}
        trainer.rollout_manager.current_branch_factor = 2
        factor = trainer.rollout_manager.get_branch_factor(normal_analysis)
        assert factor == 1  # Should return to baseline

        # Test spike conditions
        spike_analysis = {"is_spike": True}
        factor = trainer.rollout_manager.get_branch_factor(spike_analysis)
        assert factor == 2  # Should increase from baseline


if __name__ == "__main__":
    pytest.main([__file__])
