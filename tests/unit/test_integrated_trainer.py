"""
Unit tests for Integrated Advanced Trainer
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from better_ai.training.integrated_trainer import (
    IntegratedAdvancedTrainer,
    create_integrated_trainer,
)
from better_ai.test_resource_tags import low_resource

class MockModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.linear = nn.Linear(10, 100)
        class Config:
            def __init__(self, hd):
                self.hidden_dim = hd
                self.vocab_size = 100
        self.config = Config(hidden_dim)

    def generate(self, **kwargs):
        return torch.randint(0, 100, (1, 20))

    def generate_group(self, **kwargs):
        group_size = kwargs.get("group_size", 4)
        return torch.randint(0, 100, (group_size, 20))

    def forward(self, input_ids=None, **kwargs):
        batch_size = input_ids.size(0) if input_ids is not None else 1
        return {
            "logits": torch.randn(batch_size, 5, 100).to(input_ids.device),
            "last_hidden_state": torch.randn(batch_size, 5, self.config.hidden_dim).to(input_ids.device),
            "hidden_states": [torch.randn(batch_size, 5, self.config.hidden_dim).to(input_ids.device)],
            "advanced_features": {
                "reward": torch.randn(batch_size).to(input_ids.device)
            }
        }


@low_resource
class TestIntegratedAdvancedTrainer(unittest.TestCase):
    """Test integrated trainer with all optimizations"""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mock_model = MockModel(hidden_dim=64).to(self.device)

        class MockRewardModel(nn.Module):
            def score(self, prompt, response):
                return 0.5
            def forward(self, hidden_states, mask=None):
                return torch.randn(hidden_states.size(0)).to(hidden_states.device)

        self.mock_reward_model = MockRewardModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.mock_model.parameters(), lr=1e-4)
        self.full_config = {
            "enable_arpo": True,
            "enable_cleaner": True,
            "enable_kv_cache": True,
            "entropy_window": 5,
            "entropy_threshold": 2.0,
            "base_branch_factor": 1,
            "max_branch_factor": 3,
            "max_cache_size": 10,
            "cleaner_similarity_threshold": 0.5,
            "enable_purification": True,
            "device": self.device,
            "hidden_dim": 64,
        }

    def test_initialization_with_all_features(self):
        trainer = IntegratedAdvancedTrainer(
            self.mock_model, self.mock_reward_model, self.optimizer, self.full_config
        )

        self.assertTrue(trainer.config["enable_arpo"])
        self.assertTrue(trainer.config["enable_cleaner"])
        self.assertTrue(trainer.config["enable_kv_cache"])
        self.assertIsNotNone(trainer.arpo_trainer)
        self.assertIsNotNone(trainer.cleaner_collector)
        self.assertIsNotNone(trainer.kv_optimized_trainer)
        self.assertIsNone(trainer.grpo_trainer)  # Should not be used

    def test_initialization_partial_features(self):
        config = {
            "enable_arpo": False,
            "enable_cleaner": True,
            "enable_kv_cache": False,
            "device": self.device,
            "hidden_dim": 64,
        }

        trainer = IntegratedAdvancedTrainer(
            self.mock_model, self.mock_reward_model, self.optimizer, config
        )

        self.assertIsNone(trainer.arpo_trainer)
        self.assertIsNotNone(trainer.cleaner_collector)
        self.assertIsNone(trainer.kv_optimized_trainer)
        self.assertIsNotNone(trainer.grpo_trainer)  # Should fallback to GRPO

    def test_training_stats_tracking(self):
        trainer = IntegratedAdvancedTrainer(
            self.mock_model, self.mock_reward_model, self.optimizer, self.full_config
        )

        initial_stats = trainer.training_stats
        self.assertEqual(initial_stats["total_steps"], 0)
        self.assertEqual(initial_stats["arpo_improvements"], 0)
        self.assertEqual(initial_stats["cleaner_corrections"], 0)
        self.assertEqual(initial_stats["kv_cache_saves"], 0)

    def test_train_step_with_all_optimizations(self):
        trainer = IntegratedAdvancedTrainer(
            self.mock_model, self.mock_reward_model, self.optimizer, self.full_config
        )

        batch = {
            "input_ids": torch.randint(0, 100, (2, 5)).to(self.device),
            "attention_mask": torch.ones(2, 5).to(self.device),
            "target_ids": torch.randint(0, 100, (2, 5)).to(self.device),
        }

        metrics = trainer.train_step(batch)

        self.assertTrue("loss" in metrics or "total_loss" in metrics or "loss_with_cache" in metrics)
        self.assertEqual(trainer.training_stats["total_steps"], 1)

        # Check that component statistics are updated
        stats = trainer.get_comprehensive_statistics()
        self.assertEqual(stats["total_steps"], 1)

    def test_comprehensive_statistics(self):
        trainer = IntegratedAdvancedTrainer(
            self.mock_model, self.mock_reward_model, self.optimizer, self.full_config
        )

        stats = trainer.get_comprehensive_statistics()

        self.assertIn("total_steps", stats)
        self.assertIn("arpo_config", stats)
        self.assertIn("cleaner_stats", stats)
        self.assertIn("kv_cache_stats", stats)
        self.assertIn("overall_efficiency", stats)

        # Check efficiency calculations
        efficiency = stats["overall_efficiency"]
        self.assertIn("arpo_impact_rate", efficiency)
        self.assertIn("cleaner_correction_rate", efficiency)
        self.assertIn("kv_cache_saving_rate", efficiency)


@low_resource
class TestIntegratedTrainerFactory(unittest.TestCase):
    """Test factory function for integrated trainer"""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mock_model = MockModel(hidden_dim=100).to(self.device)

        class MockRewardModel(nn.Module):
            def score(self, prompt, response):
                return 0.5
            def forward(self, hidden_states, mask=None):
                return torch.randn(hidden_states.size(0)).to(hidden_states.device)

        self.mock_reward_model = MockRewardModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.mock_model.parameters(), lr=1e-4)

    def test_create_integrated_trainer_with_defaults(self):
        # Test with minimal config (should use defaults)
        user_config = {"device": self.device, "hidden_dim": 100}

        trainer = create_integrated_trainer(
            self.mock_model, self.mock_reward_model, self.optimizer, user_config
        )

        self.assertTrue(trainer.config["enable_arpo"])  # Default
        self.assertTrue(trainer.config["enable_cleaner"])  # Default
        self.assertTrue(trainer.config["enable_kv_cache"])  # Default
        self.assertEqual(trainer.config["entropy_window"], 10)  # Default
        self.assertEqual(trainer.config["max_cache_size"], 1000)  # Default

    def test_create_integrated_trainer_with_user_config(self):
        user_config = {
            "enable_arpo": False,
            "enable_cleaner": False,
            "enable_kv_cache": False,
            "entropy_window": 20,
            "custom_setting": "test",
            "hidden_dim": 100,
            "device": self.device
        }

        trainer = create_integrated_trainer(
            self.mock_model, self.mock_reward_model, self.optimizer, user_config
        )

        self.assertFalse(trainer.config["enable_arpo"])  # User override
        self.assertFalse(trainer.config["enable_cleaner"])  # User override
        self.assertFalse(trainer.config["enable_kv_cache"])  # User override
        self.assertEqual(trainer.config["entropy_window"], 20)  # User override
        self.assertEqual(trainer.config["custom_setting"], "test")  # User setting preserved


@low_resource
class TestFeatureIntegration(unittest.TestCase):
    """Test interaction between different features"""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mock_model = MockModel(hidden_dim=100).to(self.device)

        class MockRewardModel(nn.Module):
            def score(self, prompt, response):
                return 0.5
            def forward(self, hidden_states, mask=None):
                return torch.randn(hidden_states.size(0)).to(hidden_states.device)

        self.mock_reward_model = MockRewardModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.mock_model.parameters(), lr=1e-4)

    def test_feature_coordination(self):
        config = {
            "enable_arpo": True,
            "enable_cleaner": True,
            "enable_kv_cache": True,
            "device": self.device,
            "hidden_dim": 100,
        }

        trainer = IntegratedAdvancedTrainer(
            self.mock_model, self.mock_reward_model, self.optimizer, config
        )

        # Simulate a few training steps
        batch = {
            "input_ids": torch.randint(0, 100, (1, 5)).to(self.device),
            "attention_mask": torch.ones(1, 5).to(self.device),
            "target_ids": torch.randint(0, 100, (1, 5)).to(self.device),
        }

        for _ in range(3):
            metrics = trainer.train_step(batch)

        # Check that all features are working together
        stats = trainer.get_comprehensive_statistics()
        self.assertEqual(stats["total_steps"], 3)

        # Should have statistics from all enabled features
        self.assertIn("arpo_config", stats)
        self.assertIn("cleaner_stats", stats)
        self.assertIn("kv_cache_stats", stats)

        # Overall efficiency should reflect combined impact
        efficiency = stats["overall_efficiency"]
        self.assertTrue(all(rate >= 0 for rate in efficiency.values()))

    def test_state_save_and_load(self):
        config = {
            "enable_arpo": True,
            "enable_cleaner": False,
            "enable_kv_cache": True,
            "device": self.device,
            "hidden_dim": 100,
        }

        trainer = IntegratedAdvancedTrainer(
            self.mock_model, self.mock_reward_model, self.optimizer, config
        )

        # Simulate some training
        batch = {
            "input_ids": torch.randint(0, 100, (1, 5)).to(self.device),
            "attention_mask": torch.ones(1, 5).to(self.device),
            "target_ids": torch.randint(0, 100, (1, 5)).to(self.device),
        }

        trainer.train_step(batch)

        # Save state
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
            save_path = f.name

        trainer.save_optimization_state(save_path)

        # Create new trainer and load state
        new_trainer = IntegratedAdvancedTrainer(
            self.mock_model, self.mock_reward_model, self.optimizer, config
        )
        load_success = new_trainer.load_optimization_state(save_path)

        self.assertTrue(load_success)
        self.assertEqual(new_trainer.training_stats["total_steps"], 1)
        self.assertGreaterEqual(new_trainer.training_stats["arpo_improvements"], 0)

        # Cleanup
        os.unlink(save_path)


if __name__ == "__main__":
    unittest.main()
