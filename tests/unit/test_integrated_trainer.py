"""
Unit tests for Integrated Advanced Trainer
"""

import pytest
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


class TestIntegratedAdvancedTrainer:
    """Test integrated trainer with all optimizations"""

    @pytest.fixture
    def mock_model(self):
        return nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 100))

    @pytest.fixture
    def mock_reward_model(self):
        class MockRewardModel:
            def score(self, prompt, response):
                return 0.5

        return MockRewardModel()

    @pytest.fixture
    def optimizer(self, mock_model):
        return torch.optim.Adam(mock_model.parameters(), lr=1e-4)

    @pytest.fixture
    def full_config(self):
        return {
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
            "device": torch.device("cpu"),
        }

    def test_initialization_with_all_features(
        self, mock_model, mock_reward_model, optimizer, full_config
    ):
        trainer = IntegratedAdvancedTrainer(
            mock_model, mock_reward_model, optimizer, full_config
        )

        assert trainer.config["enable_arpo"] == True
        assert trainer.config["enable_cleaner"] == True
        assert trainer.config["enable_kv_cache"] == True
        assert trainer.arpo_trainer is not None
        assert trainer.cleaner_collector is not None
        assert trainer.kv_optimized_trainer is not None
        assert trainer.grpo_trainer is None  # Should not be used

    def test_initialization_partial_features(
        self, mock_model, mock_reward_model, optimizer
    ):
        config = {
            "enable_arpo": False,
            "enable_cleaner": True,
            "enable_kv_cache": False,
            "device": torch.device("cpu"),
        }

        trainer = IntegratedAdvancedTrainer(
            mock_model, mock_reward_model, optimizer, config
        )

        assert trainer.arpo_trainer is None
        assert trainer.cleaner_collector is not None
        assert trainer.kv_optimized_trainer is None
        assert trainer.grpo_trainer is not None  # Should fallback to GRPO

    def test_training_stats_tracking(
        self, mock_model, mock_reward_model, optimizer, full_config
    ):
        trainer = IntegratedAdvancedTrainer(
            mock_model, mock_reward_model, optimizer, full_config
        )

        initial_stats = trainer.training_stats
        assert initial_stats["total_steps"] == 0
        assert initial_stats["arpo_improvements"] == 0
        assert initial_stats["cleaner_corrections"] == 0
        assert initial_stats["kv_cache_saves"] == 0

    def test_train_step_with_all_optimizations(
        self, mock_model, mock_reward_model, optimizer, full_config
    ):
        trainer = IntegratedAdvancedTrainer(
            mock_model, mock_reward_model, optimizer, full_config
        )

        batch = {
            "input_ids": torch.randint(0, 100, (2, 5)),
            "attention_mask": torch.ones(2, 5),
            "target_ids": torch.randint(0, 100, (2, 5)),
        }

        metrics = trainer.train_step(batch)

        assert "loss" in metrics or "total_loss" in metrics
        assert trainer.training_stats["total_steps"] == 1

        # Check that component statistics are updated
        stats = trainer.get_comprehensive_statistics()
        assert stats["total_steps"] == 1

    def test_comprehensive_statistics(
        self, mock_model, mock_reward_model, optimizer, full_config
    ):
        trainer = IntegratedAdvancedTrainer(
            mock_model, mock_reward_model, optimizer, full_config
        )

        stats = trainer.get_comprehensive_statistics()

        assert "total_steps" in stats
        assert "arpo_config" in stats
        assert "cleaner_stats" in stats
        assert "kv_cache_stats" in stats
        assert "overall_efficiency" in stats

        # Check efficiency calculations
        efficiency = stats["overall_efficiency"]
        assert "arpo_impact_rate" in efficiency
        assert "cleaner_correction_rate" in efficiency
        assert "kv_cache_saving_rate" in efficiency


class TestIntegratedTrainerFactory:
    """Test factory function for integrated trainer"""

    @pytest.fixture
    def mock_model(self):
        return nn.Linear(10, 100)

    @pytest.fixture
    def mock_reward_model(self):
        class MockRewardModel:
            def score(self, prompt, response):
                return 0.5

        return MockRewardModel()

    @pytest.fixture
    def optimizer(self, mock_model):
        return torch.optim.Adam(mock_model.parameters(), lr=1e-4)

    def test_create_integrated_trainer_with_defaults(
        self, mock_model, mock_reward_model, optimizer
    ):
        # Test with minimal config (should use defaults)
        user_config = {"device": torch.device("cpu")}

        trainer = create_integrated_trainer(
            mock_model, mock_reward_model, optimizer, user_config
        )

        assert trainer.config["enable_arpo"] == True  # Default
        assert trainer.config["enable_cleaner"] == True  # Default
        assert trainer.config["enable_kv_cache"] == True  # Default
        assert trainer.config["entropy_window"] == 10  # Default
        assert trainer.config["max_cache_size"] == 1000  # Default

    def test_create_integrated_trainer_with_user_config(
        self, mock_model, mock_reward_model, optimizer
    ):
        user_config = {
            "enable_arpo": False,
            "enable_cleaner": False,
            "enable_kv_cache": False,
            "entropy_window": 20,
            "custom_setting": "test",
        }

        trainer = create_integrated_trainer(
            mock_model, mock_reward_model, optimizer, user_config
        )

        assert trainer.config["enable_arpo"] == False  # User override
        assert trainer.config["enable_cleaner"] == False  # User override
        assert trainer.config["enable_kv_cache"] == False  # User override
        assert trainer.config["entropy_window"] == 20  # User override
        assert trainer.config["custom_setting"] == "test"  # User setting preserved


class TestFeatureIntegration:
    """Test interaction between different features"""

    @pytest.fixture
    def mock_model(self):
        return nn.Linear(10, 100)

    @pytest.fixture
    def mock_reward_model(self):
        class MockRewardModel:
            def score(self, prompt, response):
                return 0.5

        return MockRewardModel()

    @pytest.fixture
    def optimizer(self, mock_model):
        return torch.optim.Adam(mock_model.parameters(), lr=1e-4)

    def test_feature_coordination(self, mock_model, mock_reward_model, optimizer):
        config = {
            "enable_arpo": True,
            "enable_cleaner": True,
            "enable_kv_cache": True,
            "device": torch.device("cpu"),
        }

        trainer = IntegratedAdvancedTrainer(
            mock_model, mock_reward_model, optimizer, config
        )

        # Simulate a few training steps
        batch = {
            "input_ids": torch.randint(0, 100, (1, 5)),
            "attention_mask": torch.ones(1, 5),
            "target_ids": torch.randint(0, 100, (1, 5)),
        }

        for _ in range(3):
            metrics = trainer.train_step(batch)

        # Check that all features are working together
        stats = trainer.get_comprehensive_statistics()
        assert stats["total_steps"] == 3

        # Should have statistics from all enabled features
        assert "arpo_config" in stats
        assert "cleaner_stats" in stats
        assert "kv_cache_stats" in stats

        # Overall efficiency should reflect combined impact
        efficiency = stats["overall_efficiency"]
        assert all(rate >= 0 for rate in efficiency.values())

    def test_state_save_and_load(self, mock_model, mock_reward_model, optimizer):
        config = {
            "enable_arpo": True,
            "enable_cleaner": False,
            "enable_kv_cache": True,
            "device": torch.device("cpu"),
        }

        trainer = IntegratedAdvancedTrainer(
            mock_model, mock_reward_model, optimizer, config
        )

        # Simulate some training
        batch = {
            "input_ids": torch.randint(0, 100, (1, 5)),
            "attention_mask": torch.ones(1, 5),
            "target_ids": torch.randint(0, 100, (1, 5)),
        }

        trainer.train_step(batch)

        # Save state
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
            save_path = f.name

        trainer.save_optimization_state(save_path)

        # Create new trainer and load state
        new_trainer = IntegratedAdvancedTrainer(
            mock_model, mock_reward_model, optimizer, config
        )
        load_success = new_trainer.load_optimization_state(save_path)

        assert load_success == True
        assert new_trainer.training_stats["total_steps"] == 1
        assert new_trainer.training_stats["arpo_improvements"] >= 0

        # Cleanup
        os.unlink(save_path)


if __name__ == "__main__":
    pytest.main([__file__])
