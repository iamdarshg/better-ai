#!/usr/bin/env python3
import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from better_ai.training.curriculum_mcts_trainer import (
    CurriculumMCTSTrainer,
    CurriculumMCTSConfig,
)


class TestCurriculumMCTSTrainerMocks(unittest.TestCase):
    def test_train_step_with_mocks(self):
        # Create a minimal fake trainer via object.__new__ to bypass heavy init
        trainer = object.__new__(CurriculumMCTSTrainer)

        trainer.model = nn.Linear(8, 8)
        trainer.reward_model = nn.Linear(8, 1)
        trainer.optimizer = object()
        trainer.tokenizer = None
        trainer.config = CurriculumMCTSConfig(enable_curriculum=True, enable_mcts=True)
        trainer.training_config = type(
            "Cfg", (), {"learning_rate": 1e-3, "batch_size": 2}
        )()

        # Mock components
        trainer.curriculum_scheduler = None
        trainer.mcts_searcher = MagicMock()
        trainer.mcts_searcher.search.return_value = {
            "best_value": 0.6,
            "best_reasoning_trace": ["step1", "step2"],
            "best_answer": "42",
        }
        trainer.grpo_trainer = MagicMock()
        trainer._prepare_training_batch = MagicMock(
            return_value={
                "input_ids": torch.zeros((1, 4), dtype=torch.long),
                "attention_mask": torch.ones((1, 4)),
            }
        )
        trainer._perform_grpo_update = MagicMock(return_value={"grpo_loss": 0.0})

        trainer.current_step = 0
        trainer.training_metrics = []
        trainer.mcts_generated_data = []

        batch = {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4)),
        }
        # Run a single step; should return a dict
        res = trainer.train_step(batch)
        self.assertIsInstance(res, dict)


if __name__ == "__main__":
    unittest.main()
