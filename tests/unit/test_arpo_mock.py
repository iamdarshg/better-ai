#!/usr/bin/env python3
import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from better_ai.training.arpo import ARPOTrainer


class TestARPOTrainerMock(unittest.TestCase):
    def test_train_step_with_mocked_rollouts(self):
        model = nn.Linear(16, 16)
        reward_model = nn.Linear(16, 1)
        trainer = object.__new__(ARPOTrainer)
        trainer.model = model
        trainer.reward_model = reward_model
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer.config = {"entropy_window": 5, "entropy_threshold": 2.0}
        trainer.step_count = 0
        trainer.eps_clip = 0.2
        trainer.entropy_coef = 0.01
        trainer.device = torch.device("cpu")
        trainer.entropy_monitor = type(
            "EM",
            (),
            {
                "update": lambda self, x: {
                    "current_entropy": 0.5,
                    "baseline_entropy": 0.4,
                    "is_spike": False,
                }
            },
        )()
        trainer.rollout_manager = type(
            "RM",
            (),
            {
                "get_branch_factor": lambda self, e: 1,
                "base_branch_factor": 1,
                "max_branch_factor": 4,
            },
        )()
        trainer.advantage_attributor = type(
            "AA",
            (),
            {
                "compute_step_attributions": lambda *args, **kwargs: torch.zeros(
                    2,
                ).requires_grad_(False)
            },
        )()

        # Mock generation
        trainer.generate_with_adaptive_rollouts = (
            lambda self,
            prompts,
            max_length=512,
            temperature=0.7,
            enable_adaptive=True: [
                {
                    "primary_rollout": {"logits": torch.randn(1, 4, 1000)},
                    "entropy_analysis": {"is_spike": False},
                    "all_rollouts": [],
                }
            ]
        )

        trainer._extract_prompts_from_batch = lambda self, batch: ["dummy"]
        trainer._tokenize = lambda self, text: {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4)),
        }
        trainer._detect_tool_uses = lambda self, o: []
        trainer._score_rollouts = lambda self, rollouts, prompt: [{"reward_score": 0.0}]
        trainer.train_step = ARPOTrainer.train_step.__get__(trainer, ARPOTrainer)

        batch = {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4)),
        }
        res = trainer.train_step(batch)
        self.assertIn("total_loss", res)


if __name__ == "__main__":
    unittest.main()
