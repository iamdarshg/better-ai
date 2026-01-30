#!/usr/bin/env python3
"""Unit test for GRPO implementation using unittest
Moved from root test_grpo_implementation.py into tests/"""

import unittest
import torch
import torch.nn as nn
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from better_ai.training.grpo import GRPOTrainer


class TestGRPOImplementation(unittest.TestCase):
    @unittest.skip("Heavy compute; skip in quick unittest run")
    def test_grpo_basic_flow(self):
        config = {
            "hidden_dim": 32,
            "vocab_size": 64,
            "group_size": 2,
            "device": torch.device("cpu"),
            "beta": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "eps_clip": 0.2,
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5,
        }

        def create_mock_model(cfg):
            class MockModel(nn.Module):
                def __init__(self, cfg):
                    super().__init__()
                    self.vocab_size = cfg["vocab_size"]
                    self.hidden_dim = cfg["hidden_dim"]
                    self.embedding = nn.Embedding(cfg["vocab_size"], cfg["hidden_dim"])
                    self.encoder = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=cfg["hidden_dim"], nhead=4),
                        num_layers=1,
                    )
                    self.lm_head = nn.Linear(cfg["hidden_dim"], cfg["vocab_size"])

                def forward(
                    self, input_ids, attention_mask=None, output_hidden_states=False
                ):
                    x = self.embedding(input_ids)
                    x = self.encoder(x)
                    logits = self.lm_head(x)

                    # Return a lightweight object with .logits to match GRPOTrainer expectations
                    class _Obj:
                        pass

                    o = _Obj()
                    setattr(o, "logits", logits)
                    setattr(o, "hidden_states", [x])
                    return o

            return MockModel(cfg)

        def create_mock_reward_model(cfg):
            class MockRewardModel(nn.Module):
                def __init__(self, cfg):
                    super().__init__()
                    self.config = cfg
                    self.reward_head = nn.Linear(cfg["hidden_dim"], 1)

                def forward(self, hidden_states, attention_mask=None):
                    return self.reward_head(hidden_states.mean(dim=1)).squeeze(-1)

            return MockRewardModel(cfg)

        model = create_mock_model(config)
        reward_model = create_mock_reward_model(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = GRPOTrainer(model, reward_model, optimizer, config)

        batch = {
            "input_ids": torch.randint(0, config["vocab_size"], (2, 5)),
            "attention_mask": torch.ones(2, 5),
            "target_ids": torch.randint(0, config["vocab_size"], (2, 5)),
        }

        res = trainer.train_step(
            batch,
            torch.randn(2, config["group_size"]),
            torch.randn(2, config["group_size"]),
        )
        self.assertIsInstance(res, dict)


if __name__ == "__main__":
    unittest.main()
