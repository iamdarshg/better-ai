"""
Integration tests for Machine Feedback RLHF pipeline
"""

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from better_ai.training.machine_feedback import MachineFeedbackReward, MachineFeedbackTrainer
from better_ai.models.core import DeepSeekModel
from better_ai.config import ModelConfig

def test_machine_feedback_reward():
    reward_engine = MachineFeedbackReward()

    # Good code
    good_code = "def hello():\n    print('world')"
    good_reward = reward_engine.compute_reward(good_code)

    # Bad code (syntax error)
    bad_code = "def hello("
    bad_reward = reward_engine.compute_reward(bad_code)

    # Code with style issues
    style_code = "def hello():\n    pass\n    pass\n    pass\n    pass\n    pass\n    pass\n    pass\n    pass\n    pass\n    pass\n    pass"
    style_reward = reward_engine.compute_reward(style_code)

    assert good_reward > bad_reward
    assert good_reward > style_reward
    print(f"Good: {good_reward}, Bad: {bad_reward}, Style: {style_reward}")

def test_mf_trainer_step():
    config = ModelConfig()
    config.vocab_size = 100
    config.hidden_dim = 32
    config.num_layers = 1

    model = DeepSeekModel(config)
    # Mock tokenizer if needed

    trainer = MachineFeedbackTrainer(model, {"group_size": 2, "max_new_tokens": 10})

    batch = {
        'input_ids': torch.randint(0, 100, (1, 5))
    }

    metrics = trainer.train_step(batch)

    assert "mf_reward_mean" in metrics
    assert isinstance(metrics["mf_reward_mean"], float)

if __name__ == "__main__":
    pytest.main([__file__])
