"""
Integration tests for Machine Feedback RLHF pipeline
"""

import unittest
import torch
<<<<<<< HEAD
from better_ai.training.machine_feedback import (
    MachineFeedbackReward,
    MachineFeedbackTrainer,
)
from better_ai.models.core import DeepSeekModel
from better_ai.test_config_utils import get_small_model_config
from better_ai.test_resource_tags import low_resource, high_resource

@low_resource
=======
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from better_ai.training.machine_feedback import MachineFeedbackReward, MachineFeedbackTrainer
from better_ai.models.core import DeepSeekModel
from better_ai.config import ModelConfig
from better_ai.test_config_utils import get_small_model_config

>>>>>>> 6ee6a9026156a3d656f792dbcbf9395f94c9f6e7
class TestMachineFeedbackRLHF(unittest.TestCase):
    def test_machine_feedback_reward(self):
        reward_engine = MachineFeedbackReward({"grammar_type": "python"})

        # Good code
        good_code = "def hello():\n    print('world')"
        good_reward = reward_engine.compute_reward(good_code)

        # Bad code (syntax error)
        bad_code = "def hello("
        bad_reward = reward_engine.compute_reward(bad_code)

        # Code with style issues
        style_code = "def hello():\n    pass\n    pass\n    pass\n    pass\n    pass\n    pass\n    pass\n    pass\n    pass\n    pass\n    pass"
        style_reward = reward_engine.compute_reward(style_code)

        self.assertGreater(good_reward, bad_reward)
        self.assertGreater(good_reward, style_reward)
        print(f"Good: {good_reward}, Bad: {bad_reward}, Style: {style_reward}")

    def test_mf_trainer_step(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = get_small_model_config()
        config.vocab_size = 100
        config.num_layers = 1

        model = DeepSeekModel(config).to(device)
        # Mock tokenizer if needed

        trainer = MachineFeedbackTrainer(model, {"group_size": 2, "max_new_tokens": 10})

        batch = {
            'input_ids': torch.randint(0, 100, (1, 5)).to(device)
        }

        metrics = trainer.train_step(batch)

        self.assertIn("mf_reward_mean", metrics)
        self.assertIsInstance(metrics["mf_reward_mean"], float)

if __name__ == "__main__":
    unittest.main()
