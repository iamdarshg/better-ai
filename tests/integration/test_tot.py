
import torch
import unittest
from better_ai.models.tot import TreeOfThought
from better_ai.models.enhanced_model import EnhancedDeepSeekModel
from better_ai.config import ModelConfig
from better_ai.test_config_utils import get_small_model_config

class TestTreeOfThought(unittest.TestCase):
    def test_search(self):
        config = get_small_model_config()
        model = EnhancedDeepSeekModel(config)
        tot = TreeOfThought(model, config)
        initial_state = "initial state"
        best_thought = tot.search(initial_state, num_iterations=2, k=2)
        self.assertIsInstance(best_thought, str)
        self.assertTrue(best_thought.startswith(initial_state))

if __name__ == '__main__':
    unittest.main()
