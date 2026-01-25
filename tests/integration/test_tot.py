
import torch
import unittest
from better_ai.models.tot import TreeOfThought
from better_ai.models.enhanced_model import EnhancedDeepSeekModel
from better_ai.config import ModelConfig

class TestTreeOfThought(unittest.TestCase):
    def test_search(self):
        config = ModelConfig()
        model = EnhancedDeepSeekModel(config)
        tot = TreeOfThought(model, config)
        initial_state = "initial state"
        best_thought = tot.search(initial_state, num_iterations=10, k=3)
        self.assertIsInstance(best_thought, str)
        self.assertTrue(best_thought.startswith(initial_state))

if __name__ == '__main__':
    unittest.main()
