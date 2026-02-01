import torch
import unittest
from better_ai.models.tot import TreeOfThought
from better_ai.models.enhanced_model import EnhancedDeepSeekModel
from better_ai.test_config_utils import get_small_model_config
from better_ai.test_resource_tags import low_resource, high_resource

@low_resource
class TestTreeOfThought(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_search(self):
        config = get_small_model_config()
        model = EnhancedDeepSeekModel(config).to(self.device)
        tot = TreeOfThought(model, config)
        initial_state = "initial state"
        best_thought = tot.search(initial_state, num_iterations=2, k=2)
        self.assertIsInstance(best_thought, str)
        self.assertTrue(best_thought.startswith(initial_state))


if __name__ == "__main__":
    unittest.main()
