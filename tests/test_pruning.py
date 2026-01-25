import unittest
import torch
import torch.nn as nn
from better_ai.training.pruning import prune_expert_widths

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.expert = nn.Linear(10, 10)
        self.non_expert = nn.Linear(10, 10)

class TestPruning(unittest.TestCase):
    def test_prune_expert_widths(self):
        model = SimpleModel()
        pruning_ratio = 0.5

        # Save original weights for comparison
        original_expert_weights = model.expert.weight.data.clone()
        original_non_expert_weights = model.non_expert.weight.data.clone()

        prune_expert_widths(model, pruning_ratio, ["expert"])

        # Check that some weights in the expert layer have been pruned (zeroed out)
        self.assertTrue(torch.any(model.expert.weight.data == 0))

        # Check that the non-expert layer's weights are unchanged
        self.assertTrue(torch.all(torch.eq(model.non_expert.weight.data, original_non_expert_weights)))

        # Check that the number of pruned weights is correct (50% of columns)
        num_pruned = torch.sum(model.expert.weight.data == 0)
        num_cols_pruned = num_pruned / model.expert.weight.data.shape[0]
        total_cols = model.expert.weight.data.shape[1]
        self.assertEqual(num_cols_pruned, int(pruning_ratio * total_cols))

if __name__ == "__main__":
    unittest.main()
