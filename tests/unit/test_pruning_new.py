
import unittest
import torch
import torch.nn as nn
from better_ai.training.pruning import shrink_model_after_pruning, prune_attention_heads

class TestPruningNew(unittest.TestCase):
    def test_row_and_bias_shrinkage(self):
        layer = nn.Linear(4, 4, bias=True)
        # Prune 2nd and 3rd rows
        layer.weight.data[1, :] = 0
        layer.weight.data[2, :] = 0
        # Prune 4th column
        layer.weight.data[:, 3] = 0

        # Original state
        # Row 0: non-zero
        # Row 1: all zero
        # Row 2: all zero
        # Row 3: non-zero (if we set something)
        layer.weight.data[0, 0] = 1.0
        layer.weight.data[3, 0] = 1.0

        # Bias
        layer.bias.data = torch.tensor([1.0, 2.0, 3.0, 4.0])

        model = nn.Sequential(layer)
        shrink_model_after_pruning(model)

        new_layer = model[0]
        # Should be (2, 3) because 2 rows kept, 3 columns kept
        self.assertEqual(new_layer.in_features, 3)
        self.assertEqual(new_layer.out_features, 2)
        # Bias should be [1.0, 4.0]
        self.assertEqual(new_layer.bias.data[0], 1.0)
        self.assertEqual(new_layer.bias.data[1], 4.0)

    def test_head_pruning(self):
        from better_ai.models.core import MultiHeadAttention
        # 4 heads, head_dim 2 -> hidden_size 8
        attn = MultiHeadAttention(hidden_size=8, num_heads=4, num_key_value_heads=2, head_dim=2)

        class MockLayer:
            def __init__(self, attn):
                self.self_attn = attn

        class MockModel:
            def __init__(self, layer):
                self.layers = [layer]

        layer = MockLayer(attn)
        model = MockModel(layer)

        # Prune head 1 (of 0,1,2,3)
        # KV heads are 0 (shares Q 0,1) and 1 (shares Q 2,3)
        # Pruning head 1 still leaves Q head 0, so KV head 0 is kept.
        prune_attention_heads(model, {0: [1]})

        self.assertEqual(attn.num_heads, 3)
        self.assertEqual(attn.q_proj.out_features, 6) # 3 * 2
        self.assertEqual(attn.o_proj.in_features, 6)
        self.assertEqual(attn.num_key_value_heads, 2) # Both still needed

        # Reset and try pruning both at once to see KV head removal
        attn2 = MultiHeadAttention(hidden_size=8, num_heads=4, num_key_value_heads=2, head_dim=2)
        layer2 = MockLayer(attn2)
        model2 = MockModel(layer2)

        # Pruning heads 0 and 1 (which share KV head 0)
        prune_attention_heads(model2, {0: [0, 1]})
        self.assertEqual(attn2.num_heads, 2)
        self.assertEqual(attn2.num_key_value_heads, 1) # KV head 0 removed
        self.assertEqual(attn2.k_proj.out_features, 2)

if __name__ == "__main__":
    unittest.main()
