
import torch
import unittest
import sys
import os

sys.path.append(".")
from better_ai.config import ModelConfig
from better_ai.models.core import DeepSeekModel
from better_ai.models.features.recursive_scratchpad import RecursiveScratchpad
from better_ai.test_resource_tags import low_resource

@low_resource
class TestOuroScratchpad(unittest.TestCase):
    def setUp(self):
        self.hidden_dim = 128
        self.private_subspace_dim = 96
        self.device = torch.device("cpu")

        self.module = RecursiveScratchpad(
            hidden_dim=self.hidden_dim,
            private_subspace_dim=self.private_subspace_dim,
            max_iterations=3
        ).to(self.device)

    def test_scratchpad_isolation(self):
        """Test RecursiveScratchpad with a mock layer function."""
        batch_size = 2
        seq_len = 16
        hidden_states = torch.randn(batch_size, seq_len, self.hidden_dim)

        # Mock layers_fn that just adds a small value
        def mock_layers_fn(h):
            return h + 0.01

        # Disable halting for this test
        with torch.no_grad():
            self.module.halting_gate[2].bias.fill_(-10.0)

        outputs = self.module(hidden_states, layers_fn=mock_layers_fn)

        self.assertIn("scratchpad_output", outputs)
        self.assertIn("reasoning_traces", outputs)
        self.assertIn("iteration_count", outputs)
        self.assertIn("latent_entropy", outputs)

        self.assertEqual(outputs["scratchpad_output"].shape, (batch_size, seq_len, self.hidden_dim))
        # 3 iterations + initial state = 4 traces
        self.assertEqual(outputs["reasoning_traces"].shape, (batch_size, 4, seq_len, self.private_subspace_dim))
        self.assertEqual(outputs["iteration_count"], 3)

    def test_halting_logic(self):
        """Test that the module can halt early."""
        batch_size = 1
        seq_len = 4
        hidden_states = torch.randn(batch_size, seq_len, self.hidden_dim)

        # Mock layers_fn
        def mock_layers_fn(h):
            return h

        # Manually set halting gate weights to trigger halt
        with torch.no_grad():
            # halting_gate is nn.Sequential(Linear, ReLU, Linear, Sigmoid)
            # We want to force output > 0.5
            # Accessing the last linear layer (index 2)
            self.module.halting_gate[2].bias.fill_(10.0)

        outputs = self.module(hidden_states, layers_fn=mock_layers_fn, max_iterations=5)

        # Should halt after 1 iteration (iteration 0 is the first pass in the loop)
        # traces will have initial state + 1st loop result = 2 traces
        self.assertEqual(outputs["iteration_count"], 1)
        self.assertEqual(outputs["reasoning_traces"].shape[1], 2)

    def test_model_integration(self):
        """Test DeepSeekModel calls the scratchpad correctly."""
        config = ModelConfig(
            hidden_dim=64,
            num_layers=2,
            num_attention_heads=4,
            vocab_size=1000,
            use_recursive_scratchpad=True,
            scratchpad_max_iterations=2,
            private_subspace_dim=48
        )
        model = DeepSeekModel(config).to(self.device)

        # Disable halting
        with torch.no_grad():
            model.scratchpad.halting_gate[2].bias.fill_(-10.0)

        input_ids = torch.randint(0, 1000, (1, 8))

        # Forward pass with advanced features
        outputs = model(input_ids, return_advanced_features=True)

        self.assertIn("advanced_features", outputs)
        self.assertIn("scratchpad", outputs["advanced_features"])

        scratchpad_out = outputs["advanced_features"]["scratchpad"]
        self.assertEqual(scratchpad_out["iteration_count"], 2)

        # Verify it went through layers (this is harder to check directly without hooks,
        # but if it didn't crash and returns the right shapes, it's a good sign).

if __name__ == "__main__":
    unittest.main()
