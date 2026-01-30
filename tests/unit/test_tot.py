#!/usr/bin/env python3
import unittest
from unittest.mock import patch
from better_ai.models.tot import TreeOfThought


class DummyConfig:
    hidden_dim = 16
    vocab_size = 64
    max_seq_length = 128
    num_layers = 2


class TestTreeOfThoughtMocks(unittest.TestCase):
    def test_search_with_single_mock_thought(self):
        class DummyModel:
            pass

        config = DummyConfig()
        tot = TreeOfThought(DummyModel(), config)

        with patch.object(tot, "generate_thoughts", return_value=["mock_thought"]):
            with patch.object(tot, "evaluate_states", return_value=[1.0]):
                result = tot.search("start_state", num_iterations=1, k=1)
                self.assertIn(
                    result, ["mock_thought"]
                )  # should return the only generated thought

    def test_search_with_multiple_thoughts_mock(self):
        class DummyModel:
            pass

        config = DummyConfig()
        tot = TreeOfThought(DummyModel(), config)

        with patch.object(tot, "generate_thoughts", return_value=["a", "b"]):
            with patch.object(tot, "evaluate_states", return_value=[0.1, 0.9]):
                # Even with multiple thoughts, we only validate that it returns a valid thought
                result = tot.search("start_state", num_iterations=1, k=2)
                self.assertIn(result, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
