
import unittest
from unittest.mock import patch, MagicMock
from better_ai.data.hf_datasets import create_code_dataloaders
from better_ai.models.advanced_features import (
    RecursiveScratchpad,
    CoTSpecializationHeads,
    InnerMonologue,
    STaRModule,
    ToolUseHeads,
    GBNFConstraint,
    JSONEnforcer,
    EntropicSteering,
)
import torch
from better_ai.test_resource_tags import low_resource

class DummyTokenizer:
    def encode(self, text, truncation=True, max_length=None):
        return [ord(c) for c in text]

    @property
    def pad_token_id(self):
        return 0

@low_resource
class TestRefactoredModules(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @patch("better_ai.data.unified_dataloader.load_dataset")
    def test_hf_datasets_refactoring(self, mock_load_dataset):
        # Setup mock dataset
        mock_data = [
            {"text": "def hello(): print('world')", "lang": "Python"},
            {"text": "int main() { return 0; }", "lang": "C"},
            {"text": "fn main() { println!(\"Hello\"); }", "lang": "Rust"},
            {"text": "print('more python')", "lang": "Python"},
        ]
        mock_iterable_dataset = MagicMock()
        mock_iterable_dataset.filter.return_value = mock_data
        mock_iterable_dataset.__iter__.return_value = iter(mock_data)
        mock_load_dataset.return_value = mock_iterable_dataset

        config = {
            "use_rolling_windows": True,
            "primary_dataset": "test",
            "max_train_samples": 10,
            "max_eval_samples": 10,
        }

        # Mock tokenizer that returns tensors
        tokenizer = MagicMock()
        tokenizer.side_effect = lambda text, **kwargs: {
            "input_ids": torch.zeros((1, 1024), dtype=torch.long),
            "attention_mask": torch.ones((1, 1024), dtype=torch.long)
        }
        tokenizer.pad_token_id = 0

        train_dataloader, eval_dataloader = create_code_dataloaders(
            config, tokenizer, batch_size=2
        )
        self.assertIsNotNone(train_dataloader)
        self.assertIsNotNone(eval_dataloader)

        # Check if we can get a batch
        train_iter = iter(train_dataloader)
        train_batch = next(train_iter)
        self.assertIn("input_ids", train_batch)
        self.assertEqual(train_batch["input_ids"].shape[0], 2)

        eval_iter = iter(eval_dataloader)
        eval_batch = next(eval_iter)
        self.assertIn("input_ids", eval_batch)
        self.assertEqual(eval_batch["input_ids"].shape[0], 2)

    def test_advanced_features_refactoring(self):
        hidden_dim = 16
        self.assertIsNotNone(RecursiveScratchpad(hidden_dim).to(self.device))
        self.assertIsNotNone(CoTSpecializationHeads(hidden_dim).to(self.device))
        self.assertIsNotNone(InnerMonologue(hidden_dim).to(self.device))
        self.assertIsNotNone(STaRModule(hidden_dim).to(self.device))
        self.assertIsNotNone(ToolUseHeads(hidden_dim).to(self.device))
        self.assertIsNotNone(GBNFConstraint(hidden_dim).to(self.device))
        self.assertIsNotNone(JSONEnforcer(hidden_dim).to(self.device))
        self.assertIsNotNone(EntropicSteering(hidden_dim).to(self.device))


if __name__ == "__main__":
    unittest.main()
