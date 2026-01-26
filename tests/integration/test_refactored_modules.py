
import unittest
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


class DummyTokenizer:
    def encode(self, text, truncation=True, max_length=None):
        return [ord(c) for c in text]

    @property
    def pad_token_id(self):
        return 0


class TestRefactoredModules(unittest.TestCase):
    def test_hf_datasets_refactoring(self):
        config = {
            "use_rolling_windows": True,
            "primary_dataset": "test",
            "max_train_samples": 10,
            "max_eval_samples": 10,
        }
        tokenizer = DummyTokenizer()
        train_dataloader, eval_dataloader = create_code_dataloaders(
            config, tokenizer, batch_size=2
        )
        self.assertIsNotNone(train_dataloader)
        self.assertIsNotNone(eval_dataloader)

        # Check if we can get a batch
        train_batch = next(iter(train_dataloader))
        self.assertIn("input_ids", train_batch)
        self.assertEqual(train_batch["input_ids"].shape[0], 2)

        eval_batch = next(iter(eval_dataloader))
        self.assertIn("input_ids", eval_batch)
        self.assertEqual(eval_batch["input_ids"].shape[0], 2)

    def test_advanced_features_refactoring(self):
        hidden_dim = 128
        self.assertIsNotNone(RecursiveScratchpad(hidden_dim))
        self.assertIsNotNone(CoTSpecializationHeads(hidden_dim))
        self.assertIsNotNone(InnerMonologue(hidden_dim))
        self.assertIsNotNone(STaRModule(hidden_dim))
        self.assertIsNotNone(ToolUseHeads(hidden_dim))
        self.assertIsNotNone(GBNFConstraint(hidden_dim))
        self.assertIsNotNone(JSONEnforcer(hidden_dim))
        self.assertIsNotNone(EntropicSteering(hidden_dim))


if __name__ == "__main__":
    unittest.main()
