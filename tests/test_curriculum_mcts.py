#!/usr/bin/env python3
import unittest
<<<<<<< HEAD:tests/unit/test_curriculum_mcts.py
from better_ai.test_resource_tags import high_resource
=======
import sys
import os
>>>>>>> 6ee6a9026156a3d656f792dbcbf9395f94c9f6e7:tests/test_curriculum_mcts.py
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from better_ai.training.cosine_curriculum import (
    CosineCurriculumScheduler,
    CurriculumConfig,
)
from better_ai.training.mcts_cot import (
    MCTSCoTSearcher,
    MCTSConfig,
    create_mcts_cot_searcher,
)
from better_ai.training.curriculum_mcts_trainer import (
    CurriculumMCTSTrainer,
    CurriculumMCTSConfig,
    create_curriculum_mcts_trainer,
)


@high_resource
class TestCurriculumMCTS(unittest.TestCase):
    def test_cosine_scheduler_progress(self):
        cfg = CurriculumConfig(total_steps=100, warmup_steps=10)
        sched = CosineCurriculumScheduler(cfg)
        for _ in range(5):
            sched.step()
        self.assertTrue(0.0 <= sched.difficulty_history[-1] <= 1.0)

    @unittest.skip("MCTS rollout depends on LM; skip in CI")
    def test_mcts_searcher_basic(self):
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def generate(self, **kwargs):
                return [torch.tensor([1, 2, 3])]

        class MockTokenizer:
            eos_token_id = 0

            def __call__(
                self, text, return_tensors=None, truncation=None, max_length=None
            ):
                return {"input_ids": torch.randint(0, 1000, (1, 8))}

            def decode(self, tok, skip_special_tokens=True):
                return "Reasoning step: test"

        model = MockModel()
        tokenizer = MockTokenizer()
        cfg = MCTSConfig(
            max_iterations=2, max_depth=3, max_nodes=50, max_children_per_node=2
        )
        searcher = MCTSCoTSearcher(model, tokenizer, cfg)
        res = searcher.search("What is 2+2?")
        self.assertIn("best_reasoning_trace", res)
        self.assertIn("best_answer", res)

    @unittest.skip("Heavy integration test; skip in CI")
    def test_integration_trainer_factory(self):
        # Basic smoke test for factory that binds curriculum + mcts trainer
        from better_ai.training.curriculum_mcts_trainer import (
            create_curriculum_mcts_trainer,
        )

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

        trainer = create_curriculum_mcts_trainer(
            model=DummyModel(), tokenizer=None, config=None, training_config=None
        )
        self.assertIsNotNone(trainer)


if __name__ == "__main__":
    unittest.main()
