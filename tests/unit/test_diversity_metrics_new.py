
import unittest
import torch
from better_ai.training.diversity_metrics import DiversityMeasurer, get_diversity_reward

class TestDiversityMetricsNew(unittest.TestCase):
    def setUp(self):
        self.measurer = DiversityMeasurer()

    def test_step_diversity(self):
        trajectories = [
            "Step 1: Think\n\nStep 2: Act",
            "Step 1: Think\n\nStep 2: Wait"
        ]
        # Steps: ["Think", "Act", "Think", "Wait"]
        # Unique: {"Think", "Act", "Wait"} = 3
        # Total: 4
        # Div: 3/4 = 0.75
        div = self.measurer.measure_step_diversity(trajectories)
        self.assertEqual(div, 0.75)

    def test_reasoning_patterns(self):
        traj = "I should verify the result. Let me try another way."
        patterns = self.measurer.label_reasoning_patterns(traj)
        self.assertIn("verification", patterns)
        self.assertIn("exploratory", patterns)

        traj2 = "actually I made a mistake, let me go back."
        patterns2 = self.measurer.label_reasoning_patterns(traj2)
        self.assertIn("backtracking", patterns2)

    def test_pattern_diversity(self):
        trajectories = [
            "verify and check",
            "maybe try this",
            "analyze the sum"
        ]
        # Patterns: verification, exploratory, analytical, mathematical
        # Unique: 4
        # Div: 4/5 = 0.8
        div = self.measurer.compute_pattern_diversity(trajectories)
        self.assertEqual(div, 0.8)

    def test_aggregate_reward(self):
        trajectories = ["A", "B"]
        reward = get_diversity_reward(trajectories)
        self.assertGreater(reward, 0.0)
        self.assertLessEqual(reward, 1.0)

if __name__ == "__main__":
    unittest.main()
