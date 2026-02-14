
import unittest
import torch
from better_ai.training.diversity_metrics import DiversityMeasurer, get_diversity_reward
from better_ai.test_resource_tags import low_resource

@low_resource
class TestDiversityMetrics(unittest.TestCase):
    def setUp(self):
        self.measurer = DiversityMeasurer()

    def test_approach_classification(self):
        traj_dp = "I will use dynamic programming and a table to solve this."
        traj_greedy = "A greedy approach picking the local optimum at each step."
        traj_none = "I am just thinking about the problem."

        self.assertEqual(self.measurer.classify_approach(traj_dp), "dynamic_programming")
        self.assertEqual(self.measurer.classify_approach(traj_greedy), "greedy")
        self.assertEqual(self.measurer.classify_approach(traj_none), "unknown")

    def test_approach_diversity(self):
        trajs = [
            "Use DP here",
            "Try a greedy solution",
            "Another DP approach",
            "Brute force it"
        ]
        # Unique: DP, Greedy, Brute Force -> 3/4 = 0.75
        score = self.measurer.compute_approach_diversity(trajs)
        self.assertEqual(score, 0.75)

    def test_embedding_diversity(self):
        # Similar embeddings
        emb_similar = torch.tensor([
            [1.0, 0.0],
            [0.99, 0.01]
        ])
        div_low = self.measurer.compute_embedding_diversity(emb_similar)

        # Diverse embeddings
        emb_diverse = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        div_high = self.measurer.compute_embedding_diversity(emb_diverse)

        self.assertGreater(div_high, div_low)

    def test_get_diversity_reward(self):
        trajs = ["Step 1", "Step 2"]
        reward = get_diversity_reward(trajs)
        self.assertIsInstance(reward, float)
        self.assertGreater(reward, 0)

if __name__ == "__main__":
    unittest.main()
