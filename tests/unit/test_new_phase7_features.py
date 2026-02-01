"""
Unit tests for new Phase 7 features
Tests STeCa, Fault Localization, Structural Signal, Trace Validity, Diversity, AHA-Moments, and Curation
"""

import unittest
import torch
import torch.nn as nn
from better_ai.training.steca import TrajectoryRefiner, STeCaController
from better_ai.training.fault_localization import SoftwareRepairPipeline, FaultLocalizer, PatchGenerator
from better_ai.models.features.reasoning_rewards import StructuralSignalReward, TraceValidityScorer, AHAMomentDetector
from better_ai.training.diversity_metrics import DiversityMeasurer
from better_ai.data.curation import AgentFLANDecomposer, DatasetCurator
from better_ai.utils.verification import Z3Verifier, PythonASTVerifier
from better_ai.test_resource_tags import high_resource
@high_resource
class TestPhase7Features(unittest.TestCase):
    def setUp(self):
        self.model = nn.Linear(10, 10) # Mock model
        self.tokenizer = None # Mock tokenizer

    def test_steca_refinement(self):
        refiner = TrajectoryRefiner(self.model, self.tokenizer)
        trajectory = [{"content": "Step 1: Define x"}, {"content": "Step 2: Compute x+1"}]
        refined = refiner.calibrate_trajectory(trajectory)

        self.assertEqual(len(refined), 2)
        self.assertIn("original_content", refined[0])
        self.assertIn("reflection", refined[0])
        self.assertEqual(refined[0]["content"], "[REFINED STEP]")

    def test_fault_localization(self):
        localizer = FaultLocalizer(self.model)
        generator = PatchGenerator(self.model)
        pipeline = SoftwareRepairPipeline(localizer, generator)

        results = pipeline.repair("def fix_me(): return arr[10]", "IndexError: list index out of range")
        self.assertEqual(results["status"], "success")
        self.assertGreater(len(results["faults"]), 0)
        self.assertGreater(len(results["patches"]), 0)

    def test_structural_signal_reward(self):
        reward_engine = StructuralSignalReward()
        text = "<thought>Thinking...</thought><action>Doing...</action><observation>Done.</observation>"
        score = reward_engine.compute_reward(text)
        self.assertGreater(score, 0.5)

        bad_text = "No tags here"
        bad_score = reward_engine.compute_reward(bad_text)
        self.assertLess(bad_score, score)

    def test_trace_validity_scoring(self):
        scorer = TraceValidityScorer(self.model)
        trace = ["I need to solve X", "Therefore, I do Y", "The answer is Z"]
        goal = "Solve X"
        score = scorer.score_trace(trace, goal)
        self.assertGreater(score, 0.6)

    def test_diversity_metrics(self):
        measurer = DiversityMeasurer()
        trajectories = ["A B C", "D E F", "A B C"]
        score = measurer.compute_n_gram_diversity(trajectories)
        self.assertLess(score, 1.0)

        unique_trajectories = ["A B C", "D E F", "G H I"]
        unique_score = measurer.compute_n_gram_diversity(unique_trajectories)
        self.assertGreater(unique_score, score)

    def test_aha_moment_detection(self):
        detector = AHAMomentDetector()
        trace = "I thought x was 5. Wait, I see that x is actually 10!"
        moments = detector.detect_aha_moments(trace)
        self.assertGreater(len(moments), 0)

        reward = detector.compute_aha_reward(trace)
        self.assertGreater(reward, 0)

    def test_agent_flan_decomposition(self):
        decomposer = AgentFLANDecomposer()
        trajectory = [{"content": "Step 1"}, {"content": "Step 2"}]
        decomposed = decomposer.decompose_trajectory(trajectory)
        self.assertEqual(len(decomposed), 2)
        self.assertTrue(decomposed[1]["prompt"].startswith("Context: Step 1"))

    def test_dataset_curation(self):
        curator = DatasetCurator(quality_threshold=0.8)
        dataset = [
            {"text": "Good", "quality_score": 0.9},
            {"text": "Bad", "quality_score": 0.5}
        ]
        filtered = curator.filter_by_quality(dataset)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["text"], "Good")

    def test_formal_verification(self):
        # Python AST Verifier
        verifier = PythonASTVerifier()
        success, msg = verifier.verify("def foo(): return 1", ["foo() == 1"])
        self.assertTrue(success)

        # Invalid python
        fail, msg = verifier.verify("def foo(:", [])
        self.assertFalse(fail)

        # Z3 Verifier (may be mock if Z3 not installed)
        z3_verifier = Z3Verifier()
        # Even if not available, it should return False and not crash
        res, msg = z3_verifier.verify("x=5", "x > 0")
        if z3_verifier.z3_available:
            self.assertTrue(res)
        else:
            self.assertFalse(res)
            self.assertEqual(msg, "Z3 not installed")

if __name__ == "__main__":
    unittest.main()
