
import unittest
import os
import tempfile
import torch.nn as nn
from better_ai.training.fault_localization import FaultLocalizer, SoftwareRepairPipeline

class TestFaultLocalizationNew(unittest.TestCase):
    def setUp(self):
        self.model = nn.Linear(10, 10)
        self.localizer = FaultLocalizer(self.model)

    def test_sbfl_ochiai(self):
        coverage_data = [
            {"passed": False, "covered_lines": {"file1.py": [1, 2]}},
            {"passed": True, "covered_lines": {"file1.py": [1]}},
        ]
        # Line 2 is only covered by failing test -> should have higher suspiciousness than line 1
        scores = self.localizer.calculate_sbfl_scores(coverage_data, technique="ochiai")

        self.assertEqual(len(scores), 2)
        # Ochiai(line2) = 1 / sqrt(1 * (1+0)) = 1.0
        # Ochiai(line1) = 1 / sqrt(1 * (1+1)) = 1/sqrt(2) = 0.7071
        self.assertEqual(scores[0]["line_no"], 2)
        self.assertAlmostEqual(scores[0]["suspiciousness"], 1.0)
        self.assertEqual(scores[1]["line_no"], 1)
        self.assertAlmostEqual(scores[1]["suspiciousness"], 0.70710678)

    def test_sbfl_tarantula(self):
        coverage_data = [
            {"passed": False, "covered_lines": {"file1.py": [1, 2]}},
            {"passed": True, "covered_lines": {"file1.py": [1]}},
        ]
        scores = self.localizer.calculate_sbfl_scores(coverage_data, technique="tarantula")

        # Tarantula(line2) = (1/1) / ((0/1) + (1/1)) = 1.0
        # Tarantula(line1) = (1/1) / ((1/1) + (1/1)) = 0.5
        self.assertEqual(scores[0]["line_no"], 2)
        self.assertAlmostEqual(scores[0]["suspiciousness"], 1.0)
        self.assertEqual(scores[1]["line_no"], 1)
        self.assertAlmostEqual(scores[1]["suspiciousness"], 0.5)

    def test_validate_repair(self):
        from better_ai.training.fault_localization import PatchGenerator, SoftwareRepairPipeline
        generator = PatchGenerator(self.model)
        pipeline = SoftwareRepairPipeline(self.localizer, generator)

        # Success case
        code = "def foo(): return 1"
        test_cmd = "python -c 'import repaired_code; assert repaired_code.foo() == 1'"
        success = pipeline.validate_repair("", code, test_cmd)
        self.assertTrue(success)

        # Failure case
        test_cmd_fail = "python -c 'import repaired_code; assert repaired_code.foo() == 2'"
        failure = pipeline.validate_repair("", code, test_cmd_fail)
        self.assertFalse(failure)

    def test_validate_repair_docker_mock(self):
        from unittest.mock import patch, MagicMock
        from better_ai.training.fault_localization import PatchGenerator, SoftwareRepairPipeline
        generator = PatchGenerator(self.model)
        pipeline = SoftwareRepairPipeline(self.localizer, generator)

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="success", stderr="")

            code = "def foo(): return 1"
            test_cmd = "pytest test_foo.py"
            success = pipeline.validate_repair("", code, test_cmd, use_docker=True)

            self.assertTrue(success)
            # Check if docker was called
            args, kwargs = mock_run.call_args
            self.assertIn("docker run", args[0])
            self.assertIn("python:3.10-alpine", args[0])
            self.assertIn(test_cmd, args[0])

    def test_source_context(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write("def bug():\n    return 1/0\n")
            tmp_path = tmp.name

        try:
            trace = f'  File "{tmp_path}", line 2, in bug\n    return 1/0\nZeroDivisionError: division by zero'
            faults = self.localizer.localize_fault("", trace)
            self.assertEqual(faults[0]["file"], tmp_path)
            self.assertEqual(faults[0]["line_no"], 2)
            self.assertEqual(faults[0]["context"], "return 1/0")
            self.assertIn("`return 1/0`", faults[0]["reason"])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()
