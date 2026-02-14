
import unittest
from better_ai.training.fault_localization import FaultLocalizer, PatchGenerator, SoftwareRepairPipeline, compute_repair_reward
import torch.nn as nn
from better_ai.test_resource_tags import low_resource

@low_resource
class TestFaultLocalization(unittest.TestCase):
    def setUp(self):
        self.model = nn.Linear(10, 10)
        self.localizer = FaultLocalizer(self.model)
        self.generator = PatchGenerator(self.model)
        self.pipeline = SoftwareRepairPipeline(self.localizer, self.generator)

    def test_python_trace_parsing(self):
        trace = """
Traceback (most recent call last):
  File "app.py", line 5, in <module>
    main()
  File "app.py", line 10, in main
    result = process_data(data)
  File "utils.py", line 25, in process_data
    return data[100]
IndexError: list index out of range
        """
        faults = self.localizer.localize_fault("code", trace, language="python")
        self.assertEqual(len(faults), 3)
        self.assertEqual(faults[0]["file"], "utils.py")
        self.assertEqual(faults[0]["line_no"], 25)
        self.assertGreater(faults[0]["suspiciousness"], faults[1]["suspiciousness"])

    def test_rust_trace_parsing(self):
        trace = """
thread 'main' panicked at src/main.rs:10:5:
index out of bounds: the len is 5 but the index is 10
stack backtrace:
   0: std::panicking::begin_panic
   1: main::main
             at src/main.rs:10:5
        """
        faults = self.localizer.localize_fault("code", trace, language="rust")
        self.assertEqual(len(faults), 2)
        self.assertEqual(faults[0]["file"], "src/main.rs")
        self.assertEqual(faults[0]["line_no"], 10)

    def test_c_trace_parsing(self):
        trace = """
#0  0x0000555555555139 in crash () at crash.c:5
#1  0x000055555555515e in main () at crash.c:10
        """
        faults = self.localizer.localize_fault("code", trace, language="c")
        self.assertEqual(len(faults), 2)
        self.assertEqual(faults[0]["file"], "crash.c")
        self.assertEqual(faults[0]["line_no"], 5)

    def test_patch_validation(self):
        valid_python = "def foo(): pass"
        invalid_python = "def foo(:"
        self.assertTrue(self.generator._validate_patch(valid_python, "python"))
        self.assertFalse(self.generator._validate_patch(invalid_python, "python"))

    def test_repair_reward(self):
        results = {
            "status": "success",
            "faults": [{"line_no": 25}, {"line_no": 10}]
        }
        # Exact match at rank 0
        reward = compute_repair_reward(results, test_pass_rate=1.0, ground_truth_fault_line=25)
        self.assertEqual(reward, 0.7 * 1.0 + 0.3 * 1.0)

        # Match at rank 1
        reward = compute_repair_reward(results, test_pass_rate=0.5, ground_truth_fault_line=10)
        self.assertEqual(reward, 0.7 * 0.5 + 0.3 * 0.7)

if __name__ == "__main__":
    unittest.main()
