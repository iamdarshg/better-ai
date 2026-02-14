
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
        # IndexError case
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

    def test_python_more_crashes(self):
        # ZeroDivisionError
        trace = """
Traceback (most recent call last):
  File "calc.py", line 42, in divide
    return a / b
ZeroDivisionError: division by zero
        """
        faults = self.localizer.localize_fault("code", trace, language="python")
        self.assertEqual(faults[0]["file"], "calc.py")
        self.assertEqual(faults[0]["line_no"], 42)

        # KeyError
        trace = """
Traceback (most recent call last):
  File "config.py", line 15, in get_val
    return self.settings[key]
KeyError: 'missing_key'
        """
        faults = self.localizer.localize_fault("code", trace, language="python")
        self.assertEqual(faults[0]["file"], "config.py")
        self.assertEqual(faults[0]["line_no"], 15)

    def test_unintended_behavior_localization(self):
        # Logic error (no crash, but failed assertion or unexpected value)
        # We simulate this by providing a manual "error" description instead of a trace
        error_desc = "Expected list to have 5 items, but found 0 at processor.py:88"
        faults = self.localizer.localize_fault("code", error_desc, language="python")
        # The localizer should still try to extract line info if it looks like a file/line
        self.assertTrue(any(f["file"] == "processor.py" and f["line_no"] == 88 for f in faults))

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
        # Segfault case (GDB style)
        trace = """
Program received signal SIGSEGV, Segmentation fault.
0x0000555555555139 in crash () at crash.c:5
#0  0x0000555555555139 in crash () at crash.c:5
#1  0x000055555555515e in main () at crash.c:10
        """
        faults = self.localizer.localize_fault("code", trace, language="c")
        self.assertEqual(len(faults), 3)
        self.assertEqual(faults[0]["file"], "crash.c")
        self.assertEqual(faults[0]["line_no"], 5)

        # AddressSanitizer style
        asan_trace = """
==12345==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000010
    #0 0x7f1234567890 in free_func common.c:12
    #1 0x7f1234567891 in main main.c:20
        """
        faults = self.localizer.localize_fault("code", asan_trace, language="c")
        self.assertEqual(faults[0]["file"], "common.c")
        self.assertEqual(faults[0]["line_no"], 12)

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
