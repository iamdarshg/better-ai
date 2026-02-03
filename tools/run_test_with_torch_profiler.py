#!/usr/bin/env python3
"""
Run a single unittest by test_id under PyTorch profiler, saving trace data.
Usage: python tools/run_test_with_torch_profiler.py <test_id>
Where test_id is the fully-qualified unittest id, e.g.
tests.unit.test_memory_optimization.TestMemoryOptimizedModel.test_gradient_checkpointing_integration
"""

import importlib
import sys
import unittest
import os
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler


def parse_test_id(test_id: str):
    parts = test_id.split(".")
    if len(parts) < 3:
        raise ValueError("test_id must be fully qualified: module.Class.method")
    module_path = ".".join(parts[:-2])
    class_name = parts[-2]
    method_name = parts[-1]
    return module_path, class_name, method_name


def main():
    if len(sys.argv) < 2:
        print("No test_id provided.")
        sys.exit(1)
    test_id = sys.argv[1]
    module_path, class_name, method_name = parse_test_id(test_id)

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    test_case = cls(method_name)
    test_case.setUp()
    test_method = getattr(test_case, method_name)

    # Check if test is high or low resource
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    resource_type = (
        "high_resource" if getattr(cls, "_is_high_resource", False) else "low_resource"
    )

    # Prepare profiler trace directory
    safe = test_id.replace(".", "_")
    log_dir = os.path.join("tests", "profiles", resource_type, f"{safe}_torch")
    os.makedirs(log_dir, exist_ok=True)

    # TensorBoard trace handler writes trace events to log_dir
    trace_handler = tensorboard_trace_handler(log_dir, collapse_on_step=True)

    with profile(
        activities=[ProfilerActivity.CPU],
        on_trace_ready=trace_handler,
        record_shapes=True,
    ):
        test_method()

    print(f"Torch profiler trace written to: {log_dir}")


if __name__ == "__main__":
    main()
