#!/usr/bin/env python3
"""Aggregate and run all low-resource tests as a single task.

- Discover tests annotated with @low_resource and run them in a single unittest run.
- Exit non-zero if any test fails to signal CI failure.
"""

import unittest
import sys


def _collect_low_resource_tests():
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    collected = []
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            for tt in t:
                cls = getattr(tt, "__class__", None)
                if cls is not None and getattr(cls, "_low_resource", False):
                    collected.append(tt.id())
        else:
            cls = getattr(t, "__class__", None)
            if cls is not None and getattr(cls, "_low_resource", False):
                collected.append(t.id())
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for tid in collected:
        if tid not in seen:
            seen.add(tid)
            uniq.append(tid)
    return uniq


def main():
    test_ids = _collect_low_resource_tests()
    if not test_ids:
        print("No low-resource tests found. Nothing to run.")
        return 0
    # Build a synthetic unittest discovery for the selected tests
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.TestSuite()
    for tid in test_ids:
        # Load the test by id and add to suite
        try:
            suite.addTest(unittest.defaultTestLoader.loadTestsFromName(tid))
        except Exception:
            print(f"Warning: could not load test {tid}")
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
