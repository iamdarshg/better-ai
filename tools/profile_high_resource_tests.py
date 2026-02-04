#!/usr/bin/env python3
"""
Profile High-Resource Tests and generate flame graphs using py-spy.

This script discovers tests marked as high-resource via the _high_resource flag
on their test classes and profiles them one by one. Flame graphs are stored under
tests/profiles/high_resource/<test_id>.svg
"""

import argparse
import os
import subprocess
import sys
import unittest


def iter_tests(suite):
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            yield from iter_tests(t)
        else:
            yield t


def discover_high_resource_tests():
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    high_ids = []
    for test in iter_tests(suite):
        cls = getattr(test, "__class__", None)
        if cls is not None and getattr(cls, "_is_high_resource", False):
            high_ids.append(test.id())
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for t in high_ids:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def discover_profilable_tests():
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    mapping = {}
    for test in iter_tests(suite):
        cls = getattr(test, "__class__", None)
        if cls is None:
            continue
        tid = test.id()
        if getattr(cls, "_is_high_resource", False):
            mapping[tid] = "high"
        elif getattr(cls, "_is_low_resource", False):
            mapping[tid] = "low"
    return mapping


def profile_test(test_id: str, output_svg: str):
    os.makedirs(os.path.dirname(output_svg), exist_ok=True)
    cmd = [
        "py-spy",
        "record",
        "-o",
        output_svg,
        "--",
        sys.executable,
        "-m",
        "unittest",
        test_id,
    ]
    print(f"Profiling {test_id} -> {output_svg}")
    subprocess.run(cmd, check=True)

    # Optional: also profile with PyTorch profiler for deeper insights
    if os.environ.get("PROFILE_TORCH", "0") == "1":
        torch_wrapper = [
            sys.executable,
            "tools/run_test_with_torch_profiler.py",
            test_id,
        ]
        print(f"Profiling with PyTorch profiler: {test_id}")
        subprocess.run(torch_wrapper, check=True)


def get_resource_for_test_id(test_id: str):
    module_path = ".".join(test_id.split(".")[:-2])
    class_name = test_id.split(".")[-2]
    try:
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name, None)
        if cls is not None:
            if getattr(cls, "_is_high_resource", False):
                return "high"
            if getattr(cls, "_is_low_resource", False):
                return "low"
    except Exception:
        pass
    return "high"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-id",
        dest="test_id",
        action="append",
        default=None,
        help="Profile a single test id (fully-qualified). Can be used multiple times.",
    )
    parser.add_argument(
        "--single-test",
        dest="single_test",
        default=None,
        help="Profile exactly one test id (fully-qualified).",
    )
    args = parser.parse_args()

    if args.single_test:
        tid = args.single_test
        res = get_resource_for_test_id(tid)
        safe = tid.replace(".", "_")
        output = os.path.join("tests", "profiles", res + "_resource", f"{safe}.svg")
        profile_test(tid, output)
        return
    if args.test_id:
        # Profile only requested tests
        for tid in args.test_id:
            res = get_resource_for_test_id(tid)
            safe = tid.replace(".", "_")
            output = os.path.join("tests", "profiles", res + "_resource", f"{safe}.svg")
            profile_test(tid, output)
        return
    # Otherwise profile all profilable tests (high and low resource)
    profiling_map = discover_profilable_tests()
    if not profiling_map:
        print("No profilable tests detected.")
        return
    for tid, res in profiling_map.items():
        safe = tid.replace(".", "_")
        output = os.path.join("tests", "profiles", res + "_resource", f"{safe}.svg")
        try:
            profile_test(tid, output)
        except subprocess.CalledProcessError as e:
            print(f"Profiling failed for {tid}: {e}")


if __name__ == "__main__":
    main()
