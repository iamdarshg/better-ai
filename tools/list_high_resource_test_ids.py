#!/usr/bin/env python3
"""Discover high-resource tests and emit a JSON list of test IDs."""

import json
import unittest
import sys
from typing import List


def iter_tests(suite: unittest.TestSuite) -> List[unittest.TestCase]:
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            yield from iter_tests(t)
        else:
            yield t


def discover_high_resource_tests() -> List[str]:
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    ids: List[str] = []
    for t in iter_tests(suite):
        cls = getattr(t, "__class__", None)
        if cls is not None and getattr(cls, "_high_resource", False):
            ids.append(t.id())
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for tid in ids:
        if tid not in seen:
            seen.add(tid)
            uniq.append(tid)
    return uniq


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=".ci/high_resource_ids.json",
        help="Output path for JSON list of test IDs",
    )
    args = parser.parse_args()

    test_ids = discover_high_resource_tests()
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(test_ids, f)


if __name__ == "__main__":
    main()
