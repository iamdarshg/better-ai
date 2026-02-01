#!/usr/bin/env python3
"""Triggers repository_dispatch events for profiling high/low resource tests.

This script discovers tests annotated with @high_resource (and optionally @low_resource)
and triggers GitHub Actions per-test profiling via the repository_dispatch API.

- It requires a GitHub token with repo scope to dispatch events. In CI, this can be provided
  via the GITHUB_TOKEN environment variable.
- It targets the current repo (uses GITHUB_REPOSITORY env var).
- Usage:
  python tools/trigger_profile_dispatch.py --kind high
  python tools/trigger_profile_dispatch.py --kind low
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def run_dispatch(test_id: str, event_type: str) -> None:
    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not repo or not token:
        print(
            "[WARN] Missing GITHUB_REPOSITORY or GITHUB_TOKEN; skipping dispatch for",
            test_id,
        )
        return
    url = f"https://api.github.com/repos/{repo}/dispatches"
    payload = {
        "event_type": event_type,
        "client_payload": {"test_id": test_id},
    }
    import json
    import urllib.request

    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")
    try:
        with urllib.request.urlopen(req) as resp:
            resp.read()
        print(f"Dispatched {event_type} for {test_id}")
    except Exception as e:
        print(f"Failed to dispatch {event_type} for {test_id}: {e}")


def discover_high_resource_ids() -> list:
    # Reuse the existing listing utility to gather high-resource tests
    import importlib
    import unittest

    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    ids = []
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            for sub in t:
                cls = getattr(sub, "__class__", None)
                if cls is not None and getattr(cls, "_high_resource", False):
                    ids.append(sub.id())
        else:
            cls = getattr(t, "__class__", None)
            if cls is not None and getattr(cls, "_high_resource", False):
                ids.append(t.id())
    return sorted(set(ids))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=["high", "low"],
        default="high",
        help="Resource kind to dispatch",
    )
    args = parser.parse_args()

    if args.kind == "high":
        test_ids = discover_high_resource_ids()
        for tid in test_ids:
            run_dispatch(tid, "profile_high_resource")
    else:
        # For now, discover low-resource tests similarly (if tagging exists)
        test_ids = []
        # Attempt to discover low-resource tests by scanning classes
        import unittest

        loader = unittest.TestLoader()
        suite = loader.discover("tests", pattern="test_*.py")
        for t in suite:
            cls = getattr(t, "__class__", None)
            if cls is not None and getattr(cls, "_low_resource", False):
                test_ids.append(t.id())
        for tid in test_ids:
            run_dispatch(tid, "profile_low_resource")


if __name__ == "__main__":
    main()
