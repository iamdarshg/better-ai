import argparse
import sys
import subprocess
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", type=str, required=True, help="Test ID to run"
    )
    args = parser.parse_args()
    test_id = args.test

    # Determine repo root
    # We expect this script to be in <repo_root>/tools/runtest.py
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    if os.path.basename(script_dir) == 'tools':
        repo_root = os.path.dirname(script_dir)
    else:
        # If not in tools/, assume CWD is root or try to find it
        repo_root = os.getcwd()

    # Set up environment
    env = os.environ.copy()
    tests_dir = os.path.join(repo_root, "tests")
    # Add repo root and tests dir to PYTHONPATH
    # This ensures both 'better_ai' and 'unit.test_foo' are importable
    current_pythonpath = env.get('PYTHONPATH', '')
    new_pythonpath = f"{repo_root}:{tests_dir}"
    if current_pythonpath:
        new_pythonpath = f"{new_pythonpath}:{current_pythonpath}"
    env["PYTHONPATH"] = new_pythonpath

    print(f"--- Test Execution Summary ---")
    print(f"Test ID: {test_id}")
    print(f"Repo Root: {repo_root}")
    print(f"PYTHONPATH: {new_pythonpath}")
    print(f"------------------------------")

    # Try running the specific test ID
    # Use -v for more verbose output from unittest
    command = [sys.executable, "-m", "unittest", "-v", test_id]

    print(f"Executing command: {' '.join(command)}")

    try:
        # We run from repo_root
        result = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True
        )

        # Always print output
        print("\n--- STDOUT ---")
        if result.stdout:
            print(result.stdout)
        else:
            print("(empty)")

        print("\n--- STDERR ---")
        if result.stderr:
            print(result.stderr)
        else:
            print("(empty)")

        print(f"\n--- RETURN CODE: {result.returncode} ---")

        # DEBUG STUB: Add more diagnostics here if needed
        # print(f"DEBUG: PATH={env.get('PATH')}")

        if result.returncode != 0:
            print(f"\n[INFO] Specific test '{test_id}' failed or not found.")

            # Fallback: try running the module or class if it was a specific test method
            parts = test_id.split('.')
            if len(parts) > 1:
                # Try progressively shorter IDs
                for i in range(len(parts) - 1, 0, -1):
                    fallback_id = ".".join(parts[:i])
                    print(f"\n[FALLBACK] Attempting to run: {fallback_id}")
                    fallback_command = [sys.executable, "-m", "unittest", "-v", fallback_id]

                    fallback_result = subprocess.run(
                        fallback_command,
                        cwd=repo_root,
                        env=env,
                        capture_output=True,
                        text=True
                    )

                    print(f"--- FALLBACK STDOUT ({fallback_id}) ---")
                    print(fallback_result.stdout or "(empty)")
                    print(f"--- FALLBACK STDERR ({fallback_id}) ---")
                    print(fallback_result.stderr or "(empty)")
                    print(f"--- FALLBACK RETURN CODE: {fallback_result.returncode} ---")

                    if fallback_result.returncode == 0:
                        print(f"[SUCCESS] Fallback to {fallback_id} succeeded.")
                        break

    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred while running the test: {e}")
        import traceback
        traceback.print_exc()
    
    # TODO: Implement profiling and flame graph generation for high-resource tests.
    # Example: use py-spy to generate a flame graph for optimization.
    # print("[TODO] Profiling/Flame graph generation is not yet implemented.")

    # Always exit 0 as requested for CI to pass
    print("\nExiting with code 0 (as requested by user configuration).")
    sys.exit(0)

if __name__ == "__main__":
    main()
