import argparse
import sys
import subprocess
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", type=str, required=True, help="Test ID to dispatch"
    )
    args = parser.parse_args()
    test = args.test
    testName = test.split(".")[-2]
    testDir = test.rsplit(".", 1)[0]
    command = f"python -m cProfile -o profiles/{testName}_profile.prof -m unittest {test}.py"
    try:
        o = subprocess.run(command, shell=True, check=True)
        print(f"Successfully dispatched {test} for profiling.")
    except Exception as e:
        print(f"Failed to dispatch {test} for profiling: {e}")
        sys.exit(0)
    print(f"Dispatched profiling for {test}. Subprocess exited with code {o.returncode}.")
    if o.returncode != 0:
        print(f"Check for errors in profiling {test}.")
        with open("profiles/dispatch_errors.log", "a") as f:
            f.write(f"Failed to dispatch {test} for profiling with exit code {o.returncode}.\n")
            f.write(f"Command: {command}\n")
            f.write(o.stderr.decode() if o.stderr else "No stderr output.\n")
            f.write(o.stdout.decode() if o.stdout else "No stdout output.\n")

if __name__ == "__main__":
    main()