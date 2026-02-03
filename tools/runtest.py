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
    testName = test.split(".")[1]
    testDir = test.rsplit(".")[0]
    command = f"python -m unittest {testDir}/{testName}.py"
    try:
        o = subprocess.run(command, shell=True, check=True)
        print(f"Successfully dispatched {test} for profiling.")
    except Exception as e:
        print(f"Failed to dispatch {test} for profiling: {e}")
        sys.exit(0)
    print(f"Dispatched profiling for {test}. Subprocess exited with code {o.returncode}.")
    if o.returncode != 0:
        with open("profile_dispatch_errors.log", "a") as log_file:
            log_file.write(f"Test {test} failed with exit code {o.returncode}.\n")
            log_file.write(f"Command: {command}\n\n")
            if o.stderr:
                log_file.write(f"Error Output:\n{o.stderr.decode()}\n\n")
            if o.stdout:
                log_file.write(f"Output:\n{o.stdout.decode()}\n\n")
            
        raise RuntimeError(f"Subprocess for {test} failed with exit code {o.returncode}.")
if __name__ == "__main__":
    main()