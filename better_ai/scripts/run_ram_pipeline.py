#!/usr/bin/env python3
"""
Full RAM analysis pipeline: analyze → plot → summarize

This script runs the complete RAM analysis workflow in one command.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n{'='*70}")
    print(f"→ {description}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print(f"✗ Failed: {description}")
        return False
    
    print(f"✓ Complete: {description}")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run complete RAM analysis pipeline"
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip the analysis step (use existing data)"
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Skip plot generation (text summary only)"
    )
    parser.add_argument(
        "--precision",
        nargs="+",
        choices=["bf16", "fp8"],
        help="Precisions to analyze"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        help="Batch sizes to test"
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        help="Sequence lengths to test"
    )
    
    args = parser.parse_args()
    
    success = True
    
    # Step 1: Run analysis
    if not args.skip_analysis:
        cmd = ["python", "tools/analyze_ram_usage.py"]
        if args.precision:
            cmd.extend(["--precision"] + args.precision)
        if args.batch_sizes:
            cmd.extend(["--batch-sizes"] + [str(b) for b in args.batch_sizes])
        if args.seq_lengths:
            cmd.extend(["--seq-lengths"] + [str(s) for s in args.seq_lengths])
        
        success = run_command(cmd, "Running RAM analysis") and success
    
    # Step 2: Generate plots and summary
    cmd = ["python", "better_ai/scripts/plot_ram_analysis.py"]
    if args.text_only:
        cmd.append("--text-only")
    
    success = run_command(cmd, "Generating plots and summary") and success
    
    # Final result
    print(f"\n{'='*70}")
    if success:
        print("✓ Pipeline complete!")
        analysis_file = Path(__file__).parent.parent / ".ram_analysis.json"
        if args.text_only:
            print(f"  • Analysis data: {analysis_file}")
        else:
            plot_file = Path(__file__).parent.parent / "plots" / "ram_analysis.png"
            print(f"  • Analysis data: {analysis_file}")
            print(f"  • Visualization: {plot_file}")
    else:
        print("✗ Pipeline failed!")
        return 1
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
