#!/usr/bin/env python3
"""
Quick test of the plotting setup and data verification.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scripts.plot_ram_analysis import load_analysis_data, bytes_to_gb

# Verify data can be loaded
print("\n" + "="*60)
print("RAM Analysis Data Verification")
print("="*60 + "\n")

data = load_analysis_data()

for precision in ["fp8", "bf16"]:
    entries = data.get(precision, [])
    if not entries:
        print(f"{precision.upper()}: No data")
        continue
    
    print(f"\n{precision.upper()} ({len(entries)} measurements):")
    print("-" * 60)
    
    # Find ranges
    peaks = [bytes_to_gb(d["peak_bytes"]) for d in entries]
    batch_sizes = sorted(set(d["batch_size"] for d in entries))
    seq_lens = sorted(set(d["seq_len"] for d in entries))
    
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Seq lengths: {seq_lens}")
    print(f"  Peak memory range: {min(peaks):.2f}GB - {max(peaks):.2f}GB")
    print(f"  Avg peak memory: {sum(peaks)/len(peaks):.2f}GB")

print("\n" + "="*60)
print("Data ready for analysis!")
print("="*60 + "\n")
