#!/usr/bin/env python3
"""
Plot RAM usage analysis data from .ram_analysis.json

Visualizes memory scaling across different batch sizes and sequence lengths
for both fp8 and bf16 precisions.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


def load_analysis_data() -> Dict[str, List[Dict[str, Any]]]:
    """Load RAM analysis data from .ram_analysis.json"""
    analysis_path = Path(__file__).parent.parent / ".ram_analysis.json"
    
    if not analysis_path.exists():
        print(f"Error: Analysis file not found at {analysis_path}")
        print("Run: python tools/analyze_ram_usage.py")
        sys.exit(1)
    
    with open(analysis_path, 'r') as f:
        return json.load(f)


def bytes_to_gb(bytes_val: float) -> float:
    """Convert bytes to GB"""
    return bytes_val / (1024 ** 3)


def plot_memory_scaling():
    """Plot memory scaling by batch size and sequence length"""
    if not MATPLOTLIB_AVAILABLE:
        print_text_summary()
        return
    
    data = load_analysis_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RAM Usage Analysis: FP8 vs BF16', fontsize=16, fontweight='bold')
    
    # Plot 1: Peak memory by batch size and seq len (FP8)
    ax = axes[0, 0]
    for precision, data_list in [("fp8", data.get("fp8", []))]:
        if not data_list:
            continue
        
        batch_sizes = sorted(set(d["batch_size"] for d in data_list))
        seq_lens = sorted(set(d["seq_len"] for d in data_list))
        
        for seq_len in seq_lens:
            points = [(d["batch_size"], bytes_to_gb(d["peak_bytes"])) 
                     for d in data_list if d["seq_len"] == seq_len]
            if points:
                points.sort(key=lambda x: x[0])
                batch, peak = zip(*points)
                ax.plot(batch, peak, marker='o', label=f'Seq={seq_len}', linewidth=2)
    
    ax.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Peak Memory (GB)', fontsize=11, fontweight='bold')
    ax.set_title('FP8: Peak Memory by Batch Size', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Peak memory by batch size and seq len (BF16)
    ax = axes[0, 1]
    for precision, data_list in [("bf16", data.get("bf16", []))]:
        if not data_list:
            ax.text(0.5, 0.5, 'No BF16 data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('BF16: Peak Memory by Batch Size', fontsize=12, fontweight='bold')
            continue
        
        batch_sizes = sorted(set(d["batch_size"] for d in data_list))
        seq_lens = sorted(set(d["seq_len"] for d in data_list))
        
        for seq_len in seq_lens:
            points = [(d["batch_size"], bytes_to_gb(d["peak_bytes"])) 
                     for d in data_list if d["seq_len"] == seq_len]
            if points:
                points.sort(key=lambda x: x[0])
                batch, peak = zip(*points)
                ax.plot(batch, peak, marker='s', label=f'Seq={seq_len}', linewidth=2)
    
    ax.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Peak Memory (GB)', fontsize=11, fontweight='bold')
    ax.set_title('BF16: Peak Memory by Batch Size', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Memory breakdown (param vs overhead) for FP8
    ax = axes[1, 0]
    fp8_data = data.get("fp8", [])
    if fp8_data:
        x_labels = [f"B{d['batch_size']}\nS{d['seq_len']}" for d in fp8_data[:6]]
        param_mem = [bytes_to_gb(d["param_bytes"]) for d in fp8_data[:6]]
        overhead_mem = [bytes_to_gb(d["overhead_bytes"]) for d in fp8_data[:6]]
        
        x_pos = np.arange(len(x_labels))
        ax.bar(x_pos, param_mem, label='Parameters', color='steelblue')
        ax.bar(x_pos, overhead_mem, bottom=param_mem, label='Overhead', color='coral')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel('Memory (GB)', fontsize=11, fontweight='bold')
        ax.set_title('FP8: Memory Breakdown', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
    
    # Plot 4: Overhead scaling with sequence length
    ax = axes[1, 1]
    for precision in ["fp8", "bf16"]:
        data_list = data.get(precision, [])
        if not data_list:
            continue
        
        # Group by batch size
        for batch_size in sorted(set(d["batch_size"] for d in data_list)):
            points = [(d["seq_len"], bytes_to_gb(d["overhead_bytes"])) 
                     for d in data_list if d["batch_size"] == batch_size]
            if points:
                points.sort(key=lambda x: x[0])
                seq, overhead = zip(*points)
                marker = 'o' if precision == "fp8" else 's'
                ax.plot(seq, overhead, marker=marker, label=f'{precision.upper()} B{batch_size}', linewidth=2)
    
    ax.set_xlabel('Sequence Length', fontsize=11, fontweight='bold')
    ax.set_ylabel('Overhead Memory (GB)', fontsize=11, fontweight='bold')
    ax.set_title('Overhead Scaling with Sequence Length', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent.parent.parent / "plots" / "ram_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")
    
    plt.show()


def print_text_summary():
    """Print text summary of analysis data"""
    data = load_analysis_data()
    
    print("\n" + "=" * 80)
    print("RAM USAGE ANALYSIS SUMMARY")
    print("=" * 80)
    
    for precision in ["fp8", "bf16"]:
        data_list = data.get(precision, [])
        if not data_list:
            print(f"\n{precision.upper()}: No data")
            continue
        
        print(f"\n{precision.upper()} ANALYSIS:")
        print("-" * 80)
        print(f"{'Batch':>6} {'Seq':>6} {'Peak (GB)':>12} {'Params (MB)':>14} {'Overhead (MB)':>15}")
        print("-" * 80)
        
        for entry in data_list:
            batch = entry["batch_size"]
            seq = entry["seq_len"]
            peak = bytes_to_gb(entry["peak_bytes"])
            param = entry["param_bytes"] / (1024 ** 2)
            overhead = entry["overhead_bytes"] / (1024 ** 2)
            print(f"{batch:>6} {seq:>6} {peak:>12.2f} {param:>14.2f} {overhead:>15.2f}")
        
        # Summary statistics
        peaks = [bytes_to_gb(d["peak_bytes"]) for d in data_list]
        overheads = [d["overhead_bytes"] / (1024 ** 2) for d in data_list]
        params = [d["param_bytes"] / (1024 ** 2) for d in data_list]
        
        print("-" * 80)
        print(f"{'':>6} {'':>6} {max(peaks):>12.2f}  (max) {max(params):>12.2f}  (max) {max(overheads):>13.2f}  (max)")
        print(f"{'':>6} {'':>6} {min(peaks):>12.2f}  (min) {min(params):>12.2f}  (min) {min(overheads):>13.2f}  (min)")
        print(f"{'':>6} {'':>6} {np.mean(peaks):>12.2f}  (avg) {np.mean(params):>12.2f}  (avg) {np.mean(overheads):>13.2f}  (avg)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot RAM usage analysis")
    parser.add_argument("--text-only", action="store_true", help="Print text summary only")
    args = parser.parse_args()
    
    if args.text_only or not MATPLOTLIB_AVAILABLE:
        print_text_summary()
    else:
        plot_memory_scaling()
        print_text_summary()
