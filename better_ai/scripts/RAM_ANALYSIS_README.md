# RAM Analysis Tools

This directory contains tools for analyzing and visualizing memory usage patterns of the Better AI models.

## Tools

### `analyze_ram_usage.py`
Empirically measures VRAM usage for small model variants across different batch sizes and sequence lengths.

**Usage:**
```bash
# Analyze both FP8 and BF16 (default)
python tools/analyze_ram_usage.py

# Analyze only FP8
python tools/analyze_ram_usage.py --precision fp8

# Analyze with custom batch sizes and sequence lengths
python tools/analyze_ram_usage.py --batch-sizes 1 2 --seq-lengths 128 256

# Analyze only specific precision
python tools/analyze_ram_usage.py --precision bf16 fp8 --batch-sizes 1 2 4 --seq-lengths 128 512 1024
```

**Output:**
- Saves measurements to `.ram_analysis.json` in the project root
- Provides real-time progress output during analysis
- Shows peak memory, parameter memory, and overhead for each configuration

**Sample Output:**
```
============================================================
Analyzing FP8 SCALING...
============================================================
  Testing: Batch=1, Seq=128... ✓ Peak: 0.87GB (Params: 6.98MB, Overhead: 879.06MB)
  Testing: Batch=1, Seq=256... ✓ Peak: 0.87GB (Params: 6.98MB, Overhead: 879.19MB)
  ...
```

### `plot_ram_analysis.py` (`better_ai/scripts/plot_ram_analysis.py`)
Visualizes the RAM analysis data with comprehensive charts and summary statistics.

**Usage:**
```bash
# Generate plots and summary
python better_ai/scripts/plot_ram_analysis.py

# Text summary only (no matplotlib required)
python better_ai/scripts/plot_ram_analysis.py --text-only
```

**Features:**
- **Plot 1**: Peak memory scaling by batch size for FP8
- **Plot 2**: Peak memory scaling by batch size for BF16
- **Plot 3**: Memory breakdown (parameters vs overhead) for FP8
- **Plot 4**: Overhead scaling with sequence length for both precisions

**Output:**
- Saves plot to `plots/ram_analysis.png`
- Prints detailed text summary with statistics (min, max, avg)

**Example Output:**
```
================================================================================
RAM USAGE ANALYSIS SUMMARY
================================================================================

FP8 ANALYSIS:
--------------------------------------------------------------------------------
 Batch    Seq    Peak (GB)    Params (MB)   Overhead (MB)
--------------------------------------------------------------------------------
     1    128         0.87           6.98          879.06
     1    256         0.87           6.98          879.19
     ...
     4   1024         0.87           6.98          882.47
--------------------------------------------------------------------------------
                      0.87  (max)         6.98  (max)        886.08  (max)
                      0.84  (min)         6.98  (min)        849.97  (min)
                      0.86  (avg)         6.98  (avg)        872.67  (avg)
```

## Workflow

### 1. Collect Data
```bash
# Run analysis with custom parameters or defaults
python tools/analyze_ram_usage.py --precision fp8 bf16
```

### 2. Visualize Results
```bash
# Generate plots and statistics
python better_ai/scripts/plot_ram_analysis.py

# Or just text summary
python better_ai/scripts/plot_ram_analysis.py --text-only
```

## Data Format

The `.ram_analysis.json` file contains:
```json
{
  "bf16": [
    {
      "peak_bytes": 929087488,
      "param_bytes": 7321796,
      "overhead_bytes": 921765692,
      "batch_size": 1,
      "seq_len": 128
    },
    ...
  ],
  "fp8": [...]
}
```

## Integration with Production Estimates

The `tools/update_readme_estimates.py` script uses the analysis data from `.ram_analysis.json` to extrapolate memory requirements for production model sizes:

1. Measures small model across various configurations
2. Uses scaling factors to estimate production model requirements
3. Updates `README.md` with resource estimates

## Requirements

### Core
- PyTorch with CUDA support
- psutil (for system memory monitoring)

### Optional (for visualization)
- matplotlib (for generating plots)

Install matplotlib:
```bash
pip install matplotlib
```

If matplotlib is not available, `plot_ram_analysis.py` will automatically fall back to text-only output.

## Tips

### Quick Analysis
For faster iteration, analyze fewer configurations:
```bash
python tools/analyze_ram_usage.py --precision fp8 --batch-sizes 1 2 --seq-lengths 128 1024
```

### Memory Profiling
The `measure_memory_footprint()` function can be imported and used in your own scripts:
```python
from tools.analyze_ram_usage import measure_memory_footprint
from better_ai.config import ModelConfig

config = ModelConfig.get_small_model_config()
result = measure_memory_footprint(config, batch_size=2, seq_len=512, precision="fp8")
print(f"Peak memory: {result['peak_bytes'] / 1e9:.2f}GB")
```

## Common Issues

### CUDA Out of Memory
If analysis fails with OOM errors:
- Reduce batch sizes
- Reduce sequence lengths
- Ensure no other GPU processes are running

### Matplotlib Not Available
The plotting script works in text-only mode:
```bash
python better_ai/scripts/plot_ram_analysis.py --text-only
```

### No Data Found
If you see "Analysis file not found", run the analysis first:
```bash
python tools/analyze_ram_usage.py
```
