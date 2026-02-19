
import torch
import os
import gc

def measure_baseline():
    if not torch.cuda.is_available():
        return 0

    torch.cuda.empty_cache()
    gc.collect()

    # Just loading torch and cuda context
    baseline = torch.cuda.memory_allocated()
    # Peak memory might be more indicative of the 'constant' overhead
    torch.cuda.reset_peak_memory_stats()

    # Create a tiny tensor to ensure context is fully initialized
    x = torch.zeros(1, device='cuda')
    peak = torch.cuda.max_memory_reserved()

    return peak

if __name__ == "__main__":
    overhead = measure_baseline()
    print(overhead)
