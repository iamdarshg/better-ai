"""Utility functions for DeepSeek model"""

import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from typing import Optional, Union
from contextlib import contextmanager


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For FP8 reproducibility (may impact performance slightly)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Get the appropriate device for training/inference"""
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_memory_usage(device: Optional[torch.device] = None) -> dict:
    """Get current memory usage stats"""
    if device is None:
        device = get_device()
    
    stats = {}
    
    if device.type == "cuda":
        stats["allocated"] = torch.cuda.memory_allocated(device) / 1024**3  # GB
        stats["cached"] = torch.cuda.memory_reserved(device) / 1024**3  # GB
        stats["max_allocated"] = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        stats["max_cached"] = torch.cuda.max_memory_reserved(device) / 1024**3  # GB
        
        # GPU utilization if nvidia-ml-py is available
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats["gpu_utilization"] = gpu_util.gpu
            stats["memory_utilization"] = gpu_util.memory
        except:
            pass
    
    return stats


@contextmanager
def profile_memory(enable: bool = True):
    """Context manager for memory profiling"""
    if not enable:
        yield
        return
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_memory = torch.cuda.memory_allocated()
    
    try:
        yield
    finally:
        end_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        print(f"Memory usage: {start_memory/1024**3:.2f}GB → {end_memory/1024**3:.2f}GB")
        print(f"Peak memory: {peak_memory/1024**3:.2f}GB")


def is_distributed() -> bool:
    """Check if distributed training is active"""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank"""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes"""
    if is_distributed():
        return dist.get_world_size()
    return 1


def synchronize():
    """Synchronize all processes in distributed training"""
    if is_distributed():
        dist.barrier()


def get_tensor_parallel_rank() -> int:
    """Get tensor parallel rank (for future multi-GPU support)"""
    return get_rank()


def get_tensor_parallel_world_size() -> int:
    """Get tensor parallel world size"""
    return get_world_size()


def pad_to_multiple(tensor: torch.Tensor, multiple: int, dim: int = -1, value: float = 0.0) -> torch.Tensor:
    """Pad tensor to be multiple of given number"""
    size = tensor.size(dim)
    remainder = size % multiple
    if remainder == 0:
        return tensor
    
    pad_size = multiple - remainder
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_size
    
    padding = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=dim)


def round_up_to_multiple(n: int, multiple: int) -> int:
    """Round up n to be multiple of given number"""
    return ((n + multiple - 1) // multiple) * multiple


def get_block_size(d_model: int, d_head: int) -> int:
    """Get optimal block size for attention computation"""
    # For RTX 4060, optimal block sizes are multiples of 16
    base_block_size = 128
    
    # Ensure block size is multiple of tensor core requirements
    return round_up_to_multiple(base_block_size, 16)


def check_fp8_availability() -> bool:
    """Check if FP8 is available on current hardware"""
    if not torch.cuda.is_available():
        return False
    
    # Check CUDA version (FP8 requires CUDA 11.8+)
    major, minor = torch.version.cuda.split('.')[:2]
    if int(major) < 11 or (int(major) == 11 and int(minor) < 8):
        return False
    
    # Check if device supports FP8 (Ada Lovelace and newer)
    device_capability = torch.cuda.get_device_capability()
    major_sm, minor_sm = device_capability
    
    # FP8 support starts from compute capability 8.9 (Ada Lovelace)
    return (major_sm > 8) or (major_sm == 8 and minor_sm >= 9)


def get_available_dtypes(fp8_enabled: bool = True) -> tuple:
    """Get available data types based on hardware capabilities"""
    if fp8_enabled and check_fp8_availability():
        return torch.float8_e4m3fn, torch.bfloat16, torch.float16, torch.float32
    else:
        return torch.bfloat16, torch.float16, torch.float32


def get_model_size(model: torch.nn.Module) -> dict:
    """Get model size statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate memory usage
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': total_size / 1024**2,
        'model_size_gb': total_size / 1024**3
    }