"""
Memory management utilities for edge inference.
"""

import torch
import psutil
import os
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Monitors and optimizes memory usage during inference.
    """
    @staticmethod
    def get_gpu_memory_usage():
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    @staticmethod
    def get_system_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**3

    def optimize_for_device(self, model: torch.nn.Module):
        """
        Applies device-specific memory optimizations.
        """
        gpu_mem = self.get_gpu_memory_usage()
        sys_mem = self.get_system_memory_usage()

        logger.info(f"Current usage: GPU={gpu_mem:.2f}GB, System={sys_mem:.2f}GB")

        if gpu_mem > 0.8 * 8: # Assuming 8GB threshold
            logger.warning("High GPU memory usage, enabling gradient checkpointing for inference.")
            model.gradient_checkpointing_enable()

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model

    @staticmethod
    def profile_memory(func):
        """Decorator to profile memory usage of a function"""
        def wrapper(*args, **kwargs):
            before = psutil.Process(os.getpid()).memory_info().rss
            result = func(*args, **kwargs)
            after = psutil.Process(os.getpid()).memory_info().rss
            print(f"Memory change in {func.__name__}: {(after - before) / 1024**2:.2f} MB")
            return result
        return wrapper
