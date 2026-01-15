"""
Optimizers for DeepSeek V3.2 inspired toy model
"""

from .fp8 import FP8Linear, FP8AdamW, FP8LossScaler, FP8Optimizer, get_fp8_optimizer

__all__ = [
    "FP8Linear",
    "FP8AdamW", 
    "FP8LossScaler",
    "FP8Optimizer",
    "get_fp8_optimizer"
]