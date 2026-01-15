"""
Better AI - DeepSeek V3.2 Inspired Toy Model
Ultra-optimized FP8 training & inference on RTX 4060 Laptop GPU
"""

__version__ = "0.1.1"
__author__ = "Better AI Team"

from .models import DeepSeekModel, DeepSeekMoEModel
from .config import ModelConfig, TrainingConfig, InferenceConfig
from .training import Trainer
from .inference import InferenceEngine, TextGenerator
from .optimizers import FP8AdamW, FP8Optimizer
from .utils import set_seed, get_device, get_memory_usage, get_model_size
from .data import CodeDataset, create_code_dataloaders

__all__ = [
    "DeepSeekModel",
    "DeepSeekMoEModel", 
    "ModelConfig",
    "TrainingConfig",
    "InferenceConfig",
    "Trainer",
    "InferenceEngine",
    "TextGenerator",
    "FP8AdamW",
    "FP8Optimizer",
    "set_seed",
    "get_device", 
    "get_memory_usage",
    "get_model_size",
    "CodeDataset",
    "create_code_dataloaders"
]