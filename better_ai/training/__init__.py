"""
Training infrastructure for DeepSeek model with enhanced MoE optimizations
"""

from .trainer import Trainer
from ..utils import get_model_size

# Enhanced MoE training components
from .expert_manager import ExpertSpecializationManager, MoETrainingMonitor
from .checkpointing import SelectiveCheckpointManager, AdaptiveMemoryManager
from .adaptive_optimizations import DynamicExpertCapacityManager, AdaptiveAttentionSelector
from .coherence_scheduler import CoherenceBasedScheduler
from .tui import MoETrainingTUI
from .enhanced_trainer import EnhancedMoETrainer

# RLHF and advanced training
from .grpo import GRPOTrainer, GRPOLoss
from .evaluation import (
    RLHFEvaluator,
    CodingBenchmarkEvaluator,
    SWEBenchEvaluator,
    MetricsAggregator,
    EvaluationMetrics,
)

__all__ = [
    "Trainer",
    "get_model_size",
    # Enhanced MoE components
    "ExpertSpecializationManager",
    "MoETrainingMonitor", 
    "SelectiveCheckpointManager",
    "AdaptiveMemoryManager",
    "DynamicExpertCapacityManager",
    "AdaptiveAttentionSelector",
    "CoherenceBasedScheduler",
    "MoETrainingTUI",
    "EnhancedMoETrainer",
    # RLHF components
    "GRPOTrainer",
    "GRPOLoss",
    # Evaluation
    "RLHFEvaluator",
    "CodingBenchmarkEvaluator",
    "SWEBenchEvaluator",
    "MetricsAggregator",
    "EvaluationMetrics",
]