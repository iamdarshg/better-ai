"""
Cosine Curriculum Learning for RLHF Training
Implements smooth difficulty progression using cosine scheduling
"""

import math
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, field

from ..config import TrainingConfig


@dataclass
class CurriculumConfig:
    """Configuration for cosine curriculum learning"""

    # Curriculum scheduling
    total_steps: int = 10000
    warmup_steps: int = 1000
    cooldown_steps: int = 1000

    # Difficulty bounds
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0

    # Curriculum stages
    stages: List[str] = field(
        default_factory=lambda: ["pretraining", "sft", "rlhf", "advanced"]
    )
    stage_boundaries: List[float] = field(
        default_factory=lambda: [0.0, 0.3, 0.6, 0.8, 1.0]
    )

    # Data mixing
    dataset_weights: Dict[str, float] = field(default_factory=dict)
    difficulty_aware_sampling: bool = True

    # Curriculum metrics
    track_performance: bool = True
    adaptive_adjustment: bool = True
    performance_window: int = 100


class CosineCurriculumScheduler:
    """
    Cosine-based curriculum scheduler for progressive difficulty
    """

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_step = 0
        self.current_stage_idx = 0

        # Performance tracking
        self.performance_history = []
        self.stage_performance = {stage: [] for stage in config.stages}

        # Difficulty progression
        self.difficulty_history = []
        self.current_difficulty = config.min_difficulty

        logging.info(f"Initialized Cosine Curriculum with {len(config.stages)} stages")

    def step(self) -> Dict[str, Any]:
        """Advance one step and return curriculum state"""
        self.current_step += 1

        # Calculate cosine-based difficulty
        progress = self._get_progress()
        self.current_difficulty = self._calculate_cosine_difficulty(progress)

        # Update stage if needed
        new_stage_idx = self._get_current_stage(progress)
        stage_changed = new_stage_idx != self.current_stage_idx
        if stage_changed:
            self.current_stage_idx = new_stage_idx
            logging.info(f"Curriculum advanced to stage: {self.get_current_stage()}")

        curriculum_state = {
            "step": self.current_step,
            "progress": progress,
            "difficulty": self.current_difficulty,
            "stage": self.get_current_stage(),
            "stage_idx": self.current_stage_idx,
            "stage_changed": stage_changed,
            "dataset_weights": self._get_dataset_weights(),
            "sampling_params": self._get_sampling_params(),
        }

        self.difficulty_history.append(self.current_difficulty)
        return curriculum_state

    def _get_progress(self) -> float:
        """Calculate normalized progress through curriculum"""
        if self.current_step <= self.config.warmup_steps:
            return 0.0
        elif self.current_step >= self.config.total_steps - self.config.cooldown_steps:
            return 1.0
        else:
            effective_steps = self.current_step - self.config.warmup_steps
            effective_total = (
                self.config.total_steps
                - self.config.warmup_steps
                - self.config.cooldown_steps
            )
            return effective_steps / max(1, effective_total)

    def _calculate_cosine_difficulty(self, progress: float) -> float:
        """Calculate difficulty using cosine annealing"""
        # Cosine annealing: start from min_difficulty, smoothly progress to max_difficulty
        cosine_factor = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        difficulty = (
            self.config.min_difficulty * cosine_factor
            + self.config.max_difficulty * (1 - cosine_factor)
        )
        return difficulty

    def _get_current_stage(self, progress: float) -> int:
        """Determine current curriculum stage based on progress"""
        for i, boundary in enumerate(self.config.stage_boundaries[:-1]):
            if progress <= self.config.stage_boundaries[i + 1]:
                return min(i, len(self.config.stages) - 1)
        return len(self.config.stages) - 1

    def _get_dataset_weights(self) -> Dict[str, float]:
        """Get dataset weights based on current stage and difficulty"""
        stage_name = self.get_current_stage()
        base_weights = self.config.dataset_weights.get(stage_name, {})

        if not isinstance(base_weights, dict) or not base_weights:
            # Default weights based on stage
            if stage_name == "pretraining":
                return {"the-stack-v2-dedup": 1.0}
            elif stage_name == "sft":
                return {
                    "OpenMathInstruct": 0.4,
                    "opencodeinstruct": 0.3,
                    "jupyter-agent-dataset": 0.3,
                }
            elif stage_name == "rlhf":
                return {
                    "agentic-dpo": 0.3,
                    "ultrafeedback-binarized": 0.4,
                    "helpsteer-code": 0.3,
                }
            else:  # advanced
                return {
                    "OpenMathInstruct": 0.2,
                    "opencodeinstruct": 0.2,
                    "agentic-dpo": 0.3,
                    "ultrafeedback-binarized": 0.3,
                }

        # Apply difficulty-based weighting
        if self.config.difficulty_aware_sampling:
            difficulty_factor = self.current_difficulty
            return {
                k: v * (1 + difficulty_factor * 0.5) for k, v in base_weights.items()
            }

        return base_weights

    def _get_sampling_params(self) -> Dict[str, Any]:
        """Get sampling parameters based on curriculum state"""
        stage_name = self.get_current_stage()
        difficulty = self.current_difficulty

        # Base parameters
        base_params = {
            "temperature": max(0.1, 1.0 - difficulty * 0.5),
            "top_p": max(0.5, 0.95 - difficulty * 0.2),
            "top_k": max(10, 50 - int(difficulty * 30)),
            "repetition_penalty": 1.0 + difficulty * 0.2,
        }

        # Stage-specific adjustments
        if stage_name == "pretraining":
            base_params.update(
                {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 50,
                }
            )
        elif stage_name == "sft":
            base_params.update(
                {
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "top_k": 40,
                }
            )
        elif stage_name == "rlhf":
            base_params.update(
                {
                    "temperature": 0.6,
                    "top_p": 0.85,
                    "top_k": 30,
                }
            )
        else:  # advanced
            base_params.update(
                {
                    "temperature": 0.4,
                    "top_p": 0.8,
                    "top_k": 20,
                }
            )

        return base_params

    def get_current_stage(self) -> str:
        """Get current stage name"""
        return self.config.stages[self.current_stage_idx]

    def update_performance(self, metrics: Dict[str, float]):
        """Update performance metrics for adaptive curriculum"""
        if not self.config.track_performance:
            return

        # Store performance
        self.performance_history.append(metrics)

        # Update stage-specific performance
        stage_name = self.get_current_stage()
        self.stage_performance[stage_name].append(metrics)

        # Adaptive adjustment if enabled
        if self.config.adaptive_adjustment:
            self._adaptive_adjustment(metrics)

    def _adaptive_adjustment(self, metrics: Dict[str, float]):
        """Adjust curriculum based on performance"""
        if len(self.performance_history) < self.config.performance_window:
            return

        # Get recent performance
        recent_metrics = self.performance_history[-self.config.performance_window :]
        avg_loss = np.mean([m.get("loss", 0) for m in recent_metrics])
        avg_reward = np.mean([m.get("reward", 0) for m in recent_metrics])

        # Adjust difficulty based on performance
        if avg_loss > 2.0 or avg_reward < 0.1:  # Poor performance
            # Slow down curriculum progression
            self.current_difficulty = max(
                self.config.min_difficulty, self.current_difficulty * 0.95
            )
            logging.info(
                f"Curriculum slowed due to poor performance: difficulty={self.current_difficulty:.3f}"
            )
        elif avg_loss < 0.5 and avg_reward > 0.8:  # Excellent performance
            # Speed up curriculum progression
            self.current_difficulty = min(
                self.config.max_difficulty, self.current_difficulty * 1.05
            )
            logging.info(
                f"Curriculum accelerated due to excellent performance: difficulty={self.current_difficulty:.3f}"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get curriculum statistics"""
        return {
            "current_step": self.current_step,
            "current_stage": self.get_current_stage(),
            "current_difficulty": self.current_difficulty,
            "total_steps": self.config.total_steps,
            "progress": self._get_progress(),
            "stage_performance": {
                stage: {
                    "count": len(perf),
                    "avg_loss": np.mean([p.get("loss", 0) for p in perf])
                    if perf
                    else 0,
                    "avg_reward": np.mean([p.get("reward", 0) for p in perf])
                    if perf
                    else 0,
                }
                for stage, perf in self.stage_performance.items()
            },
            "difficulty_history": self.difficulty_history[-100:],  # Last 100 steps
        }

    def save_state(self, filepath: str):
        """Save curriculum state"""
        state = {
            "config": self.config.__dict__,
            "current_step": self.current_step,
            "current_stage_idx": self.current_stage_idx,
            "current_difficulty": self.current_difficulty,
            "performance_history": self.performance_history[-1000:],  # Last 1000 steps
            "difficulty_history": self.difficulty_history,
        }
        torch.save(state, filepath)
        logging.info(f"Curriculum state saved to {filepath}")

    def load_state(self, filepath: str):
        """Load curriculum state"""
        try:
            state = torch.load(filepath)
            self.current_step = state.get("current_step", 0)
            self.current_stage_idx = state.get("current_stage_idx", 0)
            self.current_difficulty = state.get(
                "current_difficulty", self.config.min_difficulty
            )
            self.performance_history = state.get("performance_history", [])
            self.difficulty_history = state.get("difficulty_history", [])
            logging.info(f"Curriculum state loaded from {filepath}")
        except Exception as e:
            logging.error(f"Failed to load curriculum state: {e}")


def create_cosine_curriculum(config: TrainingConfig) -> CosineCurriculumScheduler:
    """Create cosine curriculum scheduler from training config"""
    curriculum_config = CurriculumConfig(
        total_steps=config.max_steps,
        warmup_steps=config.warmup_steps,
        min_difficulty=0.0,
        max_difficulty=1.0,
        track_performance=True,
        adaptive_adjustment=True,
    )

    return CosineCurriculumScheduler(curriculum_config)
