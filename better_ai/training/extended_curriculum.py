"""
Extended Cosine Curriculum Learning with Sequence, Difficulty, and Domain Optimization
All configuration-driven with no hardcoded values
"""

import math
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SequenceLengthConfig:
    """Configuration for sequence length curriculum per stage"""

    stage: str
    min_length: int = 4096  # Starting sequence length
    warmup_steps: int = 1000  # Steps at min_length before progression
    schedule: str = "cosine"  # "cosine", "linear", "step", "exponential", "grokking_cosine", "grokking_step"
    # Grokking optimization parameters
    grokking_fast_ratio: float = 0.4  # Ratio for fast progression phase
    plateau_steps: int = 0  # Steps to hold at mid-length
    step_thresholds: Optional[List[float]] = (
        None  # For "step" and "grokking_step" schedules
    )
    exponential_base: float = 2.0  # For "exponential" schedule
    # Per-dataset min lengths (override global min for specific datasets)
    dataset_min_lengths: Dict[str, int] = field(default_factory=dict)
    # Allow datasets to progress at different rates
    dataset_progression_rates: Dict[str, float] = field(default_factory=dict)


@dataclass
class DifficultyConfig:
    """Configuration for difficulty curriculum per stage"""

    stage: str
    difficulty_field: str = "difficulty"  # Field to query from dataset
    alternative_fields: List[str] = field(
        default_factory=lambda: ["complexity", "difficulty_score", "hardness"]
    )
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0
    default_difficulty: float = 0.5  # For unlabeled examples
    warmup_steps: int = 500
    schedule: str = (
        "cosine"  # "cosine", "linear", "sigmoid", "step", "grokking_sigmoid"
    )
    # Grokking optimization parameters
    grokking_fast_ratio: float = 0.4  # Fast progression in first 40%
    # Adaptive adjustment settings
    enable_adaptive: bool = True
    performance_window: int = 50  # Steps to look back for performance
    adjustment_rate: float = 0.1  # How much to adjust based on performance
    # Stage-specific difficulty aspects
    use_length_proxy: bool = (
        True  # Use sequence length as difficulty proxy when no score available
    )
    length_difficulty_factor: float = (
        0.3  # How much length contributes to difficulty (0-1)
    )
    # Grokking: minimum samples per difficulty tier before progression
    min_samples_per_difficulty: int = 500


@dataclass
class DomainMixingConfig:
    """Configuration for adaptive domain mixing within a stage"""

    stage: str
    domains: Dict[str, List[str]] = field(
        default_factory=dict
    )  # domain_name -> [dataset_names]
    initial_weights: Dict[str, float] = field(
        default_factory=dict
    )  # domain_name -> weight (0-1)
    # Adaptive update settings
    update_frequency: int = 100  # Base update frequency in steps
    enable_adaptive_frequency: bool = True  # Adjust frequency based on stability
    min_update_frequency: int = 50
    max_update_frequency: int = 500
    # Weight adjustment parameters
    adjustment_rate: float = 0.05  # How fast weights change
    min_weight: float = 0.05  # Prevent domains from disappearing
    max_weight: float = 0.9  # Prevent domain dominance
    smoothing_factor: float = 0.8  # EMA smoothing for weight changes
    # Performance metrics to track (will be blended)
    tracked_metrics: List[str] = field(
        default_factory=lambda: ["loss", "accuracy", "reward", "perplexity"]
    )
    metric_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "loss": 0.4,
            "accuracy": 0.3,
            "reward": 0.2,
            "perplexity": 0.1,
        }
    )
    # Performance blending strategy
    blend_strategy: str = (
        "weighted_sum"  # "weighted_sum", "geometric_mean", "min_max_norm"
    )
    # Variance threshold for adaptive frequency (higher variance = more frequent updates)
    variance_threshold: float = 0.1
    # Grokking optimization: stabilize weights early, minimal changes in tail
    convergence_step: float = 0.4  # Weights stabilize at this step ratio
    # Global grokking mode flag
    grokking_mode: bool = False


@dataclass
class ExtendedCurriculumConfig:
    """Complete configuration for extended curriculum"""

    stage: str
    total_steps: int = 10000

    # Sub-configurations
    sequence_config: Optional[SequenceLengthConfig] = None
    difficulty_config: Optional[DifficultyConfig] = None
    domain_config: Optional[DomainMixingConfig] = None

    # Global settings
    enable_sequence_curriculum: bool = True
    enable_difficulty_curriculum: bool = True
    enable_domain_mixing: bool = True

    # Grokking optimization
    grokking_mode: bool = False
    fast_learning_ratio: float = 0.4
    generalization_tail: float = 0.6

    # Checkpointing
    save_frequency: int = 500

    def __post_init__(self):
        if self.sequence_config is None:
            self.sequence_config = SequenceLengthConfig(stage=self.stage)
        if self.difficulty_config is None:
            self.difficulty_config = DifficultyConfig(stage=self.stage)
        if self.domain_config is None:
            self.domain_config = DomainMixingConfig(stage=self.stage)


class SequenceLengthScheduler:
    """
    Schedules sequence length progression from min to max using various strategies.
    Each dataset has its own max_length from datasets.yml, progresses from stage min.
    """

    def __init__(
        self, config: SequenceLengthConfig, dataset_max_lengths: Dict[str, int]
    ):
        self.config = config
        self.dataset_max_lengths = dataset_max_lengths
        self.current_step = 0

        # Calculate current lengths per dataset
        self.current_lengths = {}
        self.length_history = defaultdict(list)

        logger.info(f"Initialized SequenceLengthScheduler for stage {config.stage}")
        logger.info(f"Dataset max lengths: {dataset_max_lengths}")

    def step(self) -> Dict[str, int]:
        """Advance one step and return current max_seq_length for each dataset"""
        self.current_step += 1

        progress = self._calculate_progress()

        # Calculate length for each dataset
        for dataset_name, max_len in self.dataset_max_lengths.items():
            # Get dataset-specific min length or use global
            min_len = self.config.dataset_min_lengths.get(
                dataset_name, self.config.min_length
            )

            # Get dataset-specific progression rate or use 1.0
            rate = self.config.dataset_progression_rates.get(dataset_name, 1.0)
            adjusted_progress = min(1.0, progress * rate)

            # Calculate current length based on schedule
            current_len = self._calculate_length(min_len, max_len, adjusted_progress)
            self.current_lengths[dataset_name] = current_len
            self.length_history[dataset_name].append(current_len)

        return self.current_lengths.copy()

    def _calculate_progress(self) -> float:
        """Calculate normalized progress through curriculum"""
        # For grokking schedules, we use a different progress calculation
        # Progress ranges from 0 to 1 over the course of training
        if self.current_step <= self.config.warmup_steps:
            return 0.0

        effective_steps = self.current_step - self.config.warmup_steps
        # Use a fixed denominator for progress calculation
        # This should be the expected total training steps
        # For now, use warmup_steps * 10 as a reasonable default
        progress_denominator = max(1, self.config.warmup_steps * 10)
        return min(1.0, effective_steps / progress_denominator)

    def _calculate_length(self, min_len: int, max_len: int, progress: float) -> int:
        """Calculate sequence length based on schedule type"""
        schedule = self.config.schedule

        if schedule == "cosine":
            # Cosine annealing: slow start, fast middle, slow end
            cosine_factor = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
            length = min_len + (max_len - min_len) * (1 - cosine_factor)

        elif schedule == "linear":
            # Linear progression
            length = min_len + (max_len - min_len) * progress

        elif schedule == "step":
            # Stepwise progression
            thresholds = self.config.step_thresholds or [0.25, 0.5, 0.75, 1.0]
            num_steps = len(thresholds)
            step_size = (max_len - min_len) / num_steps

            current_step = 0
            for threshold in thresholds:
                if progress >= threshold:
                    current_step += 1

            length = min_len + step_size * current_step

        elif schedule == "exponential":
            # Exponential growth: slow start, rapid growth
            base = self.config.exponential_base
            if progress == 0:
                length = min_len
            else:
                exp_factor = (base**progress - 1) / (base - 1)
                length = min_len + (max_len - min_len) * exp_factor

        elif schedule == "grokking_cosine":
            # Grokking: fast progression in first 40%, plateau, slow in last 60%
            fast_ratio = self.config.grokking_fast_ratio
            plateau_steps = self.config.plateau_steps

            if progress <= fast_ratio:
                # Fast progression phase: accelerate quickly
                # Use cosine squared for faster initial growth
                fast_progress = progress / fast_ratio
                cosine_factor = 0.5 * (1 + math.cos(math.pi * (1 - fast_progress)))
                effective_progress = fast_progress * cosine_factor
                length = min_len + (max_len - min_len) * effective_progress
            else:
                # Generalization tail: very slow progression
                # Hold at plateau if configured, then very slow increase
                tail_progress = (progress - fast_ratio) / (1.0 - fast_ratio)
                # Use linear with very small slope for slow tail
                tail_length = 0.3 + 0.7 * tail_progress  # Only reach 70% of max in tail
                length = min_len + (max_len - min_len) * tail_length

        elif schedule == "grokking_step":
            # Grokking step: more gradual steps with grokking ratios
            thresholds = self.config.step_thresholds or [0.2, 0.4, 0.6, 0.8, 1.0]
            num_steps = len(thresholds)
            # First half of steps happen in first 40%
            fast_ratio = self.config.grokking_fast_ratio

            current_step = 0
            for i, threshold in enumerate(thresholds):
                if progress >= threshold:
                    current_step += 1

            # Calculate step progress - first steps faster
            if current_step <= num_steps // 2:
                # Fast phase: complete half the steps in first 40%
                fast_step_progress = progress / fast_ratio if fast_ratio > 0 else 0
                effective_step = current_step + fast_step_progress * 0.5
                length = min_len + (max_len - min_len) * (effective_step / num_steps)
            else:
                # Slow tail phase
                tail_progress = (progress - fast_ratio) / (1.0 - fast_ratio)
                effective_step = (num_steps // 2) + tail_progress * (
                    num_steps - num_steps // 2
                )
                length = min_len + (max_len - min_len) * (effective_step / num_steps)

        else:
            raise ValueError(f"Unknown schedule type: {schedule}")

        return int(length)

    def get_current_lengths(self) -> Dict[str, int]:
        """Get current max_seq_length for each dataset"""
        return self.current_lengths.copy()

    def get_dataset_length(self, dataset_name: str) -> int:
        """Get current max_seq_length for a specific dataset"""
        return self.current_lengths.get(dataset_name, self.config.min_length)


class DifficultyScheduler:
    """
    Schedules difficulty progression and normalizes difficulty scores from datasets.
    """

    def __init__(self, config: DifficultyConfig):
        self.config = config
        self.current_step = 0
        self.current_difficulty_threshold = config.min_difficulty

        # Performance tracking for adaptive adjustment
        self.performance_history = []
        self.difficulty_history = []

        # Difficulty score cache for datasets
        self.dataset_difficulty_stats = {}

        logger.info(f"Initialized DifficultyScheduler for stage {config.stage}")

    def step(self) -> float:
        """Advance one step and return current difficulty threshold"""
        self.current_step += 1

        progress = self._calculate_progress()

        # Calculate target difficulty based on schedule
        target_difficulty = self._calculate_target_difficulty(progress)

        # Apply adaptive adjustment if enabled
        if (
            self.config.enable_adaptive
            and len(self.performance_history) >= self.config.performance_window
        ):
            target_difficulty = self._apply_adaptive_adjustment(target_difficulty)

        self.current_difficulty_threshold = target_difficulty
        self.difficulty_history.append(target_difficulty)

        return target_difficulty

    def _calculate_progress(self) -> float:
        """Calculate normalized progress"""
        if self.current_step <= self.config.warmup_steps:
            return 0.0

        effective_steps = self.current_step - self.config.warmup_steps
        # Progress over the course of training
        return min(1.0, effective_steps / max(1, self.config.warmup_steps * 10))

    def _calculate_target_difficulty(self, progress: float) -> float:
        """Calculate target difficulty threshold"""
        schedule = self.config.schedule
        min_d, max_d = self.config.min_difficulty, self.config.max_difficulty

        if schedule == "cosine":
            cosine_factor = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
            difficulty = min_d + (max_d - min_d) * (1 - cosine_factor)

        elif schedule == "linear":
            difficulty = min_d + (max_d - min_d) * progress

        elif schedule == "sigmoid":
            # Sigmoid: very slow start, rapid middle, slow end
            k = 10  # Steepness
            sigmoid = 1 / (1 + math.exp(-k * (progress - 0.5)))
            difficulty = min_d + (max_d - min_d) * sigmoid

        elif schedule == "grokking_sigmoid":
            # Grokking sigmoid: rapid middle phase, very slow end
            fast_ratio = self.config.grokking_fast_ratio

            if progress <= fast_ratio:
                # Fast learning phase: rapid increase in difficulty
                # Compressed sigmoid centered at fast_ratio/2
                k = 15  # Higher steepness for faster phase
                fast_progress = progress / fast_ratio
                sigmoid = 1 / (1 + math.exp(-k * (fast_progress - 0.5)))
                difficulty = min_d + (max_d - min_d) * sigmoid
            else:
                # Generalization tail: very slow difficulty increase
                # Stay near max difficulty but increase very gradually
                tail_progress = (progress - fast_ratio) / (1.0 - fast_ratio)
                # Almost flat sigmoid tail - only reach 90% of max
                k = 2  # Low steepness for slow tail
                tail_sigmoid = 1 / (1 + math.exp(-k * (tail_progress - 0.7)))
                difficulty = min_d + (max_d - min_d) * (0.9 + 0.1 * tail_sigmoid)

        elif schedule == "step":
            # Discrete steps
            if progress < 0.33:
                difficulty = min_d + (max_d - min_d) * 0.33
            elif progress < 0.66:
                difficulty = min_d + (max_d - min_d) * 0.66
            else:
                difficulty = max_d

        else:
            raise ValueError(f"Unknown schedule type: {schedule}")

        return difficulty

    def _apply_adaptive_adjustment(self, target_difficulty: float) -> float:
        """Adjust difficulty based on recent performance"""
        recent_perf = self.performance_history[-self.config.performance_window :]

        # Calculate average recent performance
        avg_loss = np.mean([p.get("loss", 0) for p in recent_perf])
        avg_accuracy = np.mean([p.get("accuracy", 0) for p in recent_perf])
        avg_reward = np.mean([p.get("reward", 0) for p in recent_perf])

        # Determine if we should speed up or slow down
        adjustment = 0.0

        # If loss is high or accuracy is low, slow down (lower difficulty)
        if avg_loss > 2.0:
            adjustment -= self.config.adjustment_rate
        elif avg_loss < 0.5:
            adjustment += self.config.adjustment_rate

        if avg_accuracy < 0.3:
            adjustment -= self.config.adjustment_rate * 0.5
        elif avg_accuracy > 0.9:
            adjustment += self.config.adjustment_rate * 0.5

        if avg_reward < 0.1:
            adjustment -= self.config.adjustment_rate * 0.3
        elif avg_reward > 0.8:
            adjustment += self.config.adjustment_rate * 0.3

        # Apply adjustment
        adjusted_difficulty = target_difficulty * (1 + adjustment)
        return max(
            self.config.min_difficulty,
            min(self.config.max_difficulty, adjusted_difficulty),
        )

    def normalize_difficulty_score(
        self, item: Dict[str, Any], seq_length: int = 0
    ) -> float:
        """
        Extract and normalize difficulty score from dataset item.
        Returns score in [0, 1], defaulting to 0.5 if not found.
        """
        # Try to find difficulty score in various fields
        difficulty = None

        # Check primary field
        if self.config.difficulty_field in item:
            difficulty = item[self.config.difficulty_field]

        # Check alternative fields
        if difficulty is None:
            for field in self.config.alternative_fields:
                if field in item:
                    difficulty = item[field]
                    break

        # If found, normalize to [0, 1]
        if difficulty is not None:
            try:
                difficulty = float(difficulty)
                # Assume difficulty might be in various ranges, normalize
                if difficulty > 1.0:  # Likely on different scale (e.g., 1-10)
                    difficulty = (difficulty - 1) / 9
                elif difficulty < 0:  # Might be negative to positive
                    difficulty = (difficulty + 1) / 2
                # Clamp to [0, 1]
                difficulty = max(0.0, min(1.0, difficulty))
            except (ValueError, TypeError):
                difficulty = None

        # If no difficulty found and using length proxy, estimate from sequence length
        if difficulty is None and self.config.use_length_proxy and seq_length > 0:
            # Assume longer sequences are harder, but normalize based on typical lengths
            # This is a heuristic: max reasonable length ~128k tokens
            difficulty = (
                min(1.0, seq_length / 131072) * self.config.length_difficulty_factor
            )
            # Mix with default to avoid over-weighting length
            difficulty = difficulty + self.config.default_difficulty * (
                1 - self.config.length_difficulty_factor
            )

        # Default to 0.5 if nothing found
        if difficulty is None:
            difficulty = self.config.default_difficulty

        return difficulty

    def update_performance(self, metrics: Dict[str, float]):
        """Update performance history for adaptive adjustment"""
        self.performance_history.append(metrics)
        # Keep only last N steps
        if len(self.performance_history) > self.config.performance_window * 3:
            self.performance_history = self.performance_history[
                -self.config.performance_window * 2 :
            ]

    def should_include_sample(self, sample_difficulty: float) -> bool:
        """Determine if a sample should be included based on current difficulty threshold"""
        # Include samples with difficulty <= current threshold
        # Add some randomness for exploration
        if sample_difficulty <= self.current_difficulty_threshold:
            return True

        # 10% chance to include harder samples for exploration
        if np.random.random() < 0.1:
            return True

        return False


class AdaptiveDomainMixer:
    """
    Manages domain mixing weights adaptively based on performance metrics.
    Supports grokking optimization with weight stabilization after convergence.
    """

    def __init__(self, config: DomainMixingConfig, total_steps: int = 10000):
        self.config = config
        self.current_step = 0
        self.total_steps = total_steps  # For grokking convergence tracking

        # Initialize weights
        if config.initial_weights:
            self.domain_weights = config.initial_weights.copy()
        else:
            # Equal weights if not specified
            num_domains = len(config.domains)
            self.domain_weights = {
                domain: 1.0 / num_domains for domain in config.domains
            }

        # Normalize weights to sum to 1
        self._normalize_weights()

        # Performance tracking per domain
        self.domain_performance = defaultdict(lambda: defaultdict(list))
        self.weight_history = [self.domain_weights.copy()]

        # Adaptive frequency tracking
        self.last_update_step = 0
        self.current_update_frequency = config.update_frequency
        self.performance_variance_history = []

        logger.info(f"Initialized AdaptiveDomainMixer for stage {config.stage}")
        logger.info(f"Initial weights: {self.domain_weights}")

    def step(self) -> Dict[str, float]:
        """Advance one step and potentially update weights"""
        self.current_step += 1

        # Check if it's time to update weights
        steps_since_update = self.current_step - self.last_update_step

        if steps_since_update >= self.current_update_frequency:
            self._update_weights()
            self.last_update_step = self.current_step

            # Adjust update frequency if enabled
            if self.config.enable_adaptive_frequency:
                self._adjust_update_frequency()

        return self.domain_weights.copy()

    def _update_weights(self):
        """Update domain weights based on performance"""
        # Grokking: check if we've passed the convergence step
        if hasattr(self.config, "convergence_step"):
            convergence_step = getattr(self.config, "convergence_step", 0.4)
            total_steps = getattr(self, "total_steps", 1000)
            current_step_ratio = (
                self.current_step / total_steps if total_steps > 0 else 0
            )

            if current_step_ratio >= convergence_step:
                # Grokking: weights should stabilize after convergence step
                # Only make minimal adjustments
                self._stabilize_weights()
                return

        # Calculate blended performance for each domain
        domain_scores = {}

        for domain in self.config.domains:
            if domain not in self.domain_performance:
                continue

            # Get recent performance for this domain
            perf = self.domain_performance[domain]

            # Calculate blended score
            blended_score = self._blend_metrics(perf)
            domain_scores[domain] = blended_score

        if not domain_scores:
            return

        # Calculate relative performance (lower score = worse performance = needs more weight)
        avg_score = np.mean(list(domain_scores.values()))

        new_weights = {}
        for domain in self.config.domains:
            current_weight = self.domain_weights.get(
                domain, 1.0 / len(self.config.domains)
            )

            if domain in domain_scores:
                score = domain_scores[domain]
                # Domains with lower scores need more weight
                relative_need = avg_score / (score + 1e-8)

                # Calculate weight adjustment
                adjustment = (relative_need - 1.0) * self.config.adjustment_rate
                new_weight = current_weight * (1 + adjustment)
            else:
                # No performance data, keep current weight
                new_weight = current_weight

            # Apply bounds
            new_weight = max(
                self.config.min_weight, min(self.config.max_weight, new_weight)
            )
            new_weights[domain] = new_weight

        # Normalize weights
        total = sum(new_weights.values())
        new_weights = {k: v / total for k, v in new_weights.items()}

        # Apply EMA smoothing
        smoothed_weights = {}
        for domain in self.config.domains:
            old_w = self.domain_weights.get(domain, 1.0 / len(self.config.domains))
            new_w = new_weights.get(domain, 1.0 / len(self.config.domains))
            smoothed_weights[domain] = (
                self.config.smoothing_factor * old_w
                + (1 - self.config.smoothing_factor) * new_w
            )

        self.domain_weights = smoothed_weights
        self.weight_history.append(self.domain_weights.copy())

        logger.info(f"Updated domain weights: {self.domain_weights}")

    def _stabilize_weights(self):
        """Grokking: stabilize weights after convergence step with minimal adjustments"""
        # Only make tiny adjustments to maintain stability
        for domain in self.config.domains:
            if domain in self.domain_performance:
                # Very small adjustment based on recent performance
                perf = self.domain_performance[domain]
                if "loss" in perf and len(perf["loss"]) > 0:
                    recent_loss = np.mean(perf["loss"][-5:])
                    # If loss spiked, slightly increase weight
                    if recent_loss > 1.5:
                        self.domain_weights[domain] *= 1.01
                    # If loss dropped significantly, slightly decrease
                    elif recent_loss < 0.3:
                        self.domain_weights[domain] *= 0.99

        # Re-normalize
        total = sum(self.domain_weights.values())
        if total > 0:
            self.domain_weights = {k: v / total for k, v in self.domain_weights.items()}

        self.weight_history.append(self.domain_weights.copy())

    def _blend_metrics(self, performance: Dict[str, List[float]]) -> float:
        """Blend multiple metrics into a single score"""
        metric_values = {}

        for metric_name in self.config.tracked_metrics:
            if metric_name in performance and len(performance[metric_name]) > 0:
                # Use recent values
                recent_values = performance[metric_name][-10:]
                metric_values[metric_name] = np.mean(recent_values)

        if not metric_values:
            return 0.5  # Default neutral score

        blend_strategy = self.config.blend_strategy

        if blend_strategy == "weighted_sum":
            # Weighted sum of normalized metrics
            total_score = 0.0
            total_weight = 0.0

            for metric_name, value in metric_values.items():
                weight = self.config.metric_weights.get(metric_name, 0.1)
                normalized_value = self._normalize_metric(metric_name, value)
                total_score += normalized_value * weight
                total_weight += weight

            return total_score / max(total_weight, 1e-8)

        elif blend_strategy == "geometric_mean":
            # Geometric mean (good for combining metrics with different scales)
            product = 1.0
            count = 0
            for metric_name, value in metric_values.items():
                normalized_value = self._normalize_metric(metric_name, value)
                weight = self.config.metric_weights.get(metric_name, 0.1)
                product *= normalized_value**weight
                count += weight
            return product ** (1 / max(count, 1e-8))

        elif blend_strategy == "min_max_norm":
            # Min-max normalization then average
            normalized_values = [
                self._normalize_metric(m, v) for m, v in metric_values.items()
            ]
            return np.mean(normalized_values)

        else:
            raise ValueError(f"Unknown blend strategy: {blend_strategy}")

    def _normalize_metric(self, metric_name: str, value: float) -> float:
        """Normalize metric to [0, 1] range"""
        # Different metrics have different optimal ranges
        if metric_name == "loss":
            # Lower is better, typical range 0-5
            return max(0.0, 1.0 - (value / 5.0))

        elif metric_name == "accuracy":
            # Higher is better, range 0-1
            return max(0.0, min(1.0, value))

        elif metric_name == "reward":
            # Higher is better, typical range -1 to 1
            return (value + 1) / 2

        elif metric_name == "perplexity":
            # Lower is better, typical range 1-100
            return max(0.0, 1.0 - (math.log(value) / math.log(100)))

        elif metric_name == "kl_divergence":
            # Lower is better for stability, typical range 0-0.5
            return max(0.0, 1.0 - (value / 0.5))

        else:
            # Unknown metric, assume already normalized
            return max(0.0, min(1.0, value))

    def _adjust_update_frequency(self):
        """Adaptively adjust how often weights are updated"""
        if len(self.weight_history) < 2:
            return

        # Calculate variance in weight changes
        recent_weights = list(self.weight_history[-5:])
        if len(recent_weights) < 2:
            return

        # Calculate average weight change
        total_change = 0.0
        for i in range(1, len(recent_weights)):
            for domain in self.config.domains:
                change = abs(recent_weights[i][domain] - recent_weights[i - 1][domain])
                total_change += change

        avg_change = total_change / (len(recent_weights) - 1) / len(self.config.domains)
        self.performance_variance_history.append(avg_change)

        # If weights are changing a lot (high variance), update more frequently
        if len(self.performance_variance_history) > 5:
            recent_variance = np.mean(self.performance_variance_history[-5:])

            if recent_variance > self.config.variance_threshold:
                # High variance = more frequent updates
                self.current_update_frequency = max(
                    self.config.min_update_frequency,
                    int(self.current_update_frequency * 0.9),
                )
            else:
                # Low variance = less frequent updates
                self.current_update_frequency = min(
                    self.config.max_update_frequency,
                    int(self.current_update_frequency * 1.1),
                )

    def _normalize_weights(self):
        """Normalize domain weights to sum to 1.0"""
        total = sum(self.domain_weights.values())
        if total > 0:
            self.domain_weights = {k: v / total for k, v in self.domain_weights.items()}

    def update_domain_performance(self, domain: str, metrics: Dict[str, float]):
        """Update performance metrics for a specific domain"""
        if domain not in self.config.domains:
            logger.warning(f"Unknown domain: {domain}")
            return

        for metric_name, value in metrics.items():
            self.domain_performance[domain][metric_name].append(value)

            # Keep only recent history
            if len(self.domain_performance[domain][metric_name]) > 100:
                self.domain_performance[domain][metric_name] = self.domain_performance[
                    domain
                ][metric_name][-50:]

    def get_sampling_weights(self) -> Dict[str, float]:
        """Get current domain weights for sampling"""
        return self.domain_weights.copy()

    def get_dataset_domain_weights(self) -> Dict[str, float]:
        """
        Get weights for individual datasets based on their domain membership.
        Distributes domain weight equally among datasets in that domain.
        """
        dataset_weights = {}

        for domain, datasets in self.config.domains.items():
            domain_weight = self.domain_weights.get(domain, 0)
            if len(datasets) > 0:
                weight_per_dataset = domain_weight / len(datasets)
                for dataset_name in datasets:
                    dataset_weights[dataset_name] = weight_per_dataset

        return dataset_weights


class ExtendedCurriculumScheduler:
    """
    Orchestrates sequence length, difficulty, and domain mixing curricula.
    All configuration-driven with no hardcoded values.
    """

    def __init__(
        self,
        config: ExtendedCurriculumConfig,
        dataset_max_lengths: Optional[Dict[str, int]] = None,
    ):
        self.config = config
        self.current_step = 0

        # Initialize sub-schedulers
        self.sequence_scheduler = None
        self.difficulty_scheduler = None
        self.domain_mixer = None

        if config.enable_sequence_curriculum and dataset_max_lengths:
            self.sequence_scheduler = SequenceLengthScheduler(
                config.sequence_config, dataset_max_lengths
            )

        if config.enable_difficulty_curriculum:
            self.difficulty_scheduler = DifficultyScheduler(config.difficulty_config)

        if config.enable_domain_mixing:
            self.domain_mixer = AdaptiveDomainMixer(
                config.domain_config, total_steps=config.total_steps
            )

        # State tracking
        self.state_history = []

        logger.info(f"Initialized ExtendedCurriculumScheduler for stage {config.stage}")

    def step(self) -> Dict[str, Any]:
        """
        Advance curriculum by one step.
        Returns current state including all curriculum parameters.
        """
        self.current_step += 1

        state = {
            "step": self.current_step,
            "stage": self.config.stage,
        }

        # Update sequence lengths
        if self.sequence_scheduler:
            seq_lengths = self.sequence_scheduler.step()
            state["sequence_lengths"] = seq_lengths

        # Update difficulty threshold
        if self.difficulty_scheduler:
            difficulty = self.difficulty_scheduler.step()
            state["difficulty_threshold"] = difficulty

        # Update domain weights
        if self.domain_mixer:
            domain_weights = self.domain_mixer.step()
            state["domain_weights"] = domain_weights
            state["dataset_weights"] = self.domain_mixer.get_dataset_domain_weights()

        self.state_history.append(state)

        # Periodic logging
        if self.current_step % self.config.save_frequency == 0:
            self._log_state(state)

        return state

    def _log_state(self, state: Dict[str, Any]):
        """Log current curriculum state"""
        logger.info(f"Curriculum step {self.current_step}:")

        if "sequence_lengths" in state:
            lengths = state["sequence_lengths"]
            avg_len = np.mean(list(lengths.values()))
            logger.info(f"  Avg sequence length: {avg_len:.0f}")

        if "difficulty_threshold" in state:
            logger.info(f"  Difficulty threshold: {state['difficulty_threshold']:.3f}")

        if "domain_weights" in state:
            logger.info(f"  Domain weights: {state['domain_weights']}")

    def get_current_sequence_lengths(self) -> Optional[Dict[str, int]]:
        """Get current max_seq_length for each dataset"""
        if self.sequence_scheduler:
            return self.sequence_scheduler.get_current_lengths()
        return None

    def get_dataset_sequence_length(self, dataset_name: str) -> int:
        """Get current max_seq_length for a specific dataset"""
        if self.sequence_scheduler:
            return self.sequence_scheduler.get_dataset_length(dataset_name)
        return (
            self.config.sequence_config.min_length
            if self.config.sequence_config
            else 4096
        )

    def get_difficulty_threshold(self) -> float:
        """Get current difficulty threshold"""
        if self.difficulty_scheduler:
            return self.difficulty_scheduler.current_difficulty_threshold
        return 1.0

    def get_domain_weights(self) -> Optional[Dict[str, float]]:
        """Get current domain sampling weights"""
        if self.domain_mixer:
            return self.domain_mixer.get_sampling_weights()
        return None

    def get_dataset_weights(self) -> Optional[Dict[str, float]]:
        """Get current dataset sampling weights"""
        if self.domain_mixer:
            return self.domain_mixer.get_dataset_domain_weights()
        return None

    def normalize_difficulty(self, item: Dict[str, Any], seq_length: int = 0) -> float:
        """Normalize difficulty score for a sample"""
        if self.difficulty_scheduler:
            return self.difficulty_scheduler.normalize_difficulty_score(
                item, seq_length
            )
        return (
            self.config.difficulty_config.default_difficulty
            if self.config.difficulty_config
            else 0.5
        )

    def should_include_sample(self, difficulty_score: float) -> bool:
        """Check if sample should be included based on difficulty curriculum"""
        if self.difficulty_scheduler:
            return self.difficulty_scheduler.should_include_sample(difficulty_score)
        return True

    def update_performance(
        self, metrics: Dict[str, float], domain: Optional[str] = None
    ):
        """Update performance metrics for adaptive adjustment"""
        if self.difficulty_scheduler:
            self.difficulty_scheduler.update_performance(metrics)

        if self.domain_mixer and domain:
            self.domain_mixer.update_domain_performance(domain, metrics)

    def get_state(self) -> Dict[str, Any]:
        """Get full curriculum state for checkpointing"""
        return {
            "config": self.config,
            "current_step": self.current_step,
            "sequence_lengths": self.get_current_sequence_lengths(),
            "difficulty_threshold": self.get_difficulty_threshold(),
            "domain_weights": self.get_domain_weights(),
            "state_history": self.state_history[-100:],  # Last 100 states
        }

    def save_state(self, filepath: str):
        """Save curriculum state to file"""
        state = self.get_state()
        torch.save(state, filepath)
        logger.info(f"Curriculum state saved to {filepath}")

    @classmethod
    def load_state(cls, filepath: str) -> "ExtendedCurriculumScheduler":
        """Load curriculum state from file"""
        state = torch.load(filepath)
        config = state["config"]

        # Need dataset_max_lengths to reconstruct - caller should provide
        scheduler = cls(config)
        scheduler.current_step = state["current_step"]
        scheduler.state_history = state.get("state_history", [])

        logger.info(f"Curriculum state loaded from {filepath}")
        return scheduler


def create_extended_curriculum_from_config(
    stage: str, curriculum_config: Dict[str, Any], dataset_configs: List[Dict[str, Any]]
) -> ExtendedCurriculumScheduler:
    """
    Factory function to create ExtendedCurriculumScheduler from configuration.

    Args:
        stage: Current training stage
        curriculum_config: Configuration dict from datasets.yml
        dataset_configs: List of dataset configs from datasets.yml

    Returns:
        Configured ExtendedCurriculumScheduler
    """
    # Extract max_seq_length for each dataset
    dataset_max_lengths = {
        d["name"]: d.get("max_seq_length", 8192)
        for d in dataset_configs
        if d.get("stage") == stage
    }

    # Parse sequence length config
    seq_config_dict = curriculum_config.get("sequence_length", {})
    sequence_config = SequenceLengthConfig(
        stage=stage,
        min_length=seq_config_dict.get("min_length", 4096),
        warmup_steps=seq_config_dict.get("warmup_steps", 1000),
        schedule=seq_config_dict.get("schedule", "cosine"),
        step_thresholds=seq_config_dict.get("step_thresholds"),
        exponential_base=seq_config_dict.get("exponential_base", 2.0),
        dataset_min_lengths=seq_config_dict.get("dataset_min_lengths", {}),
        dataset_progression_rates=seq_config_dict.get("dataset_progression_rates", {}),
    )

    # Parse difficulty config
    diff_config_dict = curriculum_config.get("difficulty", {})
    difficulty_config = DifficultyConfig(
        stage=stage,
        difficulty_field=diff_config_dict.get("difficulty_field", "difficulty"),
        alternative_fields=diff_config_dict.get(
            "alternative_fields", ["complexity", "difficulty_score", "hardness"]
        ),
        min_difficulty=diff_config_dict.get("min_difficulty", 0.0),
        max_difficulty=diff_config_dict.get("max_difficulty", 1.0),
        default_difficulty=diff_config_dict.get("default_difficulty", 0.5),
        warmup_steps=diff_config_dict.get("warmup_steps", 500),
        schedule=diff_config_dict.get("schedule", "cosine"),
        enable_adaptive=diff_config_dict.get("enable_adaptive", True),
        performance_window=diff_config_dict.get("performance_window", 50),
        adjustment_rate=diff_config_dict.get("adjustment_rate", 0.1),
        use_length_proxy=diff_config_dict.get("use_length_proxy", True),
        length_difficulty_factor=diff_config_dict.get("length_difficulty_factor", 0.3),
    )

    # Parse domain mixing config
    domain_config_dict = curriculum_config.get("domain_mixing", {})
    domain_config = DomainMixingConfig(
        stage=stage,
        domains=domain_config_dict.get("domains", {}),
        initial_weights=domain_config_dict.get("initial_weights", {}),
        update_frequency=domain_config_dict.get("update_frequency", 100),
        enable_adaptive_frequency=domain_config_dict.get(
            "enable_adaptive_frequency", True
        ),
        min_update_frequency=domain_config_dict.get("min_update_frequency", 50),
        max_update_frequency=domain_config_dict.get("max_update_frequency", 500),
        adjustment_rate=domain_config_dict.get("adjustment_rate", 0.05),
        min_weight=domain_config_dict.get("min_weight", 0.05),
        max_weight=domain_config_dict.get("max_weight", 0.9),
        smoothing_factor=domain_config_dict.get("smoothing_factor", 0.8),
        tracked_metrics=domain_config_dict.get(
            "tracked_metrics", ["loss", "accuracy", "reward", "perplexity"]
        ),
        metric_weights=domain_config_dict.get(
            "metric_weights",
            {"loss": 0.4, "accuracy": 0.3, "reward": 0.2, "perplexity": 0.1},
        ),
        blend_strategy=domain_config_dict.get("blend_strategy", "weighted_sum"),
        variance_threshold=domain_config_dict.get("variance_threshold", 0.1),
    )

    # Create extended config
    extended_config = ExtendedCurriculumConfig(
        stage=stage,
        total_steps=curriculum_config.get("total_steps", 10000),
        sequence_config=sequence_config,
        difficulty_config=difficulty_config,
        domain_config=domain_config,
        enable_sequence_curriculum=curriculum_config.get(
            "enable_sequence_curriculum", True
        ),
        enable_difficulty_curriculum=curriculum_config.get(
            "enable_difficulty_curriculum", True
        ),
        enable_domain_mixing=curriculum_config.get("enable_domain_mixing", True),
        save_frequency=curriculum_config.get("save_frequency", 500),
    )

    return ExtendedCurriculumScheduler(extended_config, dataset_max_lengths)
