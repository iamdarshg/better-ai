"""
Integration module for ARPO, CLEANER, and KV-Cache features
Combines all three top features into cohesive training system
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any, Tuple
import logging

from .arpo import ARPOTrainer, EntropyMonitor, AdaptiveRolloutManager
from .cleaner import CLEANERDataCollector, create_cleaner_pipeline
from .kv_cache_grpo import OptimizedGRPOWithKVCache, KVCacheManager
from .grpo import GRPOTrainer
from .machine_feedback import MachineFeedbackTrainer
from .steca import STeCaController, TrajectoryRefiner


class IntegratedAdvancedTrainer:
    """
    Integrated trainer combining ARPO, CLEANER, and KV-Cache optimizations
    """

    def __init__(
        self,
        model: nn.Module,
        reward_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
    ):
        self.model = model
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.config = config

        # Initialize components based on configuration
        self._initialize_components()

        # Performance tracking
        self.training_stats = {
            "total_steps": 0,
            "arpo_improvements": 0,
            "cleaner_corrections": 0,
            "kv_cache_saves": 0,
            "memory_saved_mb": 0.0,
        }

        logging.info(
            "IntegratedAdvancedTrainer initialized with ARPO, CLEANER, KV-Cache"
        )

    def _initialize_components(self):
        """Initialize training components based on config"""
        # ARPO component
        if self.config.get("enable_arpo", True):
            self.arpo_trainer = ARPOTrainer(
                self.model, self.reward_model, self.optimizer, self.config
            )
        else:
            self.arpo_trainer = None

        # CLEANER component
        if self.config.get("enable_cleaner", True):
            self.cleaner_collector = create_cleaner_pipeline(
                min_similarity=self.config.get("cleaner_similarity_threshold", 0.5),
                purification_enabled=self.config.get("enable_purification", True),
            )
        else:
            self.cleaner_collector = None

        # KV-Cache component
        if self.config.get("enable_kv_cache", True):
            self.kv_optimized_trainer = OptimizedGRPOWithKVCache(
                self.model, self.reward_model, self.optimizer, self.config
            )
        else:
            self.kv_optimized_trainer = None

        # Machine Feedback component
        if self.config.get("enable_machine_feedback", False):
            self.mf_trainer = MachineFeedbackTrainer(
                self.model, self.config
            )
        else:
            self.mf_trainer = None

        # STeCa component
        if self.config.get("enable_steca", True):
            refiner = TrajectoryRefiner(self.model, getattr(self.model, "tokenizer", None))
            self.steca_controller = STeCaController(
                refiner, calibration_threshold=self.config.get("steca_threshold", 0.7)
            )
        else:
            self.steca_controller = None

        # Fallback to standard GRPO
        if not any([self.arpo_trainer, self.kv_optimized_trainer, self.mf_trainer]):
            self.grpo_trainer = GRPOTrainer(
                self.model, self.reward_model, self.optimizer, self.config
            )
        else:
            self.grpo_trainer = None

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Unified training step using all enabled optimizations
        """
        self.training_stats["total_steps"] += 1
        step_metrics = {}

        # Pre-process batch with CLEANER if enabled
        if self.cleaner_collector:
            batch = self._preprocess_batch_with_cleaner(batch)

        # Apply STeCa calibration if enabled
        if self.steca_controller:
            batch = self._calibrate_batch_with_steca(batch)

        # Choose training approach based on enabled components
        if self.mf_trainer and self.config.get("current_stage") == "machine_feedback":
            mf_metrics = self.mf_trainer.train_step(batch)
            step_metrics.update(mf_metrics)

        elif self.kv_optimized_trainer:
            # Use KV-cache optimized training
            kv_metrics = self.kv_optimized_trainer.train_step_with_cache_optimization(
                batch
            )
            step_metrics.update(kv_metrics)

            # Track cache savings
            self.training_stats["kv_cache_saves"] += kv_metrics.get("cache_reuses", 0)
            self.training_stats["memory_saved_mb"] += kv_metrics.get(
                "memory_saved_mb", 0
            )

        elif self.arpo_trainer:
            # Use ARPO training
            arpo_metrics = self.arpo_trainer.train_step(batch)
            step_metrics.update(arpo_metrics)

            # Track ARPO improvements
            self.training_stats["arpo_improvements"] += arpo_metrics.get(
                "entropy_spikes", 0
            )

        elif self.grpo_trainer:
            # Use standard GRPO with real data
            # Generate rewards if not provided in batch
            if "rewards" not in batch:
                with torch.no_grad():
                    outputs = self.model(batch["input_ids"], attention_mask=batch.get("attention_mask"), return_advanced_features=True)
                    rewards = outputs.get("advanced_features", {}).get("reward", torch.zeros(batch["input_ids"].size(0)))
                    logprobs = torch.nn.functional.log_softmax(outputs["logits"], dim=-1).gather(-1, batch["input_ids"].unsqueeze(-1)).squeeze(-1)
            else:
                rewards = batch["rewards"]
                logprobs = batch["logprobs"]

            grpo_metrics = self.grpo_trainer.train_step(
                batch,
                rewards,
                logprobs
            )
            step_metrics.update(grpo_metrics)

        return step_metrics

    def _preprocess_batch_with_cleaner(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Pre-process batch data using CLEANER purification
        """
        if not self.cleaner_collector:
            return batch

        # Extract trajectories from batch
        raw_trajectories = self._extract_trajectories_from_batch(batch)

        # Apply CLEANER purification
        purified_trajectories = self.cleaner_collector.collect_batch(raw_trajectories)

        # Convert back to batch format
        purified_batch = self._convert_trajectories_to_batch(purified_trajectories, batch)

        # Track corrections
        cleaner_stats = self.cleaner_collector.get_statistics()
        self.training_stats["cleaner_corrections"] += cleaner_stats.get(
            "errors_corrected", 0
        )

        return purified_batch

    def _extract_trajectories_from_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> List[List[Dict[str, Any]]]:
        """Extract real trajectory format from batch data by decoding input_ids"""
        if 'trajectories' in batch:
            return batch['trajectories']

        # Implement real parsing from input_ids
        input_ids = batch.get("input_ids")
        if input_ids is None or not hasattr(self, "tokenizer") or self.tokenizer is None:
            return []

        trajectories = []
        for i in range(input_ids.size(0)):
            text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)

            # Parse trajectory: split into steps based on markers
            # Markers could be "Step N:", "<thought>", or just newline segments
            steps_text = text.split("\n\n") # Simplified heuristic for steps
            trajectory = []
            for step_text in steps_text:
                if step_text.strip():
                    trajectory.append({
                        "content": step_text.strip(),
                        "metadata": {"batch_idx": i}
                    })
            trajectories.append(trajectory)

        return trajectories

    def _calibrate_batch_with_steca(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Calibrate batch trajectories using STeCa
        """
        if not self.steca_controller:
            return batch

        # Extract trajectories
        trajectories = self._extract_trajectories_from_batch(batch)

        # Get current rewards for calibration decision
        with torch.no_grad():
            outputs = self.model(batch["input_ids"], return_advanced_features=True)
            rewards = outputs.get("advanced_features", {}).get("reward", torch.zeros(batch["input_ids"].size(0)))

        # Calibrate
        # STeCa expects a list of dicts with "steps" key
        steca_input = [{"steps": traj} for traj in trajectories]
        calibrated_steps = self.steca_controller.calibrate(steca_input, rewards)

        # Convert back to batch
        return self._convert_trajectories_to_batch(calibrated_steps, batch)

    def _convert_trajectories_to_batch(
        self, trajectories: List[List[Dict[str, Any]]], original_batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Convert purified trajectories back to batch format via re-tokenization"""
        if not trajectories or not hasattr(self, "tokenizer") or self.tokenizer is None:
            return original_batch

        device = original_batch["input_ids"].device
        purified_texts = []
        for traj in trajectories:
            # Reconstruct text from trajectory steps
            purified_text = "\n\n".join([step["content"] for step in traj])
            purified_texts.append(purified_text)

        # Re-tokenize
        tokens = self.tokenizer(
            purified_texts,
            padding=True,
            truncation=True,
            max_length=original_batch["input_ids"].size(1),
            return_tensors="pt"
        ).to(device)

        new_batch = original_batch.copy()
        new_batch["input_ids"] = tokens["input_ids"]
        new_batch["attention_mask"] = tokens["attention_mask"]

        return new_batch

    def train_epoch(
        self, dataloader: torch.utils.data.DataLoader, num_epochs: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train for multiple epochs with all optimizations
        """
        epoch_metrics = {
            "total_loss": [],
            "cache_hit_rate": [],
            "memory_saved_mb": [],
            "entropy_spikes": [],
            "cleaner_corrections": [],
        }

        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(dataloader):
                # Training step with all optimizations
                step_metrics = self.train_step(batch)

                # Collect metrics
                for key, value in step_metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key].append(value)

                # Log progress
                if batch_idx % 10 == 0:
                    self._log_training_progress(epoch, batch_idx, step_metrics)

        return epoch_metrics

    def _log_training_progress(
        self, epoch: int, batch_idx: int, metrics: Dict[str, float]
    ):
        """Log training progress with all optimization metrics"""
        log_msg = f"Epoch {epoch}, Batch {batch_idx}: "

        if "loss" in metrics:
            log_msg += f"Loss={metrics['loss']:.4f}"

        if "cache_hit_rate" in metrics:
            log_msg += f", Cache_Hit={metrics['cache_hit_rate']}"

        if "entropy_spikes" in metrics:
            log_msg += f", Entropy_Spikes={metrics['entropy_spikes']}"

        if "memory_saved_mb" in metrics:
            log_msg += f", Memory_Saved={metrics['memory_saved_mb']:.2f}MB"

        logging.info(log_msg)

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all optimizations"""
        stats = self.training_stats.copy()

        # Add component-specific statistics
        if self.arpo_trainer:
            stats["arpo_config"] = {
                "entropy_window": self.config.get("entropy_window", 10),
                "entropy_threshold": self.config.get("entropy_threshold", 2.0),
                "base_branch_factor": self.config.get("base_branch_factor", 1),
                "max_branch_factor": self.config.get("max_branch_factor", 4),
            }

        if self.cleaner_collector:
            stats["cleaner_stats"] = self.cleaner_collector.get_statistics()

        if self.kv_optimized_trainer:
            stats["kv_cache_stats"] = (
                self.kv_optimized_trainer.get_optimization_statistics()
            )

        # Calculate overall efficiency metrics
        stats["overall_efficiency"] = self._calculate_overall_efficiency()

        return stats

    def _calculate_overall_efficiency(self) -> Dict[str, float]:
        """Calculate overall efficiency metrics"""
        steps = max(1, self.training_stats["total_steps"])

        return {
            "arpo_impact_rate": self.training_stats["arpo_improvements"] / steps * 100,
            "cleaner_correction_rate": self.training_stats["cleaner_corrections"]
            / steps
            * 100,
            "kv_cache_saving_rate": self.training_stats["kv_cache_saves"] / steps * 100,
            "memory_efficiency_mb": self.training_stats["memory_saved_mb"] / steps,
        }

    def save_optimization_state(self, filepath: str):
        """Save optimization state for resuming training"""
        state = {
            "training_stats": self.training_stats,
            "config": self.config,
        }

        # Add component states
        if self.kv_optimized_trainer:
            state["kv_cache_manager"] = {
                "cache_entries": len(
                    self.kv_optimized_trainer.cache_manager.cache_entries
                ),
                "statistics": self.kv_optimized_trainer.cache_manager.get_statistics(),
            }

        if self.cleaner_collector:
            state["cleaner_stats"] = self.cleaner_collector.get_statistics()

        torch.save(state, filepath)
        logging.info(f"Optimization state saved to {filepath}")

    def load_optimization_state(self, filepath: str):
        """Load optimization state for resuming training"""
        try:
            state = torch.load(filepath)
            self.training_stats = state.get("training_stats", {})

            logging.info(f"Optimization state loaded from {filepath}")
            return True
        except Exception as e:
            logging.error(f"Failed to load optimization state: {e}")
            return False


def create_integrated_trainer(
    model: nn.Module,
    reward_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
) -> IntegratedAdvancedTrainer:
    """
    Factory function to create integrated trainer with all optimizations
    """
    # Default configuration
    default_config = {
        "enable_arpo": True,
        "enable_cleaner": True,
        "enable_kv_cache": True,
        "entropy_window": 10,
        "entropy_threshold": 2.0,
        "base_branch_factor": 1,
        "max_branch_factor": 4,
        "max_cache_size": 1000,
        "cleaner_similarity_threshold": 0.5,
        "enable_purification": True,
    }

    # Merge with user config
    merged_config = {**default_config, **config}

    return IntegratedAdvancedTrainer(model, reward_model, optimizer, merged_config)
