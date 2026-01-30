"""
Integrated Trainer with Cosine Curriculum and MCTS for CoT
Combines progressive curriculum learning with tree-based reasoning search
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import time
from dataclasses import dataclass, field

from .cosine_curriculum import (
    CosineCurriculumScheduler,
    create_cosine_curriculum,
    CurriculumConfig,
)
from .mcts_cot import MCTSCoTSearcher, create_mcts_cot_searcher, MCTSConfig
from .grpo import GRPOTrainer
from ..config import TrainingConfig, ModelConfig


@dataclass
class CurriculumMCTSConfig:
    """Configuration for integrated curriculum + MCTS training"""

    # Curriculum settings
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    # MCTS settings
    mcts: MCTSConfig = field(default_factory=MCTSConfig)

    # Integration settings
    enable_curriculum: bool = True
    enable_mcts: bool = True

    # Training orchestration
    mcts_frequency: int = 5  # Apply MCTS every N steps
    curriculum_update_frequency: int = 10  # Update curriculum every N steps

    # Data handling
    use_mcts_for_training_data: bool = True  # Use MCTS to generate training examples
    mcts_data_ratio: float = 0.3  # Ratio of MCTS-generated data in training

    # Evaluation
    evaluate_mcts_separately: bool = True
    mcts_eval_frequency: int = 50

    # GRPO integration
    grpo_config: Dict[str, Any] = field(default_factory=dict)


class CurriculumMCTSTrainer:
    """
    Integrated trainer combining cosine curriculum learning with MCTS for CoT
    """

    def __init__(
        self,
        model: nn.Module,
        reward_model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        tokenizer=None,
        config: Optional[CurriculumMCTSConfig] = None,
        training_config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.reward_model = (
            reward_model or getattr(model, "reward_model", model)
        )
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.config = config or CurriculumMCTSConfig()
        self.training_config = training_config or TrainingConfig()

        # Initialize components
        self.curriculum_scheduler = None
        self.mcts_searcher = None
        self.grpo_trainer = None

        self._initialize_components()

        # Training state
        self.current_step = 0
        self.training_metrics = []
        self.mcts_generated_data = []

        # Performance tracking
        self.performance_stats = {
            "total_steps": 0,
            "curriculum_updates": 0,
            "mcts_searches": 0,
            "mcts_successes": 0,
            "grpo_updates": 0,
            "best_mcts_value": 0.0,
        }

        logging.info(
            "Initialized CurriculumMCTSTrainer with cosine curriculum and MCTS"
        )

    def _initialize_components(self):
        """Initialize all training components"""
        # Initialize curriculum scheduler
        if self.config.enable_curriculum:
            self.curriculum_scheduler = CosineCurriculumScheduler(
                self.config.curriculum
            )
            logging.info("Cosine curriculum scheduler initialized")

        # Initialize MCTS searcher
        if self.config.enable_mcts and self.tokenizer is not None:
            self.mcts_searcher = MCTSCoTSearcher(
                self.model, self.tokenizer, self.config.mcts
            )
            logging.info("MCTS CoT searcher initialized")

        # Initialize GRPO trainer
        if self.reward_model and self.optimizer:
            grpo_config = {
                "learning_rate": self.training_config.learning_rate,
                "batch_size": self.training_config.batch_size,
                "max_grad_norm": 1.0,
                **self.config.grpo_config,
            }
            self.grpo_trainer = GRPOTrainer(
                self.model, self.reward_model, self.optimizer, grpo_config
            )
            logging.info("GRPO trainer initialized")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step with curriculum and MCTS integration
        """
        self.current_step += 1
        self.performance_stats["total_steps"] += 1

        step_metrics = {}

        # Update curriculum if enabled
        if self.config.enable_curriculum and self.curriculum_scheduler:
            if self.current_step % self.config.curriculum_update_frequency == 0:
                curriculum_state = self.curriculum_scheduler.step()
                step_metrics.update(self._process_curriculum_state(curriculum_state))
                self.performance_stats["curriculum_updates"] += 1

        # Apply MCTS if enabled and it's time
        if (
            self.config.enable_mcts
            and self.mcts_searcher
            and self.current_step % self.config.mcts_frequency == 0
        ):
            # Select a question from batch for MCTS search
            mcts_results = self._apply_mcts_to_batch(batch)
            step_metrics.update(mcts_results)
            self.performance_stats["mcts_searches"] += 1

        # Perform standard GRPO training
        if self.grpo_trainer:
            # Mix MCTS-generated data if available
            training_batch = self._prepare_training_batch(batch)

            # GRPO update
            grpo_metrics = self._perform_grpo_update(training_batch)
            step_metrics.update(grpo_metrics)
            self.performance_stats["grpo_updates"] += 1

        # Update curriculum performance
        if self.config.enable_curriculum and self.curriculum_scheduler:
            self.curriculum_scheduler.update_performance(step_metrics)

        # Log step
        if self.current_step % 10 == 0:
            self._log_training_progress(step_metrics)

        # Store metrics
        self.training_metrics.append(step_metrics)

        return step_metrics

    def _process_curriculum_state(
        self, curriculum_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """Process curriculum state into metrics"""
        return {
            "curriculum_difficulty": curriculum_state["difficulty"],
            "curriculum_progress": curriculum_state["progress"],
            "curriculum_stage_idx": float(curriculum_state["stage_idx"]),
        }

    def _apply_mcts_to_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Apply MCTS search to questions in batch"""
        if not self.mcts_searcher or not self.tokenizer:
            return {"mcts_applied": 0.0}

        questions = self._extract_questions_from_batch(batch)

        if not questions:
            return {"mcts_applied": 0.0}

        mcts_metrics = {
            "mcts_applied": len(questions),
            "mcts_total_value": 0.0,
            "mcts_avg_reasoning_length": 0.0,
            "mcts_avg_answer_length": 0.0,
            "mcts_successes": 0,
        }

        # Apply MCTS to each question
        for question in questions[:1]:  # Limit to 1 question per batch for efficiency
            try:
                # Perform MCTS search
                mcts_result = self.mcts_searcher.search(question)

                # Store result for potential training data
                self.mcts_generated_data.append(
                    {
                        "question": question,
                        "reasoning_trace": mcts_result["best_reasoning_trace"],
                        "answer": mcts_result["best_answer"],
                        "value": mcts_result["best_value"],
                        "step": self.current_step,
                    }
                )

                # Update metrics
                mcts_metrics["mcts_total_value"] += mcts_result["best_value"]
                mcts_metrics["mcts_avg_reasoning_length"] += len(
                    mcts_result["best_reasoning_trace"]
                )
                mcts_metrics["mcts_avg_answer_length"] += len(
                    mcts_result["best_answer"]
                )

                if mcts_result["best_value"] > 0.5:
                    mcts_metrics["mcts_successes"] += 1
                    self.performance_stats["mcts_successes"] += 1

                # Track best MCTS value
                if (
                    mcts_result["best_value"]
                    > self.performance_stats["best_mcts_value"]
                ):
                    self.performance_stats["best_mcts_value"] = mcts_result[
                        "best_value"
                    ]

            except Exception as e:
                logging.warning(f"MCTS search failed for question: {e}")

        # Calculate averages
        if questions:
            mcts_metrics["mcts_avg_value"] = mcts_metrics["mcts_total_value"] / len(
                questions[:1]
            )

        return mcts_metrics

    def _extract_questions_from_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> List[str]:
        """Extract questions from batch using tokenizer decoding"""
        questions = []
        input_ids = batch.get("input_ids")

        if input_ids is not None and self.tokenizer is not None:
            # Decode top few examples
            batch_size = input_ids.shape[0]
            for i in range(min(batch_size, 2)):
                text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                # Simple heuristic: take everything before the first "Answer:" or similar
                question = text.split("Answer:")[0].strip()
                if question:
                    questions.append(question)

        return questions

    def _prepare_training_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Prepare training batch with MCTS-generated data"""
        if not self.config.use_mcts_for_training_data or not self.mcts_generated_data:
            return batch

        recent_mcts_data = [
            data
            for data in self.mcts_generated_data
            if self.current_step - data["step"] < 100
        ]

        if not recent_mcts_data:
            return batch

        mixed_batch = batch.copy()
        mixed_batch["has_mcts_data"] = torch.tensor([True], device=batch["input_ids"].device)
        mixed_batch["mcts_success_rate"] = torch.tensor(
            [
                len([d for d in recent_mcts_data if d["value"] > 0.5])
                / max(1, len(recent_mcts_data))
            ],
            device=batch["input_ids"].device
        )

        return mixed_batch

    def _perform_grpo_update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform GRPO update with real reward and logprob calculations"""
        if not self.grpo_trainer:
            return {"grpo_loss": 0.0}

        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")

        # 1. Forward pass to get real rewards and logprobs
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_advanced_features=True)
            logits = outputs.get("logits")

            # Use real reward model
            if hasattr(self.reward_model, "forward"):
                 # If reward model is separate
                 rewards = self.reward_model(outputs["last_hidden_state"], attention_mask)
            else:
                 # If reward model is internal to the model (DeepSeek style)
                 rewards = outputs.get("advanced_features", {}).get("reward", torch.zeros(input_ids.size(0)))

            # Calculate logprobs for the chosen actions
            logprobs_all = F.log_softmax(logits, dim=-1)
            logprobs = logprobs_all.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

        self.model.train()

        # Perform real GRPO update
        try:
            grpo_result = self.grpo_trainer.train_step(batch, rewards, logprobs)
            return {
                "grpo_loss": grpo_result.get("loss", 0.0),
                "grpo_reward_mean": rewards.mean().item(),
                "grpo_reward_std": rewards.std().item() if rewards.numel() > 1 else 0.0,
            }
        except Exception as e:
            logging.warning(f"GRPO update failed: {e}")
            return {"grpo_loss": 0.0, "grpo_error": 1.0}

    def _log_training_progress(self, metrics: Dict[str, float]):
        """Log training progress"""
        log_msg = f"Step {self.current_step}: "

        if "curriculum_difficulty" in metrics:
            log_msg += f"Diff={metrics['curriculum_difficulty']:.3f}, "
            log_msg += f"Progress={metrics['curriculum_progress']:.3f}, "

        if "mcts_applied" in metrics and metrics["mcts_applied"] > 0:
            log_msg += f"MCTS={metrics['mcts_applied']}, "
            if "mcts_avg_value" in metrics:
                log_msg += f"MCTS_Val={metrics['mcts_avg_value']:.3f}, "

        if "grpo_loss" in metrics:
            log_msg += f"Loss={metrics['grpo_loss']:.4f}, "

        if self.performance_stats["mcts_searches"] > 0:
            success_rate = (
                self.performance_stats["mcts_successes"]
                / self.performance_stats["mcts_searches"]
            )
            log_msg += f"MCTS_Success_Rate={success_rate:.2f}, "

        logging.info(log_msg.rstrip(", "))

    def train_epoch(self, dataloader) -> Dict[str, List[float]]:
        """Train for one epoch"""
        epoch_metrics = {
            "total_loss": [],
            "curriculum_difficulty": [],
            "mcts_avg_value": [],
            "mcts_successes": [],
            "grpo_loss": [],
        }

        for batch_idx, batch in enumerate(dataloader):
            step_metrics = self.train_step(batch)

            for key, value in step_metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key].append(value)

            if batch_idx % 20 == 0:
                logging.info(f"Epoch batch {batch_idx}/{len(dataloader)} completed")

        return epoch_metrics

    def evaluate(self, eval_dataloader) -> Dict[str, float]:
        """Evaluate model with and without MCTS"""
        eval_results = {}

        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = self.model(**batch)
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    total_loss += outputs.loss.item()
                    total_samples += 1

        eval_results["eval_loss"] = total_loss / max(1, total_samples)

        if self.config.enable_mcts and self.config.evaluate_mcts_separately:
            mcts_eval_results = self._evaluate_mcts_separately()
            eval_results.update(mcts_eval_results)

        if self.config.enable_curriculum and self.curriculum_scheduler:
            curriculum_stats = self.curriculum_scheduler.get_statistics()
            eval_results.update(
                {
                    f"curriculum_{k}": v
                    for k, v in curriculum_stats.items()
                    if isinstance(v, (int, float, bool))
                }
            )

        return eval_results

    def _evaluate_mcts_separately(self) -> Dict[str, float]:
        """Evaluate MCTS performance separately"""
        if not self.mcts_searcher:
            return {}

        eval_questions = [
            "What is 15 + 27?",
            "If a train travels 60 mph for 2 hours, how far does it travel?",
            "What is the square root of 144?",
        ]

        mcts_results = {
            "mcts_eval_avg_value": 0.0,
            "mcts_eval_avg_reasoning_steps": 0.0,
            "mcts_eval_success_rate": 0.0,
        }

        successful_searches = 0

        for question in eval_questions:
            try:
                result = self.mcts_searcher.search(question)
                mcts_results["mcts_eval_avg_value"] += result["best_value"]
                mcts_results["mcts_eval_avg_reasoning_steps"] += len(
                    result["best_reasoning_trace"]
                )

                if result["best_value"] > 0.5:
                    successful_searches += 1

            except Exception as e:
                logging.warning(f"MCTS evaluation failed for question: {e}")

        if eval_questions:
            mcts_results["mcts_eval_avg_value"] /= len(eval_questions)
            mcts_results["mcts_eval_avg_reasoning_steps"] /= len(eval_questions)
            mcts_results["mcts_eval_success_rate"] = successful_searches / len(
                eval_questions
            )

        return mcts_results

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        stats = self.performance_stats.copy()

        if self.config.enable_curriculum and self.curriculum_scheduler:
            stats["curriculum"] = self.curriculum_scheduler.get_statistics()

        if self.config.enable_mcts and self.mcts_searcher:
            stats["mcts_search_stats"] = self.mcts_searcher.search_stats
            stats["mcts_generated_data_count"] = len(self.mcts_generated_data)

        if self.training_metrics:
            recent_metrics = self.training_metrics[-10:]
            stats["recent_avg_loss"] = sum(
                m.get("grpo_loss", 0) for m in recent_metrics
            ) / len(recent_metrics)
            stats["recent_avg_mcts_value"] = sum(
                m.get("mcts_avg_value", 0) for m in recent_metrics
            ) / len(recent_metrics)

        return stats

    def save_training_state(self, filepath: str):
        """Save complete training state"""
        state = {
            "config": self.config.__dict__,
            "current_step": self.current_step,
            "performance_stats": self.performance_stats,
            "training_metrics": self.training_metrics[-1000:],
            "mcts_generated_data": self.mcts_generated_data[-100:],
        }

        if self.config.enable_curriculum and self.curriculum_scheduler:
            curriculum_state_path = filepath.replace(".pt", "_curriculum.pt")
            self.curriculum_scheduler.save_state(curriculum_state_path)
            state["curriculum_state_path"] = curriculum_state_path

        model_state_path = filepath.replace(".pt", "_model.pt")
        torch.save(self.model.state_dict(), model_state_path)
        state["model_state_path"] = model_state_path

        torch.save(state, filepath)
        logging.info(f"Training state saved to {filepath}")

    def load_training_state(self, filepath: str):
        """Load training state"""
        try:
            state = torch.load(filepath)

            self.current_step = state.get("current_step", 0)
            self.performance_stats = state.get("performance_stats", {})
            self.training_metrics = state.get("training_metrics", [])
            self.mcts_generated_data = state.get("mcts_generated_data", [])

            if self.config.enable_curriculum and self.curriculum_scheduler:
                curriculum_state_path = state.get("curriculum_state_path")
                if curriculum_state_path:
                    self.curriculum_scheduler.load_state(curriculum_state_path)

            model_state_path = state.get("model_state_path")
            if model_state_path:
                self.model.load_state_dict(torch.load(model_state_path))

            logging.info(f"Training state loaded from {filepath}")

        except Exception as e:
            logging.error(f"Failed to load training state: {e}")


def create_curriculum_mcts_trainer(
    model: nn.Module,
    reward_model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    tokenizer=None,
    config: Optional[CurriculumMCTSConfig] = None,
    training_config: Optional[TrainingConfig] = None,
) -> CurriculumMCTSTrainer:
    """Factory function to create integrated trainer"""
    return CurriculumMCTSTrainer(
        model=model,
        reward_model=reward_model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        config=config,
        training_config=training_config,
    )
