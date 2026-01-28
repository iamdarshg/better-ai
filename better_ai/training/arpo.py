"""
ARPO: Agentic Reinforced Policy Optimization
Implements entropy-based adaptive rollout mechanism with advantage attribution for multi-turn tool interactions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Any
import math
import logging


class EntropyMonitor:
    """
    Monitors token entropy during generation to detect high-uncertainty phases
    """

    def __init__(self, window_size: int = 10, threshold_multiplier: float = 2.0):
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier
        self.entropy_history = []
        self.baseline_entropy = None

    def compute_token_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy for each token position"""
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return entropy

    def update(self, logits: torch.Tensor) -> Dict[str, Any]:
        """Update entropy monitor and return analysis"""
        token_entropy = self.compute_token_entropy(logits)
        current_entropy = token_entropy.mean().item()

        self.entropy_history.append(current_entropy)
        if len(self.entropy_history) > self.window_size:
            self.entropy_history.pop(0)

        # Initialize baseline if needed
        if (
            self.baseline_entropy is None
            and len(self.entropy_history) >= self.window_size
        ):
            self.baseline_entropy = sum(self.entropy_history) / len(
                self.entropy_history
            )

        # Detect entropy spikes (high uncertainty)
        is_spike = False
        if self.baseline_entropy is not None:
            threshold = self.baseline_entropy * self.threshold_multiplier
            is_spike = current_entropy > threshold

        return {
            "current_entropy": current_entropy,
            "baseline_entropy": self.baseline_entropy,
            "is_spike": is_spike,
            "entropy_history": self.entropy_history.copy(),
        }


class AdaptiveRolloutManager:
    """
    Manages adaptive rollout branching based on entropy monitoring
    """

    def __init__(self, base_branch_factor: int = 1, max_branch_factor: int = 4):
        self.base_branch_factor = base_branch_factor
        self.max_branch_factor = max_branch_factor
        self.current_branch_factor = base_branch_factor

    def get_branch_factor(self, entropy_analysis: Dict[str, Any]) -> int:
        """Determine branching factor based on entropy"""
        if entropy_analysis.get("is_spike", False):
            # Increase branching during high uncertainty
            self.current_branch_factor = min(
                self.max_branch_factor, self.current_branch_factor + 1
            )
        else:
            # Gradually return to baseline
            self.current_branch_factor = max(
                self.base_branch_factor, self.current_branch_factor - 1
            )

        return self.current_branch_factor


class StepLevelAdvantageAttributor:
    """
    Computes step-level advantage attribution for multi-turn tool interactions
    """

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam

    def compute_step_attributions(
        self,
        tool_states: List[Dict[str, Any]],
        final_rewards: torch.Tensor,
        tool_sequence: List[int],
    ) -> torch.Tensor:
        """
        Compute step-level advantages for tool-use sequences

        Args:
            tool_states: List of states after each tool use
            final_rewards: Final rewards for each trajectory
            tool_sequence: Sequence of tool indices used
        """
        num_steps = len(tool_states)
        num_trajectories = len(final_rewards)

        # Initialize step-level advantages
        step_advantages = torch.zeros(num_steps, num_trajectories)

        for traj_idx in range(num_trajectories):
            # Work backwards from final reward
            accumulated_reward = final_rewards[traj_idx]

            for step in reversed(range(num_steps)):
                # Estimate immediate reward for this step
                state_value = self._estimate_state_value(tool_states[step])
                next_state_value = (
                    self._estimate_state_value(tool_states[step + 1])
                    if step < num_steps - 1
                    else 0
                )

                # Temporal difference error
                td_error = (
                    state_value + self.gamma * next_state_value - accumulated_reward
                )

                # GAE computation
                step_advantages[step, traj_idx] = td_error
                accumulated_reward = accumulated_reward * self.lam + td_error

        return step_advantages

    def _estimate_state_value(self, state: Dict[str, Any]) -> float:
        """Estimate value of a given state"""
        # Simple heuristic based on state properties
        value = 0.0

        # Consider tool execution success
        if state.get("tool_success", False):
            value += 0.5

        # Consider error presence
        if not state.get("has_error", False):
            value += 0.3

        # Consider progress toward goal
        value += state.get("progress_score", 0.0)

        return value


class ARPOTrainer:
    """
    Agentic Reinforced Policy Optimization Trainer
    Extends GRPO with entropy-based adaptive rollouts and step-level attribution
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

        # ARPO components
        self.entropy_monitor = EntropyMonitor(
            window_size=config.get("entropy_window", 10),
            threshold_multiplier=config.get("entropy_threshold", 2.0),
        )
        self.rollout_manager = AdaptiveRolloutManager(
            base_branch_factor=config.get("base_branch_factor", 1),
            max_branch_factor=config.get("max_branch_factor", 4),
        )
        self.advantage_attributor = StepLevelAdvantageAttributor(
            gamma=config.get("gamma", 0.99), lam=config.get("lam", 0.95)
        )

        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.tool_use_history = []

        # Configuration
        self.beta = config.get("beta", 0.01)
        self.eps_clip = config.get("eps_clip", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.device = config.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        logging.info(f"ARPOTrainer initialized with device: {self.device}")

    def generate_with_adaptive_rollouts(
        self,
        prompts: List[str],
        max_length: int = 512,
        temperature: float = 0.7,
        enable_adaptive: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate responses with adaptive rollout branching

        Returns list of trajectories with metadata
        """
        trajectories = []

        for prompt in prompts:
            # Initial generation
            with torch.no_grad():
                inputs = self._tokenize(prompt)
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                )

            # Monitor entropy during generation
            entropy_analysis = self._analyze_generation_entropy(outputs)

            # Determine branching factor
            branch_factor = self.rollout_manager.base_branch_factor
            if enable_adaptive:
                branch_factor = self.rollout_manager.get_branch_factor(entropy_analysis)

            # Generate additional rollouts if needed
            rollouts = [outputs]  # Primary rollout
            if branch_factor > 1:
                rollouts.extend(
                    self._generate_additional_rollouts(
                        prompt, max_length, temperature, branch_factor - 1
                    )
                )

            # Score all rollouts
            scored_rollouts = self._score_rollouts(rollouts, prompt)

            trajectories.append(
                {
                    "prompt": prompt,
                    "primary_rollout": outputs,
                    "all_rollouts": scored_rollouts,
                    "entropy_analysis": entropy_analysis,
                    "branch_factor": branch_factor,
                    "tool_uses": self._detect_tool_uses(outputs),
                }
            )

        return trajectories

    def _analyze_generation_entropy(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze entropy in generation outputs"""
        if "scores" not in outputs:
            return {"current_entropy": 0.0, "baseline_entropy": 0.0, "is_spike": False}

        # Analyze entropy across generation steps
        all_scores = torch.stack(outputs["scores"])
        entropy_analysis = {}

        for step_scores in all_scores:
            analysis = self.entropy_monitor.update(step_scores)
            if not entropy_analysis:  # First analysis
                entropy_analysis = analysis
            else:  # Keep track of maximum entropy
                if analysis["current_entropy"] > entropy_analysis["current_entropy"]:
                    entropy_analysis = analysis

        return entropy_analysis

    def _generate_additional_rollouts(
        self, prompt: str, max_length: int, temperature: float, num_rollouts: int
    ) -> List[Dict[str, Any]]:
        """Generate additional rollouts for adaptive exploration"""
        rollouts = []

        for _ in range(num_rollouts):
            with torch.no_grad():
                inputs = self._tokenize(prompt)
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature * 1.2,  # Higher temp for exploration
                    do_sample=True,
                    return_dict_in_generate=True,
                )
                rollouts.append(outputs)

        return rollouts

    def _score_rollouts(
        self, rollouts: List[Dict[str, Any]], prompt: str
    ) -> List[Dict[str, Any]]:
        """Score rollouts using reward model"""
        scored_rollouts = []

        for rollout in rollouts:
            # Extract generated text
            generated_text = self._decode_output(rollout)

            # Score with reward model
            with torch.no_grad():
                reward_score = self.reward_model.score(prompt, generated_text)

            scored_rollouts.append(
                {
                    "rollout": rollout,
                    "text": generated_text,
                    "reward_score": reward_score,
                    "tool_uses": self._detect_tool_uses(rollout),
                }
            )

        # Sort by reward score
        scored_rollouts.sort(key=lambda x: x["reward_score"], reverse=True)
        return scored_rollouts

    def _detect_tool_uses(self, rollout: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect tool usage in rollout"""
        tool_uses = []

        if "sequences" in rollout:
            generated_ids = rollout["sequences"][0]
            generated_text = self._decode_output(rollout)

            # Simple heuristic for tool detection (would be more sophisticated in practice)
            tool_indicators = ["python:", "search(", "calculate(", "execute("]

            for i, indicator in enumerate(tool_indicators):
                if indicator in generated_text:
                    tool_uses.append(
                        {
                            "tool_type": indicator.replace(":", ""),
                            "position": generated_text.find(indicator),
                            "step_id": i,
                        }
                    )

        return tool_uses

    def compute_step_level_advantages(
        self, trajectories: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute step-level advantages for multi-turn tool interactions
        """
        all_step_advantages = {}

        for traj_idx, trajectory in enumerate(trajectories):
            tool_uses = trajectory["tool_uses"]
            rollout_rewards = [r["reward_score"] for r in trajectory["all_rollouts"]]

            if not tool_uses:
                continue

            # Create mock tool states for demonstration
            tool_states = []
            for i, tool_use in enumerate(tool_uses):
                tool_states.append(
                    {
                        "tool_success": True,
                        "has_error": False,
                        "progress_score": i * 0.1,
                    }
                )

            # Compute step-level advantages
            step_advantages = self.advantage_attributor.compute_step_attributions(
                tool_states,
                torch.tensor(rollout_rewards),
                [tool["step_id"] for tool in tool_uses],
            )

            all_step_advantages[f"trajectory_{traj_idx}"] = step_advantages

        return all_step_advantages

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize input text"""
        # This would use the model's tokenizer in practice
        return {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        }

    def _decode_output(self, output: Dict[str, Any]) -> str:
        """Decode model output to text"""
        # Mock implementation
        if "sequences" in output:
            return "Generated response"
        return "Generated response"

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single ARPO training step with adaptive rollouts and step-level attribution
        """
        self.model.train()

        # Generate adaptive rollouts
        prompts = self._extract_prompts_from_batch(batch)
        trajectories = self.generate_with_adaptive_rollouts(
            prompts, enable_adaptive=self.config.get("enable_adaptive_rollouts", True)
        )

        # Compute step-level advantages
        step_advantages = self.compute_step_level_advantages(trajectories)

        # Compute policy loss with step-level attribution
        total_loss = 0.0
        loss_components = {}

        for traj_key, advantages in step_advantages.items():
            traj_idx = int(traj_key.split("_")[1])
            trajectory = trajectories[traj_idx]

            # Use primary rollout for loss computation
            primary_rollout = trajectory["primary_rollout"]

            # Compute log probabilities for primary rollout
            logits = primary_rollout.get("logits", torch.randn(1, 10, 1000))
            log_probs = F.log_softmax(logits, dim=-1)

            # Weight advantages by entropy analysis
            entropy_weight = 1.0
            if trajectory["entropy_analysis"]["is_spike"]:
                entropy_weight = self.config.get("high_entropy_weight", 2.0)

            weighted_advantages = advantages * entropy_weight

            # Policy loss
            loss = -(log_probs * weighted_advantages.unsqueeze(-1)).mean()
            total_loss += loss

            loss_components[traj_key] = loss.item()

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.step_count += 1

        return {
            "total_loss": total_loss.item(),
            "num_trajectories": len(trajectories),
            "avg_branch_factor": sum(t["branch_factor"] for t in trajectories)
            / len(trajectories),
            "entropy_spikes": sum(
                1 for t in trajectories if t["entropy_analysis"]["is_spike"]
            ),
            **loss_components,
        }
