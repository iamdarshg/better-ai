"""
GRPO (Group Reward Policy Optimization) Algorithm
Replaces PPO with group-based advantage estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Any
import math
from torch.utils.data import DataLoader


class GRPOTrainer:
    """
    Group Reward Policy Optimization Trainer
    Uses group-based advantage estimation for more stable RLHF
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

        # GRPO hyperparameters
        self.beta = config.get("beta", 0.01)  # KL penalty weight
        self.gamma = config.get("gamma", 0.99)  # Discount factor
        self.lam = config.get("lam", 0.95)  # GAE lambda
        self.eps_clip = config.get("eps_clip", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.group_size = config.get("group_size", 4)  # Group size for advantage estimation
        self.device = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = config.get("tokenizer", getattr(model, "tokenizer", None))

        # Value function for baseline
        hidden_dim = config.get("hidden_dim", getattr(model, "config", None).hidden_dim if hasattr(model, "config") else 128)
        self.value_head = nn.Linear(hidden_dim, 1).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value_head.parameters(), lr=config.get("value_lr", 5e-5))

        # Ref policy for KL divergence computation
        self.ref_model = None

    def compute_group_advantages(
        self,
        group_rewards: torch.Tensor,
        group_logprobs: torch.Tensor,
        group_values: torch.Tensor,
        group_dones: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute advantages using group-based estimation

        Args:
            group_rewards: (batch_size, group_size) rewards for each group
            group_logprobs: (batch_size, group_size) log probabilities
            group_values: (batch_size, group_size) baseline values
            group_dones: (batch_size, group_size) done flags

        Returns:
            (advantages, returns, normalized_advantages)
        """
        batch_size, group_size = group_rewards.shape

        # Compute returns within each group
        returns = torch.zeros_like(group_rewards)
        advantages = torch.zeros_like(group_rewards)

        # Use GAE (Generalized Advantage Estimation) within each group
        next_value = 0
        gae = 0

        for t in reversed(range(group_size)):
            if group_dones is not None and t < group_size - 1:
                next_value = group_values[:, t + 1] * (1 - group_dones[:, t + 1])
            else:
                next_value = 0

            # TD error
            delta = group_rewards[:, t] + self.gamma * next_value - group_values[:, t]

            # GAE
            gae = delta + self.gamma * self.lam * gae
            if group_dones is not None:
                gae = gae * (1 - group_dones[:, t])

            advantages[:, t] = gae
            returns[:, t] = gae + group_values[:, t]

        # Normalize advantages per group
        group_mean = advantages.mean(dim=1, keepdim=True)
        group_std = advantages.std(dim=1, keepdim=True) + 1e-8
        normalized_advantages = (advantages - group_mean) / group_std

        return advantages, returns, normalized_advantages

    def compute_policy_loss(
        self,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        ref_logprobs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute GRPO policy loss with clipping

        Args:
            old_logprobs: (batch_size,) old policy log probabilities
            new_logprobs: (batch_size,) new policy log probabilities
            advantages: (batch_size,) computed advantages
            ref_logprobs: (batch_size,) reference policy log probabilities for KL penalty

        Returns:
            (loss, loss_dict)
        """
        # Probability ratio
        ratio = torch.exp(new_logprobs - old_logprobs)

        # Clipped objective
        clipped_ratio = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)

        # Policy loss (negative because we want to maximize)
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL divergence penalty (if reference model provided)
        kl_loss = 0.0
        if ref_logprobs is not None:
            kl = (old_logprobs - ref_logprobs).exp().mean()
            kl_loss = self.beta * kl
            policy_loss = policy_loss + kl_loss

        loss_dict = {
            "policy_loss": policy_loss.item(),
            "kl_penalty": kl_loss if isinstance(kl_loss, float) else kl_loss.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std().item(),
        }

        return policy_loss, loss_dict

    def compute_value_loss(
        self,
        value_preds: torch.Tensor,
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Compute value function loss"""
        value_loss = F.mse_loss(value_preds.squeeze(-1), returns)
        return value_loss, value_loss.item()

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        reward_scores: torch.Tensor,
        old_logprobs: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single GRPO training step

        Args:
            batch: Dictionary with input_ids, attention_mask, etc.
            reward_scores: (batch_size, group_size) reward scores
            old_logprobs: (batch_size, group_size) old log probabilities

        Returns:
            Dictionary of loss metrics
        """
        self.model.train()

        # Get new logprobs from model
        outputs = self.model(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            output_hidden_states=True,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]

        # Compute log probabilities
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # Use last token
        # Handle case where target_ids might not exist or have wrong shape
        if "target_ids" in batch and batch["target_ids"].shape[1] > 0:
            target_ids = batch["target_ids"][:, -1].to(self.device)
        else:
            # Fallback: use the last token from input_ids as target
            target_ids = batch["input_ids"][:, -1].to(self.device)

        # Ensure target_ids is the right shape for gather
        if target_ids.dim() == 1:
            target_ids = target_ids.unsqueeze(1)

        new_logprobs = log_probs.gather(1, target_ids).squeeze(-1)

        # Compute value estimates
        value_preds = self.value_head(hidden_states[:, -1, :])

        # Compute advantages
        advantages, returns, norm_advantages = self.compute_group_advantages(
            reward_scores,
            old_logprobs,
            value_preds.detach(),
        )

        # Flatten for loss computation
        flat_advantages = norm_advantages.view(-1)
        flat_new_logprobs = new_logprobs.view(-1) if new_logprobs.dim() > 1 else new_logprobs
        flat_old_logprobs = old_logprobs.view(-1)

        # Compute policy loss
        policy_loss, policy_loss_dict = self.compute_policy_loss(
            flat_old_logprobs,
            flat_new_logprobs,
            flat_advantages,
        )

        # Compute value loss
        value_loss, value_loss_val = self.compute_value_loss(
            value_preds,
            returns.view(-1),
        )

        # Entropy bonus for exploration
        probs = F.softmax(logits[:, -1, :], dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # Total loss
        total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.value_optimizer.step()

        loss_dict = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss_dict["policy_loss"],
            "value_loss": value_loss_val,
            "entropy": entropy.item(),
            "kl_penalty": policy_loss_dict["kl_penalty"],
        }

        return loss_dict

    def train_epoch(
        self,
        dataloader: DataLoader,
        num_epochs: int = 3,
    ) -> Dict[str, List[float]]:
        """Train for multiple epochs"""
        metrics = {
            "total_loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        }

        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(dataloader):
                # Compute reward scores using the reward model
                reward_scores, old_logprobs = self._compute_group_rewards_and_logprobs(batch)

                loss_dict = self.train_step(batch, reward_scores, old_logprobs)

                for key, val in loss_dict.items():
                    if key in metrics:
                        metrics[key].append(val)

        return metrics

    def _compute_group_rewards_and_logprobs(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reward scores and log probabilities for group-based training

        Args:
            batch: Dictionary containing input_ids, attention_mask, etc.

        Returns:
            (reward_scores, old_logprobs) where both have shape (batch_size, group_size)
        """
        batch_size = batch["input_ids"].shape[0]

        # Initialize tensors for rewards and logprobs
        reward_scores = torch.zeros(batch_size, self.group_size, device=self.device)
        old_logprobs = torch.zeros(batch_size, self.group_size, device=self.device)

        # Store generated trajectories for diversity reward
        all_group_trajectories = [[] for _ in range(batch_size)]

        # Extract input sequences (remove padding if needed)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Generate multiple responses per input for group-based training
        for group_idx in range(self.group_size):
            try:
                # Generate response with different sampling parameters for diversity
                generated_response, response_logprobs = self._generate_response_with_logprobs(
                    input_ids, attention_mask, group_idx
                )

                # Compute reward for the generated response
                response_reward = self._compute_response_reward(generated_response, attention_mask)

                # Store results
                reward_scores[:, group_idx] = response_reward
                old_logprobs[:, group_idx] = response_logprobs

                # Decode for diversity reward if tokenizer is available
                if self.tokenizer:
                    for i in range(batch_size):
                        # Extract only the generated part
                        input_len = input_ids.shape[1]
                        gen_tokens = generated_response[i, input_len:]
                        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                        all_group_trajectories[i].append(text)

            except Exception as e:
                # Fallback to random values if generation/reward computation fails
                import logging
                logging.warning(f"Failed to compute reward for group {group_idx}: {e}")
                reward_scores[:, group_idx] = torch.randn(batch_size, device=self.device)
                old_logprobs[:, group_idx] = torch.randn(batch_size, device=self.device)

        # Apply diversity reward if enabled and tokenizer is available
        if self.config.get("use_diversity_reward", False) and self.tokenizer:
            from .diversity_metrics import get_diversity_reward
            diversity_weight = self.config.get("diversity_reward_weight", 0.1)

            for i in range(batch_size):
                if len(all_group_trajectories[i]) == self.group_size:
                    div_reward = get_diversity_reward(all_group_trajectories[i])
                    reward_scores[i, :] += diversity_weight * div_reward

        return reward_scores, old_logprobs

    def _generate_response_with_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        group_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a response and compute its log probabilities

        Args:
            input_ids: (batch_size, seq_len) input token IDs
            attention_mask: (batch_size, seq_len) attention mask
            group_idx: Index of the group member (for sampling parameter variation)

        Returns:
            (generated_response, response_logprobs) where response_logprobs is the log probability of the generated response
        """
        self.model.eval()

        with torch.no_grad():
            # Set different sampling parameters for each group member to encourage diversity
            # Group 0: More conservative (lower temperature, higher top-p)
            # Group 3: More exploratory (higher temperature, lower top-p)
            base_temperature = 0.8
            base_top_p = 0.9

            # Vary sampling parameters based on group index
            temperature = base_temperature * (0.8 + 0.4 * group_idx / (self.group_size - 1))  # 0.8 to 1.2
            top_p = base_top_p * (1.1 - 0.2 * group_idx / (self.group_size - 1))  # 0.9 to 0.72

            # Generate response
            generated_ids = self._sample_tokens(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                temperature=temperature,
                top_p=top_p,
            )

            # Compute log probabilities of the generated response
            # We need to compute the log probability of the generated tokens given the input
            full_sequence = generated_ids
            full_attention_mask = attention_mask

            if attention_mask is not None and full_sequence.shape[1] > attention_mask.shape[1]:
                # Extend attention mask for generated tokens
                new_tokens_mask = torch.ones(
                    full_sequence.shape[0],
                    full_sequence.shape[1] - attention_mask.shape[1],
                    device=attention_mask.device
                )
                full_attention_mask = torch.cat([attention_mask, new_tokens_mask], dim=1)

            # Forward pass to get logits
            outputs = self.model(
                input_ids=full_sequence,
                attention_mask=full_attention_mask,
                output_hidden_states=False,
            )

            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

            # Compute log probabilities
            # Shift logits and labels to align them
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = full_sequence[..., 1:].contiguous()

            # Compute log probabilities for the generated tokens only
            # Find where the input ends and generation begins
            input_length = input_ids.shape[1]
            if shift_logits.shape[1] > input_length - 1:
                gen_logits = shift_logits[:, input_length - 1:, :]
                gen_labels = shift_labels[:, input_length - 1:]

                # Compute log probabilities
                log_probs = F.log_softmax(gen_logits, dim=-1)
                gen_logprobs = log_probs.gather(-1, gen_labels.unsqueeze(-1)).squeeze(-1)

                # Average log probability across generated tokens for each sequence
                response_logprobs = gen_logprobs.mean(dim=-1)  # (batch_size,)
            else:
                # No tokens were generated
                response_logprobs = torch.zeros(input_ids.shape[0], device=input_ids.device)

        self.model.train()
        return generated_ids, response_logprobs

    def _sample_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Sample tokens from the model with specified parameters

        Args:
            input_ids: (batch_size, seq_len) input token IDs
            attention_mask: (batch_size, seq_len) attention mask
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated token IDs (batch_size, seq_len + max_new_tokens)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Store generated tokens
        generated = input_ids.clone()
        current_attention_mask = attention_mask.clone() if attention_mask is not None else None

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.model(
                input_ids=generated[:, -1:] if current_attention_mask is not None else generated,
                attention_mask=current_attention_mask,
                output_hidden_states=False,
            )

            logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Top-P (Nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold (nucleus filtering)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                # scatter sorted tensors to original indexing
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (batch_size,)

            # Append to generated sequence
            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)

            # Update attention mask if present
            if current_attention_mask is not None:
                new_mask = torch.ones(batch_size, 1, device=device)
                current_attention_mask = torch.cat([current_attention_mask, new_mask], dim=1)

            # Stop if all sequences have generated EOS token
            if (next_tokens == 2).all():  # Assuming EOS token ID is 2
                break

        return generated

    def _compute_response_reward(
        self,
        generated_response: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute reward for a generated response using the reward model

        Args:
            generated_response: (batch_size, seq_len) generated token IDs
            attention_mask: (batch_size, seq_len) attention mask

        Returns:
            Reward scores (batch_size,)
        """
        self.reward_model.eval()

        with torch.no_grad():
            try:
                # Forward pass through the model to get hidden states
                outputs = self.model(
                    input_ids=generated_response,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)

                # Compute reward using the reward model
                if hasattr(self.reward_model, 'forward'):
                    # Standard reward model interface
                    reward_scores = self.reward_model(hidden_states, attention_mask)
                elif hasattr(self.reward_model, 'score_pair'):
                    # Pair-wise scoring interface (for comparison-based rewards)
                    # For now, we'll use the standard interface
                    reward_scores = self.reward_model(hidden_states, attention_mask)
                else:
                    # Fallback: use mean of hidden states as simple reward
                    if attention_mask is not None:
                        mask_expanded = attention_mask.unsqueeze(-1).float()
                        sum_hidden = (hidden_states * mask_expanded).sum(1)
                        sum_mask = mask_expanded.sum(1)
                        hidden_repr = sum_hidden / (sum_mask + 1e-9)
                    else:
                        hidden_repr = hidden_states[:, -1, :]  # Last token

                    # Simple linear projection as fallback reward
                    reward_scores = torch.nn.functional.linear(hidden_repr, torch.randn(hidden_repr.shape[-1], 1, device=hidden_repr.device)).squeeze(-1)

            except Exception as e:
                # Fallback to random rewards if reward computation fails
                import logging
                logging.warning(f"Reward computation failed: {e}")
                reward_scores = torch.randn(generated_response.shape[0], device=generated_response.device)

        self.reward_model.train()
        return reward_scores


class GRPOLoss(nn.Module):
    """Standalone GRPO loss module for custom training loops"""

    def __init__(self, beta: float = 0.01, eps_clip: float = 0.2):
        super().__init__()
        self.beta = beta
        self.eps_clip = eps_clip

    def forward(
        self,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        ref_logprobs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute GRPO loss"""
        ratio = torch.exp(new_logprobs - old_logprobs)
        clipped_ratio = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)

        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        loss = -torch.min(surr1, surr2).mean()

        if ref_logprobs is not None:
            kl = torch.mean(old_logprobs - ref_logprobs)
            loss = loss + self.beta * kl

        return loss
