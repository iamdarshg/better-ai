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
        
        # Value function for baseline
        self.value_head = nn.Linear(config["hidden_dim"], 1).to(self.device)
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
        new_logprobs = log_probs.gather(1, batch["target_ids"][:, -1].to(self.device).unsqueeze(1)).squeeze(-1)
        
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
                # Get reward scores (would come from reward model in practice)
                # This is a placeholder - actual implementation would score with reward_model
                batch_size = batch["input_ids"].shape[0]
                reward_scores = torch.randn(batch_size, self.group_size).to(self.device)
                old_logprobs = torch.randn(batch_size, self.group_size).to(self.device)
                
                loss_dict = self.train_step(batch, reward_scores, old_logprobs)
                
                for key, val in loss_dict.items():
                    if key in metrics:
                        metrics[key].append(val)
        
        return metrics


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
