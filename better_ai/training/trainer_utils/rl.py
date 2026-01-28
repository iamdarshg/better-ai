
import torch
import torch.nn.functional as F
from typing import Dict, Any

def rl_forward_pass(self, batch: Dict[str, Any]) -> tuple:
    """PPO-like forward pass for RLHF."""
    input_ids = batch.get('input_ids')
    if input_ids is None:
        return torch.tensor(0.0), torch.tensor(0.0), None

    # 1. Get policy and value estimates
    outputs = self.model(input_ids=input_ids, return_advanced_features=True)
    logits = outputs.get("logits")
    values = outputs.get("advanced_features", {}).get("value")

    if logits is None or values is None:
        return torch.tensor(0.0), torch.tensor(0.0), None

    # 2. Generate a response
    probs = torch.nn.functional.softmax(logits, dim=-1)
    generated_ids = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(probs.size(0), probs.size(1))

    # 3. Get rewards
    rewards = self.model.reward_model(outputs["hidden_states"], batch.get("attention_mask"))

    # 4. Calculate advantages
    if values.dim() == 3:
        values_seq = values[:, -1, :].squeeze(-1)
    else:
        values_seq = values.squeeze(-1)

    advantages = rewards - values_seq
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 5. Calculate PPO loss
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    action_log_probs = log_probs.gather(dim=-1, index=generated_ids.unsqueeze(-1)).squeeze(-1)

    # For simplicity, we use a clipped loss without the ratio
    clip_range = 0.2
    advantages_expanded = advantages.unsqueeze(-1).expand(-1, action_log_probs.size(1))

    policy_loss = -torch.min(
        action_log_probs * advantages_expanded,
        torch.clamp(action_log_probs, 1 - clip_range, 1 + clip_range) * advantages_expanded
    ).mean()

    value_loss = F.mse_loss(values_seq, rewards)

    total_loss = policy_loss + 0.5 * value_loss
    aux_loss = outputs.get('aux_loss', torch.tensor(0.0, device=self.device))
    expert_ids = outputs.get('expert_ids')

    return total_loss, aux_loss, expert_ids

def rl_stage2_forward_pass(self, batch: Dict[str, Any]) -> tuple:
    """Stage 2 RLHF using multi-attribute rewards."""
    input_ids = batch.get('input_ids')
    if input_ids is None:
        return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), None

    # 1. Get policy and value estimates
    outputs = self.model(input_ids=input_ids, return_advanced_features=True)
    logits = outputs.get("logits")
    values = outputs.get("advanced_features", {}).get("value")

    if logits is None or values is None:
        return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), None

    # 2. Get multi-attribute rewards
    multi_attr_out = self.model.multi_attr_reward(outputs["hidden_states"], batch.get("attention_mask"))
    point_estimates = multi_attr_out.get("point_estimates")  # (batch_size, num_attributes)

    # Combined reward is the average of all attribute point estimates
    rewards = point_estimates.mean(dim=-1)

    # 3. Calculate advantages
    # Ensure values and rewards match
    if values.dim() == 3:
        # Take the value of the last token for sequence-level reward
        values_seq = values[:, -1, :].squeeze(-1)
    else:
        values_seq = values.squeeze(-1)

    advantages = rewards - values_seq
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 4. Calculate PPO loss
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    generated_ids = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(probs.size(0), probs.size(1))

    action_log_probs = log_probs.gather(dim=-1, index=generated_ids.unsqueeze(-1)).squeeze(-1)

    clip_range = 0.2
    # Expand advantages to match sequence length
    advantages_expanded = advantages.unsqueeze(-1).expand(-1, action_log_probs.size(1))

    policy_loss = -torch.min(
        action_log_probs * advantages_expanded,
        torch.clamp(action_log_probs, 1 - clip_range, 1 + clip_range) * advantages_expanded
    ).mean()

    value_loss = F.mse_loss(values_seq, rewards)

    # Multi-attribute quantile loss (if targets available in batch, else skipped)
    multi_loss = torch.tensor(0.0, device=self.device)
    if 'reward_targets' in batch:
        multi_loss = self.model.multi_attr_reward.quantile_loss(multi_attr_out, batch['reward_targets'])

    total_loss = policy_loss + 0.5 * value_loss + multi_loss
    aux_loss = outputs.get('aux_loss', torch.tensor(0.0, device=self.device))
    expert_ids = outputs.get('expert_ids')

    return total_loss, aux_loss, expert_ids
