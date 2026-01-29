
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

def compute_length_aware_dpo_loss(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: Dict[str, Any],
    beta: float = 0.1,
    length_penalty_coeff: float = 0.01,
    complexity_scaling: float = 0.1,
) -> torch.Tensor:
    """
    Compute DPO loss with a dynamic penalty for thinking tokens.

    Args:
        model: The model being trained.
        ref_model: The reference model.
        batch: Dictionary with 'chosen_input_ids', 'rejected_input_ids', etc.
        beta: DPO temperature parameter.
        length_penalty_coeff: Base penalty for thinking tokens.
        complexity_scaling: How much target length reduces the penalty.
    """
    chosen_input_ids = batch['chosen_input_ids']
    rejected_input_ids = batch['rejected_input_ids']

    # 1. Get log probs from model
    chosen_logits = model(chosen_input_ids)["logits"]
    rejected_logits = model(rejected_input_ids)["logits"]

    # 2. Get log probs from ref model
    with torch.no_grad():
        ref_chosen_logits = ref_model(chosen_input_ids)["logits"]
        ref_rejected_logits = ref_model(rejected_input_ids)["logits"]

    # Compute log probs for the completion part
    def get_log_probs(logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return per_token_log_probs.sum(dim=-1)

    # Assuming batch['chosen_labels'] and batch['rejected_labels'] mark the completion
    chosen_log_probs = get_log_probs(chosen_logits[:, :-1, :], chosen_input_ids[:, 1:])
    rejected_log_probs = get_log_probs(rejected_logits[:, :-1, :], rejected_input_ids[:, 1:])

    ref_chosen_log_probs = get_log_probs(ref_chosen_logits[:, :-1, :], chosen_input_ids[:, 1:])
    ref_rejected_log_probs = get_log_probs(ref_rejected_logits[:, :-1, :], rejected_input_ids[:, 1:])

    # DPO Loss
    pi_logratios = chosen_log_probs - rejected_log_probs
    ref_logratios = ref_chosen_log_probs - ref_rejected_log_probs

    logits = pi_logratios - ref_logratios
    dpo_loss = -F.logsigmoid(beta * logits).mean()

    # Dynamic Thought Length Penalty
    # We identify thought tokens between <thought> and </thought>
    # For simplicity, let's assume we have token IDs for these.
    thought_start_id = getattr(model.config, "thought_token_id", 100)
    thought_end_id = getattr(model.config, "thought_end_token_id", 101)

    def count_thought_tokens(input_ids):
        # This is a simplified version
        # In practice, you'd find indices of thought_start and thought_end
        mask = (input_ids == thought_start_id).cumsum(dim=-1) > (input_ids == thought_end_id).cumsum(dim=-1)
        return mask.sum(dim=-1).float()

    chosen_thought_len = count_thought_tokens(chosen_input_ids)

    # Complexity proxy: length of the intended answer (final output after </thought>)
    # We'll approximate this by total length - thought length - prompt length
    prompt_len = batch.get('prompt_len', 0)
    intended_ans_len = (chosen_input_ids != 0).sum(dim=-1).float() - chosen_thought_len - prompt_len

    # Penalty scaling: less penalty if intended answer is longer
    # Penalty = length_penalty_coeff * thought_len * exp(-complexity_scaling * intended_ans_len)
    dynamic_penalty_coeff = length_penalty_coeff * torch.exp(-complexity_scaling * intended_ans_len / 100.0)
    length_penalty = (dynamic_penalty_coeff * chosen_thought_len).mean()

    total_loss = dpo_loss + length_penalty

    return total_loss

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
