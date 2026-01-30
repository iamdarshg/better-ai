
import torch
import torch.nn.functional as F
from typing import Dict, Any

def rl_forward_pass(self, batch: Dict[str, Any]) -> tuple:
    """PPO-like forward pass for RLHF with correct probability ratios."""
    input_ids = batch.get('input_ids')
    if input_ids is None:
        return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), None

    # 1. Get policy and value estimates
    outputs = self.model(input_ids=input_ids, return_advanced_features=True)
    logits = outputs.get("logits")
    values = outputs.get("advanced_features", {}).get("value")

    if logits is None or values is None:
        return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), None

    # 2. Reference model for KL penalty
    with torch.no_grad():
        ref_outputs = self.ref_model(input_ids=input_ids)
        ref_logits = ref_outputs.get("logits")

    # 3. Calculate advantages
    # Use Hierarchical Reward Model (HRM) for production scoring
    if hasattr(self.model, "hrm"):
        rewards = self.model.hrm(outputs["last_hidden_state"], batch.get("attention_mask"))
    else:
        rewards = self.model.reward_model(outputs["last_hidden_state"], batch.get("attention_mask"))

    if values.dim() == 3:
        values_seq = values[:, -1, :].squeeze(-1)
    else:
        values_seq = values.squeeze(-1)

    advantages = rewards - values_seq
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 4. Calculate PPO loss with probability ratio
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)

    # Sample actions from real rollouts if not provided
    generated_ids = batch.get('generated_ids')
    if generated_ids is None:
        # Perform real rollout generation
        self.model.eval()
        with torch.no_grad():
            # Use model's internal generation or a simplified sampler
            # For this step, we'll assume we generate a few tokens
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.8
            )
        self.model.train()

    # Align logits with generated tokens
    # Note: logits from forward pass might only be for input_ids
    # We need to perform a forward pass on the full generated sequence to get action logprobs
    full_outputs = self.model(input_ids=generated_ids)
    full_logits = full_outputs["logits"]

    with torch.no_grad():
        full_ref_outputs = self.ref_model(input_ids=generated_ids)
        full_ref_logits = full_ref_outputs["logits"]

    def get_action_logprobs(logits, ids):
        # Shift to align prediction with target
        shift_logits = logits[:, :-1, :].contiguous()
        shift_ids = ids[:, 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        return log_probs.gather(dim=-1, index=shift_ids.unsqueeze(-1)).squeeze(-1)

    action_log_probs = get_action_logprobs(full_logits, generated_ids)
    old_action_log_probs = get_action_logprobs(full_ref_logits, generated_ids)

    # ratio = exp(log_p - log_p_old)
    ratio = torch.exp(action_log_probs - old_action_log_probs)

    clip_range = 0.2
    advantages_expanded = advantages.unsqueeze(-1).expand(-1, ratio.size(1))

    surr1 = ratio * advantages_expanded
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages_expanded

    policy_loss = -torch.min(surr1, surr2).mean()

    # 5. KL Divergence Penalty
    kl_div = (torch.exp(ref_log_probs) * (ref_log_probs - log_probs)).sum(dim=-1).mean()
    kl_coeff = 0.01

    value_loss = F.mse_loss(values_seq, rewards)

    total_loss = policy_loss + 0.5 * value_loss + kl_coeff * kl_div
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
    Compute DPO loss with a robust penalty for thinking tokens.
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

    def get_log_probs(logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return per_token_log_probs.sum(dim=-1)

    chosen_log_probs = get_log_probs(chosen_logits[:, :-1, :], chosen_input_ids[:, 1:])
    rejected_log_probs = get_log_probs(rejected_logits[:, :-1, :], rejected_input_ids[:, 1:])

    ref_chosen_log_probs = get_log_probs(ref_chosen_logits[:, :-1, :], chosen_input_ids[:, 1:])
    ref_rejected_log_probs = get_log_probs(ref_rejected_logits[:, :-1, :], rejected_input_ids[:, 1:])

    pi_logratios = chosen_log_probs - rejected_log_probs
    ref_logratios = ref_chosen_log_probs - ref_rejected_log_probs

    logits = pi_logratios - ref_logratios
    dpo_loss = -F.logsigmoid(beta * logits).mean()

    # Robust Thought Length Penalty
    thought_start_id = getattr(model.config, "thought_token_id", 100)
    thought_end_id = getattr(model.config, "thought_end_token_id", 101)

    def count_thought_tokens(ids):
        # Find indices of start and end tokens
        # We handle multiple thought blocks by summing all segments between start and end
        starts = (ids == thought_start_id).int()
        ends = (ids == thought_end_id).int()

        # cumulative sum: 1 when inside thought block, 0 when outside
        # (assuming thoughts don't nest, which is standard)
        inside = (starts.cumsum(dim=-1) - ends.cumsum(dim=-1)) > 0
        return inside.sum(dim=-1).float()

    chosen_thought_len = count_thought_tokens(chosen_input_ids)

    prompt_len = batch.get('prompt_len', 0)
    # Real intended answer length: everything after the last </thought>
    # or just everything excluding thoughts and prompt.
    intended_ans_len = (chosen_input_ids != 0).sum(dim=-1).float() - chosen_thought_len - prompt_len

    dynamic_penalty_coeff = length_penalty_coeff * torch.exp(-complexity_scaling * intended_ans_len / 100.0)
    length_penalty = (dynamic_penalty_coeff * chosen_thought_len).mean()

    total_loss = dpo_loss + length_penalty

    return total_loss

def rl_stage2_forward_pass(self, batch: Dict[str, Any]) -> tuple:
    """Stage 2 RLHF using multi-attribute rewards and PPO ratio."""
    input_ids = batch.get('input_ids')
    if input_ids is None:
        return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), None

    outputs = self.model(input_ids=input_ids, return_advanced_features=True)
    logits = outputs.get("logits")
    values = outputs.get("advanced_features", {}).get("value")

    if logits is None or values is None:
        return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), None

    # Reference model for KL and PPO ratio
    with torch.no_grad():
        ref_outputs = self.ref_model(input_ids=input_ids)
        ref_logits = ref_outputs.get("logits")

    multi_attr_out = self.model.multi_attr_reward(outputs["last_hidden_state"], batch.get("attention_mask"))
    point_estimates = multi_attr_out.get("point_estimates")
    rewards = point_estimates.mean(dim=-1)

    if values.dim() == 3:
        values_seq = values[:, -1, :].squeeze(-1)
    else:
        values_seq = values.squeeze(-1)

    advantages = rewards - values_seq
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)

    action_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    old_action_log_probs = ref_log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

    ratio = torch.exp(action_log_probs - old_action_log_probs)

    clip_range = 0.2
    advantages_expanded = advantages.unsqueeze(-1).expand(-1, ratio.size(1))

    surr1 = ratio * advantages_expanded
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages_expanded

    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(values_seq, rewards)

    multi_loss = torch.tensor(0.0, device=self.device)
    if 'reward_targets' in batch:
        multi_loss = self.model.multi_attr_reward.quantile_loss(multi_attr_out, batch['reward_targets'])

    total_loss = policy_loss + 0.5 * value_loss + multi_loss
    aux_loss = outputs.get('aux_loss', torch.tensor(0.0, device=self.device))
    expert_ids = outputs.get('expert_ids')

    return total_loss, aux_loss, expert_ids
