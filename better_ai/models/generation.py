
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict

def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    use_cache: bool = True,
) -> torch.Tensor:
    """
    Generate tokens using the model

    Args:
        input_ids: (batch_size, seq_len)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-K sampling parameter
        top_p: Nucleus (Top-P) sampling parameter
        use_cache: Whether to use KV cache

    Returns:
        Generated token ids (batch_size, seq_len + max_new_tokens)
    """
    batch_size = input_ids.shape[0]
    device = input_ids.device

    # Initialize past key values
    past_key_values = None

    # Store all generated tokens
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # Forward pass
        outputs = self.forward(
            input_ids=generated[:, -1:] if past_key_values else generated,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_advanced_features=False,
        )

        logits = outputs["logits"][:, -1, :]  # (batch_size, vocab_size)

        # Apply temperature
        logits = logits / temperature

        # Top-K sampling
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(-1, top_k_indices, top_k_logits)

        # Top-P (Nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (nucleus filtering)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (batch_size,)

        # Append to generated
        generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)

        # Stop if EOS token
        if (next_tokens == 2).all():  # Assuming EOS token ID is 2
            break

    return generated

def compute_loss(
    self,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    reward_labels: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute combined loss for training

    Args:
        input_ids: (batch_size, seq_len)
        labels: (batch_size, seq_len) - language modeling targets
        reward_labels: (batch_size,) - reward/preference labels for RLHF

    Returns:
        Dictionary with loss components
    """
    outputs = self.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_advanced_features=True,
    )

    logits = outputs["logits"]

    # Language modeling loss
    lm_loss = F.cross_entropy(
        logits.view(-1, self.vocab_size),
        labels.view(-1),
        reduction="mean",
    )

    losses = {"lm_loss": lm_loss}

    # Reward modeling loss (if labels provided)
    if reward_labels is not None:
        advanced_features = outputs.get("advanced_features", {})

        # BR-RM loss
        if "reward" in advanced_features:
            reward_pred = advanced_features["reward"]
            reward_loss = F.mse_loss(reward_pred, reward_labels.float())
            losses["reward_loss"] = reward_loss

        # Multi-attribute reward loss
        if "multi_attr_reward" in advanced_features:
            multi_attr = advanced_features["multi_attr_reward"]
            # Simplified multi-attribute loss
            multi_loss = sum([
                F.mse_loss(v.mean(dim=-1), reward_labels)
                for v in multi_attr.values()
                if isinstance(v, torch.Tensor) and v.dim() > 1
            ]) / max(len([v for v in multi_attr.values() if isinstance(v, torch.Tensor)]), 1)
            losses["multi_attr_loss"] = multi_loss

    # Total loss
    total_loss = sum(losses.values())
    losses["total_loss"] = total_loss

    return losses

def self_correct(
    self,
    input_ids: torch.Tensor,
    tokenizer,
    max_new_tokens: int = 128,
    verification_keyword: str = "error",
) -> Tuple[str, bool]:
    """
    Generates a response and performs self-correction if a keyword is detected.

    Args:
        input_ids: The input prompt token IDs.
        tokenizer: The tokenizer for decoding.
        max_new_tokens: The maximum number of tokens to generate.
        verification_keyword: The keyword to check for in the initial response.

    Returns:
        A tuple containing the final response and a boolean indicating if correction was performed.
    """
    # 1. Generate initial response
    initial_response_ids = self.generate(input_ids, max_new_tokens=max_new_tokens)
    initial_response_text = tokenizer.decode(initial_response_ids[0], skip_special_tokens=True)

    # 2. Verify the response
    needs_correction = verification_keyword in initial_response_text.lower()

    if not needs_correction:
        return initial_response_text, False

    # 3. If correction is needed, generate a new response with a correction prompt
    correction_prompt = (
        f"The following response contains an error: '{initial_response_text}'."
        "Please correct the error and provide a new, accurate response."
    )
    correction_input_ids = tokenizer(correction_prompt, return_tensors="pt").input_ids.to(input_ids.device)

    corrected_response_ids = self.generate(correction_input_ids, max_new_tokens=max_new_tokens)
    corrected_response_text = tokenizer.decode(corrected_response_ids[0], skip_special_tokens=True)

    return corrected_response_text, True
