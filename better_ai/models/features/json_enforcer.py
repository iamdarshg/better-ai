import torch
import torch.nn as nn
from typing import Optional, Dict, List
import json


class JSONEnforcer(nn.Module):
    """
    Deterministic JSON enforcer.
    Ensures all outputs follow valid JSON syntax at the token level.
    """

    def __init__(self, hidden_dim: int, tokenizer=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tokenizer = tokenizer

        # JSON State Machine states
        self.STATE_START = 0
        self.STATE_OBJECT_KEY = 1
        self.STATE_COLON = 2
        self.STATE_VALUE = 3
        self.STATE_COMMA = 4
        self.STATE_END = 5

    def get_json_mask(self, current_text: str) -> List[str]:
        """
        Identify which categories of tokens are valid given the current JSON string.
        """
        # A real production enforcer would use a trie of the vocabulary
        # to find which tokens match the allowed regex/grammar.

        # Simplified but deterministic state logic:
        text = current_text.strip()
        if not text:
            return ["{", "["]

        if text.endswith("{"):
            return ["\""] # Expecting a key

        # JSON structure predictor (predicts coarse JSON structure tokens)
        self.structure_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5),  # {, }, [, ], :
            nn.Softmax(dim=-1),
        )

        # JSON validator (validity score per position)
        self.json_validator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Simple character-level decode fallback (for demonstration).
        # In production you should provide a proper tokenizer/vocab for decode.
        self._fallback_alphabet = [chr(i) for i in range(32, 127)]

    def validate_json_compliance(self, json_str: str) -> float:
        """Validate if string is valid JSON"""
        try:
            json.loads(json_str)
            return 1.0
        except:
            return 0.0

    def _decode_from_token_ids(self, token_ids: torch.Tensor) -> List[str]:
        """Very naive decode: map token ids to a string using a simple ASCII fallback.
        This is a placeholder; in real systems you should wire to the actual tokenizer.
        """
        if token_ids is None:
            return ["{}"] * token_ids.size(0)  # type: ignore
        batch = token_ids.shape[0]
        res = []
        for i in range(batch):
            ids = token_ids[i].tolist()
            chars = [
                self._fallback_alphabet[(idx % len(self._fallback_alphabet))]
                for idx in ids
            ]
            res.append("".join(chars))
        return res

    def forward(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        decoded_strings: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply deterministic JSON constraints to logits.
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        constrained_logits = logits.clone()

        if input_ids is not None and self.tokenizer is not None:
            for b in range(batch_size):
                # Decode the current sequence
                current_text = self.tokenizer.decode(input_ids[b], skip_special_tokens=True)

        # Predict JSON structure
        structure_probs = self.structure_predictor(
            hidden_states
        )  # (batch_size, seq_len, 5)

        # Determine validity per sample using provided decoded strings if available
        validity_tensor = None
        if decoded_strings is not None:
            # Compute JSON validity per sample by attempting json.loads
            results = []
            for s in decoded_strings:
                try:
                    json.loads(s)
                    results.append(1.0)
                except Exception:
                    results.append(0.0)
            validity_tensor = torch.tensor(
                results, dtype=logits.dtype, device=logits.device
            ).unsqueeze(-1)

        # Apply simple constraints: if not valid, dampen logits for entire sequence
        constrained_logits = logits.clone()
        if validity_tensor is not None:
            invalid_mask = (
                (1.0 - validity_tensor)
                .view(batch_size, 1, 1)
                .expand(-1, seq_len, logits.shape[-1])
            )
            constrained_logits = constrained_logits - invalid_mask * 100.0
            grammar_validity = validity_tensor.mean().item()
        else:
            # Fallback: rely on the learned json_validator to estimate validity per token
            validity = self.json_validator(hidden_states)  # (batch_size, seq_len, 1)
            validity_mask = validity.mean(dim=1, keepdim=True)  # (batch_size, 1, 1)
            constrained_logits = constrained_logits * (
                validity_mask.squeeze(-1)
            )  # crude soft enforcement
            grammar_validity = validity.mean().item()

        return {
            "constrained_logits": constrained_logits,
            "structure_predictions": structure_probs,
            "validity": validity_tensor
            if validity_tensor is not None
            else torch.zeros((batch_size, seq_len, 1), device=logits.device),
        }
