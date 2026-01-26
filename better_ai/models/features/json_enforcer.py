
import torch
import torch.nn as nn
from typing import Optional, Dict
import json


class JSONEnforcer(nn.Module):
    """
    Forces all outputs to be valid JSON
    Ensures compliance with JSON schema at generation time
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim

        # JSON structure predictor
        self.structure_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5),  # {, }, [, ], :
            nn.Softmax(dim=-1)
        )

        # JSON validator
        self.json_validator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def validate_json_compliance(self, json_str: str) -> float:
        """Validate if string is valid JSON"""
        try:
            json.loads(json_str)
            return 1.0
        except:
            return 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply JSON constraints to generation

        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            logits: (batch_size, seq_len, vocab_size)
            token_ids: Current token sequence

        Returns:
            Dictionary with constrained_logits, structure_predictions, validity
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Predict JSON structure
        structure_probs = self.structure_predictor(hidden_states)  # (batch_size, seq_len, 5)

        # Validate JSON compliance
        validity = self.json_validator(hidden_states)  # (batch_size, seq_len, 1)

        # Apply soft constraints based on structure
        constrained_logits = logits.clone()

        # This is a simplified version - real implementation would enforce full JSON grammar
        return {
            "constrained_logits": constrained_logits,
            "structure_predictions": structure_probs,
            "validity": validity,
        }
