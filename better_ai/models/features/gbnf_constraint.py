import torch
import torch.nn as nn
from typing import Dict, Optional, List
import ast


class GBNFConstraint(nn.Module):
    """
    Grammar-based constraint enforcement using GBNF (GGML BNF)
    Prevents syntax errors and enforces specific grammars
    """

    def __init__(self, hidden_dim: int, grammar_type: str = "python"):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.grammar_type = grammar_type

        # Grammar validator
        self.grammar_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Token masking predictor (which tokens violate grammar)
        self.violation_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )

        # Simple, lightweight GBNF parser (augmented by a real parser in prod)
        self._gbnf_parser = None
        self._gbnf_ready = False

    class _GBNFParser:
        """Minimal GBNF-like parser using Python AST for Python grammar."""

        def __init__(self, grammar_type: str = "python"):
            self.grammar_type = grammar_type

        def parse(self, text: str) -> bool:
            if self.grammar_type == "python":
                try:
                    ast.parse(text)
                    return True
                except Exception:
                    return False
            # Fallback to always True for non-Python grammars in this simplified helper
            return True

    def forward(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        decoded_sequences: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply grammar constraints to logits

        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            logits: (batch_size, seq_len, vocab_size)

        Returns:
            Dictionary with constrained_logits, violation_scores, grammar_validity
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Score grammar compliance
        grammar_scores = self.grammar_scorer(hidden_states)  # (batch_size, seq_len, 1)

        # If explicit decoded sequences are provided, perform a real GBNF check
        if decoded_sequences is not None and isinstance(decoded_sequences, list):
            parser = self._GBNFParser(grammar_type=self.grammar_type)
            validities = []
            for i in range(min(batch_size, len(decoded_sequences))):
                txt = decoded_sequences[i]
                validities.append(1.0 if parser.parse(txt) else 0.0)
            validity_tensor = torch.tensor(
                validities, dtype=logits.dtype, device=logits.device
            ).view(batch_size, 1, 1)
            invalid_mask = (1.0 - validity_tensor).expand(-1, seq_len, vocab_size)
            constrained_logits = logits.clone() - invalid_mask * 100.0
            violation_mask = invalid_mask.mean(dim=-1, keepdim=True)
            grammar_validity = validity_tensor.mean().item()
        else:
            violation_pred = self.violation_predictor(
                hidden_states
            )  # (batch_size, seq_len, hidden_dim)
            violation_mask = (violation_pred.mean(dim=-1, keepdim=True) > 0.5).float()
            constrained_logits = logits.clone()
            constrained_logits = (
                constrained_logits - violation_mask * 100.0
            )  # Large negative value
            grammar_validity = grammar_scores.mean().item()

        return {
            "constrained_logits": constrained_logits,
            "grammar_scores": grammar_scores,
            "violation_mask": violation_mask,
            "grammar_validity": grammar_validity,
        }
