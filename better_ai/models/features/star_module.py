
import torch
import torch.nn as nn
from typing import List, Dict


class STaRModule(nn.Module):
    """
    Self-Taught Reasoner (STaR) for bootstrapping
    Iteratively improves reasoning with self-consistency checking
    """

    def __init__(self, hidden_dim: int, num_bootstrap_rounds: int = 3, consistency_samples: int = 8):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_bootstrap_rounds = num_bootstrap_rounds
        self.consistency_samples = consistency_samples

        # Self-consistency checker
        self.consistency_checker = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Reasoning trace validator
        self.trace_validator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def check_consistency(
        self,
        reasoning_trace1: torch.Tensor,
        reasoning_trace2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check if two reasoning traces are consistent

        Returns:
            Consistency score (0-1)
        """
        combined = torch.cat([reasoning_trace1, reasoning_trace2], dim=-1)
        consistency = self.consistency_checker(combined)
        return consistency.squeeze(-1)

    def validate_trace(self, reasoning_trace: torch.Tensor) -> torch.Tensor:
        """Validate a single reasoning trace"""
        validity = self.trace_validator(reasoning_trace)
        return validity.squeeze(-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        reasoning_traces: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        STaR bootstrapping for reasoning improvement

        Args:
            hidden_states: (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)
            reasoning_traces: List of reasoning traces from multiple rounds

        Returns:
            Dictionary with bootstrapped_trace, consistency_scores, validity_scores
        """
        if not reasoning_traces:
            # Return empty results if no traces provided
            batch_size = hidden_states.shape[0]
            return {
                "bootstrapped_trace": hidden_states if hidden_states.dim() > 1 else hidden_states.unsqueeze(0),
                "validity_scores": torch.ones(batch_size, 1),
                "consistency_scores": torch.ones(batch_size, 1),
                "best_trace_idx": torch.zeros(batch_size, dtype=torch.long),
            }

        if hidden_states.dim() == 3:
            # Use mean pooling across sequence for validity scoring
            hidden_repr = hidden_states.mean(dim=1)  # (batch_size, hidden_dim)
        else:
            hidden_repr = hidden_states

        batch_size = hidden_repr.shape[0]

        # Validate each trace - traces are (batch_size, seq_len, hidden_dim)
        validity_scores = []
        for trace in reasoning_traces:
            if trace.dim() == 3:
                trace_repr = trace.mean(dim=1)  # Pool to (batch_size, hidden_dim)
            else:
                trace_repr = trace
            validity = self.validate_trace(trace_repr)  # (batch_size,)
            validity_scores.append(validity)

        validity_scores = torch.stack(validity_scores, dim=1)  # (batch_size, num_traces)

        # Check consistency between top traces
        consistency_matrix = []
        for i, trace1 in enumerate(reasoning_traces):
            for trace2 in reasoning_traces[i+1:]:
                trace1_repr = trace1.mean(dim=1) if trace1.dim() == 3 else trace1
                trace2_repr = trace2.mean(dim=1) if trace2.dim() == 3 else trace2
                consistency = self.check_consistency(trace1_repr, trace2_repr)
                consistency_matrix.append(consistency)

        consistency_scores = torch.stack(consistency_matrix, dim=1) if consistency_matrix else torch.ones(batch_size, 1)

        # Select best trace based on validity
        best_validity_idx = validity_scores.argmax(dim=1)  # (batch_size,)
        bootstrapped_trace = torch.stack([
            reasoning_traces[best_validity_idx[i].item()] for i in range(batch_size)
        ])

        return {
            "bootstrapped_trace": bootstrapped_trace,
            "validity_scores": validity_scores,
            "consistency_scores": consistency_scores,
            "best_trace_idx": best_validity_idx,
        }
