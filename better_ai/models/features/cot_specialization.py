
import torch
import torch.nn as nn
from typing import Dict


class CoTSpecializationHeads(nn.Module):
    """
    Dedicated Chain-of-Thought (CoT) specialization heads
    Prevents reasoning token pollution in final outputs
    """

    def __init__(self, hidden_dim: int, num_cot_heads: int = 4, cot_hidden_dim: int = 384):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_cot_heads = num_cot_heads
        self.cot_hidden_dim = cot_hidden_dim

        # Specialized CoT heads for reasoning
        self.cot_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, cot_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(cot_hidden_dim, hidden_dim)
            )
            for _ in range(num_cot_heads)
        ])

        # CoT router to select which heads to use
        self.cot_router = nn.Sequential(
            nn.Linear(hidden_dim, cot_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cot_hidden_dim // 2, num_cot_heads),
            nn.Softmax(dim=-1)
        )

        # Isolation mechanism - prevents reasoning tokens from affecting output
        self.output_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_reasoning_phase: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Process through CoT specialization heads

        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            is_reasoning_phase: If True, route to CoT heads; if False, isolate outputs

        Returns:
            Dictionary with cot_output, routing_weights, isolation_gate
        """
        # Get routing weights
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)
        routing_weights = self.cot_router(hidden_flat)  # (batch_size*seq_len, num_cot_heads)

        # Apply CoT heads
        cot_outputs = []
        for head in self.cot_heads:
            cot_out = head(hidden_flat)
            cot_outputs.append(cot_out)

        cot_outputs = torch.stack(cot_outputs, dim=1)  # (batch_size*seq_len, num_cot_heads, hidden_dim)

        # Combine with routing weights
        routing_weights_expanded = routing_weights.unsqueeze(-1)  # (batch_size*seq_len, num_cot_heads, 1)
        combined_cot = (cot_outputs * routing_weights_expanded).sum(dim=1)  # (batch_size*seq_len, hidden_dim)

        # Reshape back
        combined_cot = combined_cot.view(batch_size, seq_len, hidden_dim)

        # Output isolation gate - prevents reasoning pollution
        isolation_gate = self.output_gate(hidden_states)  # (batch_size, seq_len, 1)

        if is_reasoning_phase:
            # During reasoning, use CoT outputs
            output = combined_cot
        else:
            # During output generation, gate out reasoning tokens
            output = hidden_states * isolation_gate

        return {
            "cot_output": combined_cot,
            "routing_weights": routing_weights.view(batch_size, seq_len, -1),
            "isolation_gate": isolation_gate,
            "final_output": output,
        }
