
import torch
import torch.nn as nn
from typing import Optional, Dict


class InnerMonologue(nn.Module):
    """
    Inner Monologue with private subspaces
    Uses special tokens (<thought>, </thought>) for reasoning
    """

    def __init__(self, hidden_dim: int, private_subspace_dim: int = 256):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.private_subspace_dim = private_subspace_dim

        # Project to private subspace for reasoning
        self.to_private = nn.Linear(hidden_dim, private_subspace_dim)
        self.from_private = nn.Linear(private_subspace_dim, hidden_dim)

        # Subspace switching logic
        self.subspace_switch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        thought_token_id: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process through inner monologue with private subspace

        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            token_ids: (batch_size, seq_len) token IDs
            thought_token_id: ID of <thought> token

        Returns:
            Dictionary with private_reasoning, output, subspace_usage
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Determine which tokens should use private subspace
        if token_ids is not None and thought_token_id is not None:
            # Mark regions between <thought> and </thought> for private subspace
            in_thought = False
            is_private = torch.zeros_like(token_ids, dtype=torch.bool)
            for i in range(seq_len):
                if token_ids[0, i] == thought_token_id:
                    in_thought = not in_thought
                is_private[:, i] = in_thought
        else:
            # Use learned subspace switch
            switch_scores = self.subspace_switch(hidden_states)  # (batch_size, seq_len, 1)
            is_private = switch_scores > 0.5

        # Project to private subspace
        private_reasoning = self.to_private(hidden_states)  # (batch_size, seq_len, private_dim)

        # Blend public and private
        is_private_expanded = is_private.float()
        public_hidden = hidden_states * (1 - is_private_expanded)
        private_projected = self.from_private(private_reasoning) * is_private_expanded

        output = public_hidden + private_projected

        return {
            "private_reasoning": private_reasoning,
            "output": output,
            "is_private": is_private,
            "subspace_usage": is_private.float().mean(dim=1),
        }
