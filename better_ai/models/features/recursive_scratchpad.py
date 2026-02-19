
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Callable


class RecursiveScratchpad(nn.Module):
    """
    Unified Looped Latent Reasoning (Ouro-inspired)
    Combines Recursive Scratchpad and Inner Monologue concepts.
    Uses parameter-shared looping through model layers in a private latent subspace.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_iterations: int = 5,
        private_subspace_dim: int = 3072,
        latent_vocab_dim: int = 128,
        **kwargs
    ):
        super().__init__()

        # Support old 'scratchpad_dim' name if passed
        if 'scratchpad_dim' in kwargs:
            private_subspace_dim = kwargs['scratchpad_dim']

        self.hidden_dim = hidden_dim
        self.private_subspace_dim = private_subspace_dim
        self.max_iterations = max_iterations
        self.latent_vocab_dim = latent_vocab_dim

        # Projections to/from private subspace
        self.to_private = nn.Linear(hidden_dim, private_subspace_dim)
        self.from_private = nn.Linear(private_subspace_dim, hidden_dim)

        # Latent entropy projection (for uncertainty estimation without tokenization)
        self.entropy_proj = nn.Linear(private_subspace_dim, latent_vocab_dim)

        # Gating mechanisms
        # Entry gate: decides if we should enter the reasoning loop
        self.entry_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

        # Halting gate: decides when to stop looping
        # Takes both the private state and its entropy as input
        self.halting_gate = nn.Sequential(
            nn.Linear(private_subspace_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def compute_latent_entropy(self, private_state: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of a projected latent distribution to estimate uncertainty.
        No discrete tokenization is involved.
        """
        logits = self.entropy_proj(private_state)  # (batch, seq, latent_vocab_dim)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1, keepdim=True)
        # Normalize entropy by log(latent_vocab_dim)
        return entropy / torch.log(torch.tensor(self.latent_vocab_dim, dtype=torch.float32, device=private_state.device))

    def forward(
        self,
        hidden_states: torch.Tensor,
        layers_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process through looped latent reasoning

        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            layers_fn: Callable that takes hidden_states and returns updated hidden_states
                       (typically runs the model's transformer layers)
            max_iterations: Override default max iterations

        Returns:
            Dictionary with scratchpad_output, reasoning_traces, iteration_count, and gating info
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        max_iter = max_iterations or self.max_iterations

        # 1. Determine which tokens enter the reasoning loop
        entry_scores = self.entry_gate(hidden_states)  # (batch_size, seq_len, 1)
        # For simplicity in this implementation, we apply looping to the whole batch
        # but only update those where entry_scores > 0.5 or we use it as a blend factor.

        # 2. Project to private subspace
        private_state = self.to_private(hidden_states)  # (batch_size, seq_len, private_subspace_dim)

        reasoning_traces = [private_state.detach()]
        active_mask = torch.ones(batch_size, seq_len, 1, device=hidden_states.device, dtype=torch.bool)

        # If no layers_fn provided, we can't loop effectively
        if layers_fn is None:
            return {
                "scratchpad_output": hidden_states,
                "reasoning_traces": torch.stack(reasoning_traces, dim=1),
                "iteration_count": 0,
                "entry_scores": entry_scores,
            }

        for iteration in range(max_iter):
            # 3. Project back to full hidden dim for model layers
            current_public = self.from_private(private_state)

            # 4. Pass through model layers (Parameter-shared loop)
            looped_output = layers_fn(current_public)

            # 5. Project back to private subspace
            new_private_state = self.to_private(looped_output)

            # Only update tokens that haven't halted yet
            private_state = torch.where(active_mask, new_private_state, private_state)

            reasoning_traces.append(private_state.detach())

            # 6. Entropy-inspired halting decision
            entropy = self.compute_latent_entropy(private_state)

            # Halting gate takes state and entropy
            halting_input = torch.cat([private_state, entropy], dim=-1)
            halt_scores = self.halting_gate(halting_input)

            # Update active mask (stop if halt_score > 0.5)
            # In inference, we could use this to break the loop early
            new_halts = halt_scores > 0.5
            active_mask = active_mask & (~new_halts)

            if not active_mask.any():
                break

        # 7. Final projection back to public space
        final_private = self.from_private(private_state)

        # 8. Blend with original input based on entry gate
        # This allows the model to choose how much "thinking" to incorporate
        output = hidden_states * (1 - entry_scores) + final_private * entry_scores

        return {
            "scratchpad_output": output,
            "reasoning_traces": torch.stack(reasoning_traces, dim=1),
            "iteration_count": len(reasoning_traces) - 1,
            "entry_scores": entry_scores,
            "final_halting_scores": halt_scores,
            "latent_entropy": entropy
        }
