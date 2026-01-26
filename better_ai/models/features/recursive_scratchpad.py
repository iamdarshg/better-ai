
import torch
import torch.nn as nn
from typing import Optional, Dict


class RecursiveScratchpad(nn.Module):
    """
    Recursive Scratchpad for iterative reasoning
    Recursively processes scratchpad content through the full model
    Maintains sequence length throughout all iterations
    """

    def __init__(self, hidden_dim: int, max_iterations: int = 5, scratchpad_dim: int = 512):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_iterations = max_iterations
        self.scratchpad_dim = scratchpad_dim

        # Scratchpad state management
        self.scratchpad_encoder = nn.Linear(hidden_dim, scratchpad_dim)
        self.scratchpad_decoder = nn.Linear(scratchpad_dim, hidden_dim)

        # Iteration control - decides when to stop recursing
        self.stop_token_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Scratchpad content selector - which tokens get fed back
        self.content_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        model_forward_fn: Optional[callable] = None,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process through recursive scratchpad

        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            model_forward_fn: Callable that takes hidden_states and returns updated hidden_states
                             If None, just encodes/decodes without recursion
            max_iterations: Override default max iterations

        Returns:
            Dictionary with scratchpad_output, reasoning_traces, iteration_count
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        max_iter = max_iterations or self.max_iterations

        # Initialize scratchpad state for each sequence position
        scratchpad_state = self.scratchpad_encoder(hidden_states)  # (batch_size, seq_len, scratchpad_dim)

        reasoning_traces = [hidden_states.detach()]
        current_hidden = hidden_states.clone()

        for iteration in range(max_iter):
            # Encode current state to scratchpad
            scratchpad_state = self.scratchpad_encoder(current_hidden)

            # Decode scratchpad back to hidden space for next iteration
            scratchpad_decoded = self.scratchpad_decoder(scratchpad_state)  # (batch_size, seq_len, hidden_dim)

            # If model_forward_fn provided, recursively call full model on scratchpad content
            if model_forward_fn is not None:
                scratchpad_output = model_forward_fn(scratchpad_decoded)  # (batch_size, seq_len, hidden_dim)
            else:
                scratchpad_output = scratchpad_decoded

            # Select which tokens contribute to next iteration (prevents dilution)
            content_scores = self.content_selector(current_hidden)  # (batch_size, seq_len, 1)

            # Blend scratchpad output with current hidden state
            current_hidden = current_hidden * (1 - 0.3 * content_scores) + scratchpad_output * (0.3 * content_scores)

            reasoning_traces.append(current_hidden.detach())

            # Check if we should stop recursing (per-token decision, aggregate for batch)
            stop_logits = self.stop_token_predictor(current_hidden)  # (batch_size, seq_len, 1)
            stop_decision = (stop_logits.mean(dim=1) > 0.5).any()  # Aggregate across sequence

            if stop_decision and iteration > 0:
                break

        return {
            "scratchpad_output": current_hidden,  # (batch_size, seq_len, hidden_dim)
            "reasoning_traces": torch.stack(reasoning_traces, dim=1),  # (batch_size, num_iterations, seq_len, hidden_dim)
            "iteration_count": len(reasoning_traces),
        }
