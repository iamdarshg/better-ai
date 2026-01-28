"""
TiDAR (Think In Diffusion, Output using transformers)
Implementation of a diffusion-based scratchpad refiner
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List


class TiDAR(nn.Module):
    """
    TiDAR module that operates on the scratchpad using a classical diffusion-like process.
    Uses the prompt to iteratively steer the main transformer's internal state.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_steps: int = 5,
        diffusion_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.diffusion_dim = diffusion_dim

        # Small steering transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=diffusion_dim,
            nhead=num_heads,
            dim_feedforward=diffusion_dim * 2,
            batch_first=True,
            norm_first=True
        )
        self.steering_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Projections
        self.input_proj = nn.Linear(hidden_dim, diffusion_dim)
        self.prompt_proj = nn.Linear(hidden_dim, diffusion_dim)
        self.output_proj = nn.Linear(diffusion_dim, hidden_dim)

        # Timestep embeddings for diffusion process
        self.time_embed = nn.Sequential(
            nn.Linear(1, diffusion_dim),
            nn.SiLU(),
            nn.Linear(diffusion_dim, diffusion_dim)
        )

    def forward(
        self,
        scratchpad: torch.Tensor,
        prompt: torch.Tensor,
        noise_level: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Iteratively refine the scratchpad hidden states.

        Args:
            scratchpad: (batch_size, seq_len, hidden_dim) - Initial scratchpad states
            prompt: (batch_size, hidden_dim) - Global prompt embedding for steering
            noise_level: Scale for the noise added during the process

        Returns:
            Dictionary with refined scratchpad and refinement traces
        """
        batch_size, seq_len, _ = scratchpad.shape
        device = scratchpad.device

        # Initial state
        x_t = scratchpad.clone()

        # Project prompt
        c = self.prompt_proj(prompt).unsqueeze(1)  # (batch_size, 1, diffusion_dim)

        traces = [x_t.detach()]

        for t in range(self.num_steps):
            # Normalized timestep [0, 1]
            t_normalized = torch.tensor([[t / self.num_steps]], device=device).repeat(batch_size, 1)
            t_emb = self.time_embed(t_normalized).unsqueeze(1)  # (batch_size, 1, diffusion_dim)

            # Map to diffusion space
            h = self.input_proj(x_t)  # (batch_size, seq_len, diffusion_dim)

            # Combine with prompt and time information
            # We use additive conditioning here
            h_input = h + c + t_emb

            # Steering transformer predicts the noise or the update
            # As per user's request: "adding the smallest change based in the prompt"
            steering_delta = self.steering_transformer(h_input)

            # Map back to hidden_dim
            update = self.output_proj(steering_delta)

            # Update step (simplified diffusion step)
            # In a real diffusion model, we might add noise and then denoise
            # Here we follow the "steering" description: use previous as noisemap and add change
            if noise_level > 0:
                noise = torch.randn_like(update) * noise_level * (1.0 - t / self.num_steps)
                x_t = x_t + 0.1 * update + noise
            else:
                x_t = x_t + 0.1 * update

            traces.append(x_t.detach())

        return {
            "refined_scratchpad": x_t,
            "traces": torch.stack(traces, dim=1),
            "num_steps": self.num_steps
        }
