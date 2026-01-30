"""
TiDAR (Think In Diffusion, Output using transformers)
Implementation of a robust diffusion-based scratchpad refiner
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import math


class TiDAR(nn.Module):
    """
    TiDAR module that operates on the scratchpad using a robust diffusion process.
    Uses iterative refinement with a noise schedule to steer hidden states.
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

        # Steering transformer (denoising network)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=diffusion_dim,
            nhead=num_heads,
            dim_feedforward=diffusion_dim * 4,
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

        # Timestep embeddings (standard sinusoidal + MLP)
        self.time_embed = nn.Sequential(
            nn.Linear(diffusion_dim, diffusion_dim),
            nn.SiLU(),
            nn.Linear(diffusion_dim, diffusion_dim)
        )

    def _get_timestep_embedding(self, timesteps, dim):
        """Standard sinusoidal timestep embedding"""
        half_dim = dim // 2
        exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        emb = torch.exp(exponent)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(
        self,
        scratchpad: torch.Tensor,
        prompt: torch.Tensor,
        noise_level: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """
        Iteratively refine scratchpad states using a diffusion-inspired loop.
        """
        batch_size, seq_len, _ = scratchpad.shape
        device = scratchpad.device

        # Initial hidden states
        x_t = scratchpad.clone()

        # Project prompt for conditioning
        c = self.prompt_proj(prompt).unsqueeze(1)  # (B, 1, D)

        traces = [x_t.detach()]

        # Iterative refinement loop
        # We treat the hidden state as the "image" and the prompt as conditioning
        for t in range(self.num_steps):
            # Timestep embedding
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            t_emb = self._get_timestep_embedding(t_tensor, self.diffusion_dim)
            t_emb = self.time_embed(t_emb).unsqueeze(1) # (B, 1, D)

            # Map to diffusion space
            h = self.input_proj(x_t)

            # Condition on prompt and time
            h_cond = h + c + t_emb

            # Predict the "noise" or refinement delta
            epsilon_theta = self.steering_transformer(h_cond)
            delta = self.output_proj(epsilon_theta)

            # Refinement step: x_{t-1} = x_t - alpha * delta + noise
            # We use a simple linear schedule for alpha
            alpha = 1.0 - (t / self.num_steps)

            # Update state
            x_t = x_t + 0.1 * alpha * delta

            if noise_level > 0 and self.training:
                noise = torch.randn_like(x_t) * noise_level * alpha
                x_t = x_t + noise

            traces.append(x_t.detach())

        return {
            "refined_scratchpad": x_t,
            "traces": torch.stack(traces, dim=1),
            "num_steps": self.num_steps
        }
