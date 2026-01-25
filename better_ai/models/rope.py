import torch
import torch.nn as nn
from typing import Tuple

class RoPECache(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: int = 10000, device: torch.device = None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.device = device

        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self._cache = self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int):
        t = torch.arange(max_seq_len, device=self.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.unsqueeze(0).unsqueeze(0) # [1, 1, max_seq_len, dim]

    def to(self, device, **kwargs):
        super().to(device, **kwargs)
        if self.device != device:
            self.device = device
            self.inv_freq = self.inv_freq.to(device)
            self._cache = self._build_cache(self.max_seq_len)
        return self

    def _apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x_embed = (x * cos) + (self._rotate_half(x) * sin)
        return x_embed

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        if seq_len > self.max_seq_len:
            self._cache = self._build_cache(seq_len)

        cache = self._cache[:, :, :seq_len, :].to(q.device)
        cos = cache.cos()
        sin = cache.sin()

        q_rope = self._apply_rotary_emb(q, cos, sin)
        k_rope = self._apply_rotary_emb(k, cos, sin)
        return q_rope, k_rope
