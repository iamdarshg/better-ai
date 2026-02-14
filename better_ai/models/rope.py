import torch
import torch.nn as nn
from typing import Tuple, Optional


class RoPECache(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: int = 10000, device: torch.device = None, scaling_factor: float = 1.0, use_yarn: bool = False):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.device = device
        self.scaling_factor = scaling_factor
        self.use_yarn = use_yarn

        self.inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self._cache = self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int):
        # Implement Dynamic NTK scaling for long context
        base = self.base
        if self.scaling_factor > 1.0:
            # Dynamic NTK scaling formula: base * (scaling_factor * seq_len / max_seq_len) ^ (dim / (dim-2))
            # Simplified version for initialization:
            base = base * (self.scaling_factor ** (self.dim / (self.dim - 2)))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(self.device) / self.dim))
            self.inv_freq = inv_freq

        t = torch.arange(max_seq_len, device=self.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, dim]

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

    def get_cos_sin(
        self,
        indices: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin values for arbitrary indices.
        Useful for striped attention where indices are not contiguous.
        """
        max_idx = indices.max().item()
        if max_idx >= self.max_seq_len:
            self._cache = self._build_cache(int(max_idx + 1))
            self.max_seq_len = int(max_idx + 1)

        # Extract relevant part of cache
        # _cache is [1, 1, max_seq_len, dim]
        freqs = self._cache[:, :, indices, :].to(device=device, dtype=dtype)
        return freqs.cos(), freqs.sin()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = q.shape[-2]  # Assuming [B, H, S, D] or [B, S, H, D]

        # Determine dimension and shape
        # Handle both [B, H, S, D] and [B, S, D]
        if q.dim() == 4:
            s_idx = 2
        else:
            s_idx = 1

        needed_len = offset + seq_len
        if needed_len > self.max_seq_len:
            self._cache = self._build_cache(needed_len)
            self.max_seq_len = needed_len

        # Extract relevant part of cache
        cache = self._cache[:, :, offset : offset + seq_len, :].to(
            q.device, dtype=q.dtype
        )
        cos = cache.cos()
        sin = cache.sin()

        q_rope = self._apply_rotary_emb(q, cos, sin)
        k_rope = self._apply_rotary_emb(k, cos, sin)
        return q_rope, k_rope
