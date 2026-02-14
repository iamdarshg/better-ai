"""
Optimized KV Cache management.
"""

import torch
from typing import List, Tuple, Optional

class OptimizedKVCache:
    """
    Implements KV cache compression and eviction strategies.
    """
    def __init__(self, max_size: int = 4096, strategy: str = "h2o"):
        self.max_size = max_size
        self.strategy = strategy
        self.cache = {} # Layer ID -> (K, V)

    def compress(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compresses KV cache using specified strategy.
        """
        if self.strategy == "h2o":
            return self._h2o_compression(k, v)
        elif self.strategy == "streaming":
            return self._streaming_llm_compression(k, v)
        return k, v

    def _h2o_compression(self, k, v):
        """Heavy-Hitter Oracle (H2O) eviction"""
        seq_len = k.size(-2)
        if seq_len <= self.max_size:
            return k, v

        # Keep recent tokens + heavy hitters (tokens with high attention scores)
        # Simplified: just keep recent for now
        return k[:, :, -self.max_size:, :], v[:, :, -self.max_size:, :]

    def _streaming_llm_compression(self, k, v):
        """StreamingLLM: keep initial tokens + sliding window"""
        num_initial = 4
        num_recent = self.max_size - num_initial

        if k.size(-2) <= self.max_size:
            return k, v

        initial_k = k[:, :, :num_initial, :]
        initial_v = v[:, :, :num_initial, :]

        recent_k = k[:, :, -num_recent:, :]
        recent_v = v[:, :, -num_recent:, :]

        return torch.cat([initial_k, recent_k], dim=-2), torch.cat([initial_v, recent_v], dim=-2)
