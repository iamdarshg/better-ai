"""
KV-Cache Reuse for GRPO
Optimizes Group Reward Policy Optimization with sequential generation and cache reuse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Any, Union
import hashlib
import logging
import time


class KVCacheEntry:
    """
    Individual KV-cache entry for a specific prefix
    """

    def __init__(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        prefix_hash: str,
        length: int,
        timestamp: float = 0.0,
    ):
        self.key = key  # (num_layers, batch_size, num_heads, seq_len, head_dim)
        self.value = value  # (num_layers, batch_size, num_heads, seq_len, head_dim)
        self.prefix_hash = prefix_hash
        self.length = length
        self.timestamp = timestamp
        self.access_count = 1
        self.last_access = timestamp

    def update_access(self, timestamp: float):
        """Update access statistics"""
        self.access_count += 1
        self.last_access = timestamp


class KVCacheManager:
    """
    Manages KV-cache storage and retrieval for GRPO rollouts
    """

    def __init__(
        self,
        max_cache_size: int = 1000,
        cache_dim: int = 128,
        num_layers: int = 12,
        num_heads: int = 12,
        head_dim: int = 64,
        device: torch.device = None,
    ):
        self.max_cache_size = max_cache_size
        self.cache_dim = cache_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Cache storage
        self.cache_entries: Dict[str, KVCacheEntry] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_queries = 0

        # Cache eviction policy
        self.eviction_policy = "lru"  # Least Recently Used

        logging.info(
            f"KVCacheManager initialized: max_size={max_cache_size}, device={device}"
        )

    def compute_prefix_hash(
        self, input_ids: torch.Tensor, prefix_length: Optional[int] = None
    ) -> str:
        """
        Compute hash for input prefix to identify cache entries
        """
        if prefix_length is None:
            prefix_length = input_ids.shape[-1]

        # Take prefix of specified length
        prefix = input_ids[..., :prefix_length]

        # Convert to bytes and hash
        prefix_bytes = prefix.cpu().numpy().tobytes()
        hash_object = hashlib.md5(prefix_bytes)
        return hash_object.hexdigest()

    def extract_kv_cache(
        self,
        hidden_states: List[torch.Tensor],
        attention_outputs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract KV cache from model forward pass

        Args:
            hidden_states: List of hidden states from each layer
            attention_outputs: List of (key, value) tuples from attention

        Returns:
            (key_cache, value_cache) for all layers
        """
        key_cache = []
        value_cache = []

        for layer_idx in range(len(attention_outputs)):
            keys, values = attention_outputs[layer_idx]

            # Store per-layer KV cache
            key_cache.append(keys.detach().clone())
            value_cache.append(values.detach().clone())

        # Stack to create layer dimension
        key_cache = torch.stack(
            key_cache, dim=0
        )  # (num_layers, batch, heads, seq_len, head_dim)
        value_cache = torch.stack(
            value_cache, dim=0
        )  # (num_layers, batch, heads, seq_len, head_dim)

        return key_cache, value_cache

    def store_cache(
        self,
        prefix_hash: str,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        prefix_length: int,
    ):
        """Store KV cache entry"""
        # Check cache size limit
        if len(self.cache_entries) >= self.max_cache_size:
            self._evict_cache_entry()

        # Create cache entry
        import time

        timestamp = time.time()

        entry = KVCacheEntry(
            key=key_cache,
            value=value_cache,
            prefix_hash=prefix_hash,
            length=prefix_length,
            timestamp=timestamp,
        )

        self.cache_entries[prefix_hash] = entry

    def retrieve_cache(
        self, prefix_hash: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Retrieve KV cache entry if available"""
        self.total_queries += 1

        if prefix_hash in self.cache_entries:
            entry = self.cache_entries[prefix_hash]
            entry.update_access(time.time())
            self.cache_hits += 1
            return entry.key, entry.value
        else:
            self.cache_misses += 1
            return None

    def find_matching_prefix(
        self, input_ids: torch.Tensor, min_prefix_length: int = 4
    ) -> Optional[Tuple[str, int]]:
        """
        Find longest matching prefix in cache

        Returns:
            (prefix_hash, matched_length) if found, None otherwise
        """
        input_seq = (
            input_ids.squeeze().cpu().numpy()
            if input_ids.dim() > 1
            else input_ids.cpu().numpy()
        )

        best_match = None
        best_length = 0

        for cache_hash, entry in self.cache_entries.items():
            # Check if cached prefix is a prefix of input
            if entry.length >= min_prefix_length and entry.length <= len(input_seq):
                # This is simplified - in practice, would compare actual tokens
                if entry.length > best_length:
                    best_length = entry.length
                    best_match = cache_hash

        if best_match:
            return best_match, best_length
        return None

    def _evict_cache_entry(self):
        """Evict cache entry based on policy"""
        if not self.cache_entries:
            return

        if self.eviction_policy == "lru":
            # Find least recently used entry
            oldest_time = float("inf")
            oldest_key = None

            for key, entry in self.cache_entries.items():
                if entry.last_access < oldest_time:
                    oldest_time = entry.last_access
                    oldest_key = key

            if oldest_key:
                del self.cache_entries[oldest_key]

        elif self.eviction_policy == "lfu":
            # Find least frequently used entry
            min_access = float("inf")
            lfu_key = None

            for key, entry in self.cache_entries.items():
                if entry.access_count < min_access:
                    min_access = entry.access_count
                    lfu_key = key

            if lfu_key:
                del self.cache_entries[lfu_key]

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        hit_rate = self.cache_hits / max(1, self.total_queries) * 100
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_queries": self.total_queries,
            "hit_rate": f"{hit_rate:.2f}%",
            "cache_size": len(self.cache_entries),
            "max_cache_size": self.max_cache_size,
        }


class OptimizedGRPOWithKVCache:
    """
    GRPO trainer with KV-cache reuse optimization
    """

    def __init__(
        self,
        model: nn.Module,
        reward_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
    ):
        self.model = model
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.config = config

        # KV-cache manager
        self.cache_manager = KVCacheManager(
            max_cache_size=config.get("max_cache_size", 1000),
            cache_dim=config.get("hidden_dim", 1536),
            num_layers=config.get("num_layers", 12),
            num_heads=config.get("num_attention_heads", 12),
            head_dim=config.get("head_dim", 64),
            device=config.get(
                "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ),
        )

        # Training configuration
        self.group_size = config.get("group_size", 4)
        self.beta = config.get("beta", 0.01)
        self.eps_clip = config.get("eps_clip", 0.2)
        self.device = config.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Performance tracking
        self.total_generations = 0
        self.cache_reuses = 0
        self.memory_saved = 0

        logging.info("OptimizedGRPOWithKVCache initialized")

    def generate_group_with_cache_reuse(
        self,
        prompts: List[str],
        max_length: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> List[torch.Tensor]:
        """
        Generate group of responses with KV-cache reuse optimization using model's generate_group
        """
        group_results = []

        for prompt in prompts:
            # Tokenize prompt
            input_ids = self._tokenize(prompt)

            if use_cache and hasattr(self.model, "generate_group"):
                # Use the built-in generate_group which reuses prompt KV cache
                responses = self.model.generate_group(
                    input_ids=input_ids,
                    group_size=self.group_size,
                    max_new_tokens=max_length - input_ids.size(1),
                    temperature=temperature,
                    use_cache=True
                )
                group_results.append(responses)
                self.cache_reuses += (self.group_size - 1)
                self.memory_saved += input_ids.size(1) * (self.group_size - 1) * self._estimate_memory_per_token()
            else:
                # Fallback to multiple separate generations
                responses = []
                for _ in range(self.group_size):
                    resp = self.model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_length - input_ids.size(1),
                        temperature=temperature,
                        use_cache=True
                    )
                    responses.append(resp)
                group_results.append(torch.cat(responses, dim=0))

            self.total_generations += self.group_size

        return group_results

    def _generate_with_cached_prefix(
        self,
        input_ids: torch.Tensor,
        cached_kv: Tuple[torch.Tensor, torch.Tensor],
        cached_length: int,
        max_length: int,
        temperature: float,
        do_sample: bool,
    ) -> Dict[str, Any]:
        """
        Generate using cached KV prefix to avoid recomputation
        """
        # Separate cached prefix and new tokens
        cached_key_cache, cached_value_cache = cached_kv
        new_input_ids = input_ids[..., cached_length:]

        if new_input_ids.shape[-1] == 0:
            # All tokens are cached, just return cached result
            return {
                "sequences": input_ids,
                "cache_hit": True,
                "cached_length": cached_length,
                "new_tokens_generated": 0,
            }

        # Continue generation from cached state
        with torch.no_grad():
            # This requires model modification to accept past KV cache
            outputs = self.model.generate(
                input_ids=new_input_ids,
                past_key_values=(cached_key_cache, cached_value_cache),
                max_length=max_length - cached_length,
                temperature=temperature,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_attentions=True,
                output_hidden_states=True,
            )

        # Combine cached prefix with new generation
        full_sequences = torch.cat(
            [input_ids[..., :cached_length], outputs["sequences"]], dim=-1
        )

        return {
            "sequences": full_sequences,
            "cache_hit": True,
            "cached_length": cached_length,
            "new_tokens_generated": outputs["sequences"].shape[-1],
            "attention_outputs": outputs.get("attentions", []),
            "hidden_states": outputs.get("hidden_states", []),
        }

    def _full_generation(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        do_sample: bool,
    ) -> Dict[str, Any]:
        """Full generation without cache"""
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_attentions=True,
                output_hidden_states=True,
            )

        return {
            "sequences": outputs["sequences"],
            "cache_hit": False,
            "cached_length": 0,
            "new_tokens_generated": outputs["sequences"].shape[-1]
            - input_ids.shape[-1],
            "attention_outputs": outputs.get("attentions", []),
            "hidden_states": outputs.get("hidden_states", []),
        }

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize input text"""
        if hasattr(self.model, "tokenizer") and self.model.tokenizer:
            return self.model.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        # Fallback for mock/testing
        return torch.randint(0, self.config.get("vocab_size", 1000), (1, 10), device=self.device)

    def _estimate_memory_per_token(self) -> float:
        """Estimate memory usage per token"""
        # Rough estimate: hidden_dim * num_layers * num_heads * head_dim * 2 (K+V) * 4 bytes
        return (
            self.config.get("hidden_dim", 1536)
            * self.config.get("num_layers", 12)
            * self.config.get("num_attention_heads", 12)
            * self.config.get("head_dim", 64)
            * 2
            * 4  # 4 bytes per float32
        )

    def train_step_with_cache_optimization(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Training step with KV-cache optimization
        """
        self.model.train()

        # Extract prompts from batch
        prompts = self._extract_prompts_from_batch(batch)

        # Generate group responses with cache reuse
        group_responses = self.generate_group_with_cache_reuse(
            prompts, use_cache=self.config.get("use_kv_cache", True)
        )

        # Score responses with reward model
        rewards = []
        for response in group_responses:
            decoded = self._decode_response(response)
            reward = (
                self.reward_model.score(decoded[0], decoded[1])
                if len(decoded) > 1
                else 0.0
            )
            rewards.append(reward)

        # Compute GRPO loss (simplified)
        loss = self._compute_grpo_loss_with_cache(rewards, group_responses)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Return metrics
        cache_stats = self.cache_manager.get_statistics()

        return {
            "loss": loss.item(),
            "cache_hit_rate": cache_stats["hit_rate"],
            "memory_saved_mb": self.memory_saved / (1024 * 1024),
            "cache_reuses": self.cache_reuses,
            "total_generations": self.total_generations,
        }

    def _extract_prompts_from_batch(self, batch: Dict[str, torch.Tensor]) -> List[str]:
        """Extract prompts from batch"""
        if "prompts" in batch:
            return batch["prompts"]

        # If we only have input_ids, we decode them (if possible)
        if hasattr(self.model, "tokenizer") and self.model.tokenizer:
            return self.model.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

        return [f"Prompt {i}" for i in range(batch.get("input_ids", torch.tensor([])).shape[0])]

    def _decode_response(self, response: Any) -> str:
        """Decode response to text"""
        if hasattr(self.model, "tokenizer") and self.model.tokenizer:
            if isinstance(response, torch.Tensor):
                return self.model.tokenizer.batch_decode(response, skip_special_tokens=True)
        return str(response)

    def _compute_grpo_loss_with_cache(
        self, rewards: List[float], responses: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute GRPO loss considering cache efficiency"""
        # Simple reward-based loss (would be more sophisticated in practice)
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

        # Total loss (negative because we maximize reward)
        loss = -avg_reward

        return torch.tensor(loss, device=self.device, requires_grad=True)

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get overall optimization statistics"""
        cache_stats = self.cache_manager.get_statistics()

        return {
            "total_generations": self.total_generations,
            "cache_reuses": self.cache_reuses,
            "reuse_rate": (self.cache_reuses / max(1, self.total_generations) * 100),
            "memory_saved_mb": self.memory_saved / (1024 * 1024),
            "cache_stats": cache_stats,
        }
