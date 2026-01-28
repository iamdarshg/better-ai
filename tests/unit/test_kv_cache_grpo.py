"""
Unit tests for KV-Cache GRPO optimization
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from better_ai.training.kv_cache_grpo import (
    KVCacheEntry,
    KVCacheManager,
    OptimizedGRPOWithKVCache,
)


class TestKVCacheEntry:
    """Test KV cache entry functionality"""

    def test_entry_creation(self):
        key = torch.randn(12, 1, 8, 10, 64)
        value = torch.randn(12, 1, 8, 10, 64)

        entry = KVCacheEntry(
            key=key, value=value, prefix_hash="test_hash", length=10, timestamp=12345.0
        )

        assert entry.prefix_hash == "test_hash"
        assert entry.length == 10
        assert entry.access_count == 1
        assert entry.timestamp == 12345.0

    def test_access_update(self):
        key = torch.randn(12, 1, 8, 10, 64)
        value = torch.randn(12, 1, 8, 10, 64)

        entry = KVCacheEntry(key, value, "test_hash", 10, 1000.0)

        # Update access
        entry.update_access(2000.0)

        assert entry.access_count == 2
        assert entry.last_access == 2000.0
        assert entry.timestamp == 1000.0  # Original timestamp unchanged


class TestKVCacheManager:
    """Test KV cache management"""

    @pytest.fixture
    def cache_manager(self):
        return KVCacheManager(
            max_cache_size=2,
            cache_dim=128,
            num_layers=12,
            num_heads=8,
            head_dim=64,
            device=torch.device("cpu"),
        )

    def test_prefix_hash_computation(self, cache_manager):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        hash1 = cache_manager.compute_prefix_hash(input_ids)
        hash2 = cache_manager.compute_prefix_hash(input_ids)

        assert hash1 == hash2  # Should be deterministic
        assert len(hash1) == 32  # MD5 hash length

    def test_cache_storage_and_retrieval(self, cache_manager):
        key_cache = torch.randn(12, 1, 8, 5, 64)
        value_cache = torch.randn(12, 1, 8, 5, 64)
        prefix_hash = "test_hash"
        prefix_length = 5

        # Store cache
        cache_manager.store_cache(prefix_hash, key_cache, value_cache, prefix_length)

        # Retrieve cache
        retrieved = cache_manager.retrieve_cache(prefix_hash)

        assert retrieved is not None
        retrieved_key, retrieved_value = retrieved
        assert torch.equal(retrieved_key, key_cache)
        assert torch.equal(retrieved_value, value_cache)

    def test_cache_miss(self, cache_manager):
        result = cache_manager.retrieve_cache("nonexistent_hash")
        assert result is None

    def test_cache_eviction_lru(self, cache_manager):
        # Fill cache beyond max size
        for i in range(4):  # max_size is 2
            key_cache = torch.randn(12, 1, 8, 1, 64)
            value_cache = torch.randn(12, 1, 8, 1, 64)
            cache_manager.store_cache(f"hash_{i}", key_cache, value_cache, 1)

        # Should only keep 2 most recent entries
        assert len(cache_manager.cache_entries) == 2

        # Check that oldest entries were evicted
        assert "hash_0" not in cache_manager.cache_entries
        assert "hash_1" not in cache_manager.cache_entries
        assert "hash_2" in cache_manager.cache_entries
        assert "hash_3" in cache_manager.cache_entries

    def test_cache_statistics(self, cache_manager):
        # Perform some operations
        key_cache = torch.randn(12, 1, 8, 5, 64)
        value_cache = torch.randn(12, 1, 8, 5, 64)

        cache_manager.store_cache("hash_1", key_cache, value_cache, 5)
        cache_manager.retrieve_cache("hash_1")  # Hit
        cache_manager.retrieve_cache("hash_2")  # Miss

        stats = cache_manager.get_statistics()

        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["total_queries"] == 2
        assert stats["hit_rate"] == "50.00%"
        assert stats["cache_size"] == 1


class TestOptimizedGRPOWithKVCache:
    """Test optimized GRPO with KV cache"""

    @pytest.fixture
    def mock_model(self):
        return nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 100))

    @pytest.fixture
    def mock_reward_model(self):
        class MockRewardModel:
            def score(self, prompt, response):
                return 0.5

        return MockRewardModel()

    @pytest.fixture
    def optimizer(self, mock_model):
        return torch.optim.Adam(mock_model.parameters(), lr=1e-4)

    @pytest.fixture
    def config(self):
        return {
            "max_cache_size": 10,
            "hidden_dim": 64,
            "num_layers": 2,
            "num_attention_heads": 4,
            "head_dim": 16,
            "device": torch.device("cpu"),
            "group_size": 2,
            "use_kv_cache": True,
        }

    def test_initialization(self, mock_model, mock_reward_model, optimizer, config):
        trainer = OptimizedGRPOWithKVCache(
            mock_model, mock_reward_model, optimizer, config
        )

        assert trainer.cache_manager.max_cache_size == 10
        assert trainer.config["use_kv_cache"] == True
        assert trainer.group_size == 2
        assert trainer.device == torch.device("cpu")

    def test_generate_group_with_cache_reuse(
        self, mock_model, mock_reward_model, optimizer, config
    ):
        trainer = OptimizedGRPOWithKVCache(
            mock_model, mock_reward_model, optimizer, config
        )

        prompts = ["test prompt 1", "test prompt 2"]
        results = trainer.generate_group_with_cache_reuse(
            prompts,
            max_length=20,
            use_cache=False,  # Disable for easier testing
        )

        assert len(results) == 2
        for result in results:
            assert "sequences" in result
            assert "cache_hit" in result
            assert "new_tokens_generated" in result

    def test_generate_with_cached_prefix(
        self, mock_model, mock_reward_model, optimizer, config
    ):
        trainer = OptimizedGRPOWithKVCache(
            mock_model, mock_reward_model, optimizer, config
        )

        # Mock cached KV
        cached_key = torch.randn(2, 1, 4, 5, 16)
        cached_value = torch.randn(2, 1, 4, 5, 16)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        cached_length = 5
        new_input_ids = input_ids[..., cached_length:]

        # Mock model to return specific sequence
        def mock_generate(**kwargs):
            if "past_key_values" in kwargs:
                return {
                    "sequences": torch.tensor([[9, 10, 11]]),
                    "attentions": [],
                    "hidden_states": [],
                }
            else:
                return {
                    "sequences": torch.tensor([[6, 7, 8, 9, 10, 11]]),
                    "attentions": [],
                    "hidden_states": [],
                }

        original_generate = mock_model.generate
        mock_model.generate = mock_generate

        # Test cached generation
        result = trainer._generate_with_cached_prefix(
            input_ids, (cached_key, cached_value), cached_length, 20, 0.7, True
        )

        assert result["cache_hit"] == True
        assert result["cached_length"] == 5
        assert result["new_tokens_generated"] == 3

        # Restore original method
        mock_model.generate = original_generate

    def test_train_step_with_cache_optimization(
        self, mock_model, mock_reward_model, optimizer, config
    ):
        trainer = OptimizedGRPOWithKVCache(
            mock_model, mock_reward_model, optimizer, config
        )

        batch = {
            "input_ids": torch.randint(0, 100, (2, 5)),
            "attention_mask": torch.ones(2, 5),
            "target_ids": torch.randint(0, 100, (2, 5)),
        }

        metrics = trainer.train_step_with_cache_optimization(batch)

        assert "loss" in metrics
        assert "cache_hit_rate" in metrics
        assert "total_generations" in metrics

        # Check statistics update
        stats = trainer.get_optimization_statistics()
        assert "total_generations" in stats
        assert "cache_stats" in stats

    def test_memory_per_token_estimation(
        self, mock_model, mock_reward_model, optimizer, config
    ):
        trainer = OptimizedGRPOWithKVCache(
            mock_model, mock_reward_model, optimizer, config
        )

        memory_per_token = trainer._estimate_memory_per_token()

        # Should be positive and reasonable
        assert memory_per_token > 0
        # Rough calculation: 64 * 2 * 4 * 16 * 2 * 4 = 65,536 bytes
        assert memory_per_token > 50000  # At least this much
        assert memory_per_token < 100000  # But not too high


class TestIntegrationFeatures:
    """Test integration of all optimization features"""

    def test_kv_cache_memory_efficiency(self):
        """Test that KV cache provides memory efficiency"""
        manager = KVCacheManager(
            max_cache_size=100,
            cache_dim=1536,
            num_layers=12,
            num_heads=12,
            head_dim=64,
            device=torch.device("cpu"),
        )

        # Simulate cache hits and misses
        key_cache = torch.randn(12, 1, 12, 10, 64)
        value_cache = torch.randn(12, 1, 12, 10, 64)

        # Store multiple entries
        for i in range(10):
            manager.store_cache(f"hash_{i}", key_cache, value_cache, 10)

        # Perform retrievals
        hits = 0
        for i in range(10):
            result = manager.retrieve_cache(f"hash_{i}")
            if result is not None:
                hits += 1

        stats = manager.get_statistics()
        assert stats["cache_hits"] == 10
        assert stats["hit_rate"] == "100.00%"

        # Memory saved calculation
        estimated_memory_per_token = (
            1536 * 12 * 12 * 64 * 2 * 4  # bytes
        )
        saved_memory = hits * 10 * estimated_memory_per_token
        assert saved_memory > 0


if __name__ == "__main__":
    pytest.main([__file__])
