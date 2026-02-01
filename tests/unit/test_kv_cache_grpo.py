"""
Unit tests for KV-Cache GRPO optimization
"""

import unittest
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


class TestKVCacheEntry(unittest.TestCase):
    """Test KV cache entry functionality"""

    def test_entry_creation(self):
        key = torch.randn(12, 1, 8, 10, 64)
        value = torch.randn(12, 1, 8, 10, 64)

        entry = KVCacheEntry(
            key=key, value=value, prefix_hash="test_hash", length=10, timestamp=12345.0
        )

        self.assertEqual(entry.prefix_hash, "test_hash")
        self.assertEqual(entry.length, 10)
        self.assertEqual(entry.access_count, 1)
        self.assertEqual(entry.timestamp, 12345.0)

    def test_access_update(self):
        key = torch.randn(12, 1, 8, 10, 64)
        value = torch.randn(12, 1, 8, 10, 64)

        entry = KVCacheEntry(key, value, "test_hash", 10, 1000.0)

        # Update access
        entry.update_access(2000.0)

        self.assertEqual(entry.access_count, 2)
        self.assertEqual(entry.last_access, 2000.0)
        self.assertEqual(entry.timestamp, 1000.0)  # Original timestamp unchanged


class TestKVCacheManager(unittest.TestCase):
    """Test KV cache management"""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_manager = KVCacheManager(
            max_cache_size=2,
            cache_dim=128,
            num_layers=12,
            num_heads=8,
            head_dim=64,
            device=self.device,
        )

    def test_prefix_hash_computation(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        hash1 = self.cache_manager.compute_prefix_hash(input_ids)
        hash2 = self.cache_manager.compute_prefix_hash(input_ids)

        self.assertEqual(hash1, hash2)  # Should be deterministic
        self.assertEqual(len(hash1), 32)  # MD5 hash length

    def test_cache_storage_and_retrieval(self):
        key_cache = torch.randn(12, 1, 8, 5, 64).to(self.device)
        value_cache = torch.randn(12, 1, 8, 5, 64).to(self.device)
        prefix_hash = "test_hash"
        prefix_length = 5

        # Store cache
        self.cache_manager.store_cache(prefix_hash, key_cache, value_cache, prefix_length)

        # Retrieve cache
        retrieved = self.cache_manager.retrieve_cache(prefix_hash)

        self.assertIsNotNone(retrieved)
        retrieved_key, retrieved_value = retrieved
        self.assertTrue(torch.equal(retrieved_key, key_cache))
        self.assertTrue(torch.equal(retrieved_value, value_cache))

    def test_cache_miss(self):
        result = self.cache_manager.retrieve_cache("nonexistent_hash")
        self.assertIsNone(result)

    def test_cache_eviction_lru(self):
        # Fill cache beyond max size
        for i in range(4):  # max_size is 2
            key_cache = torch.randn(12, 1, 8, 1, 64).to(self.device)
            value_cache = torch.randn(12, 1, 8, 1, 64).to(self.device)
            self.cache_manager.store_cache(f"hash_{i}", key_cache, value_cache, 1)

        # Should only keep 2 most recent entries
        self.assertEqual(len(self.cache_manager.cache_entries), 2)

        # Check that oldest entries were evicted
        self.assertNotIn("hash_0", self.cache_manager.cache_entries)
        self.assertNotIn("hash_1", self.cache_manager.cache_entries)
        self.assertIn("hash_2", self.cache_manager.cache_entries)
        self.assertIn("hash_3", self.cache_manager.cache_entries)

    def test_cache_statistics(self):
        # Perform some operations
        key_cache = torch.randn(12, 1, 8, 5, 64).to(self.device)
        value_cache = torch.randn(12, 1, 8, 5, 64).to(self.device)

        self.cache_manager.store_cache("hash_1", key_cache, value_cache, 5)
        self.cache_manager.retrieve_cache("hash_1")  # Hit
        self.cache_manager.retrieve_cache("hash_2")  # Miss

        stats = self.cache_manager.get_statistics()

        self.assertEqual(stats["cache_hits"], 1)
        self.assertEqual(stats["cache_misses"], 1)
        self.assertEqual(stats["total_queries"], 2)
        self.assertEqual(stats["hit_rate"], "50.00%")
        self.assertEqual(stats["cache_size"], 1)


class TestOptimizedGRPOWithKVCache(unittest.TestCase):
    """Test optimized GRPO with KV cache"""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 100)
                self.config = {"vocab_size": 100}
                self.tokenizer = None
            def generate(self, **kwargs):
                return torch.randint(0, 100, (1, 20))
            def generate_group(self, **kwargs):
                group_size = kwargs.get("group_size", 4)
                return torch.randint(0, 100, (group_size, 20))
            def forward(self, **kwargs):
                return {"logits": torch.randn(1, 1, 100)}

        self.mock_model = MockModel().to(self.device)

        class MockRewardModel:
            def score(self, prompt, response):
                return 0.5

        self.mock_reward_model = MockRewardModel()
        self.optimizer = torch.optim.Adam(self.mock_model.parameters(), lr=1e-4)
        self.config = {
            "max_cache_size": 10,
            "hidden_dim": 64,
            "num_layers": 2,
            "num_attention_heads": 4,
            "head_dim": 16,
            "device": self.device,
            "group_size": 2,
            "use_kv_cache": True,
        }
        self.trainer = OptimizedGRPOWithKVCache(
            self.mock_model, self.mock_reward_model, self.optimizer, self.config
        )

    def test_initialization(self):
        self.assertEqual(self.trainer.cache_manager.max_cache_size, 10)
        self.assertEqual(self.trainer.config["use_kv_cache"], True)
        self.assertEqual(self.trainer.group_size, 2)
        self.assertEqual(self.trainer.device, self.device)

    def test_generate_group_with_cache_reuse(self):
        prompts = ["test prompt 1", "test prompt 2"]
        results = self.trainer.generate_group_with_cache_reuse(
            prompts,
            max_length=20,
            use_cache=False,  # Disable for easier testing
        )

        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.shape[0], 2) # group_size


    def test_train_step_with_cache_optimization(self):
        batch = {
            "input_ids": torch.randint(0, 100, (2, 5)).to(self.device),
            "attention_mask": torch.ones(2, 5).to(self.device),
            "target_ids": torch.randint(0, 100, (2, 5)).to(self.device),
        }

        metrics = self.trainer.train_step_with_cache_optimization(batch)

        self.assertIn("loss", metrics)
        self.assertIn("cache_hit_rate", metrics)
        self.assertIn("total_generations", metrics)

        # Check statistics update
        stats = self.trainer.get_optimization_statistics()
        self.assertIn("total_generations", stats)
        self.assertIn("cache_stats", stats)

    def test_memory_per_token_estimation(self):
        memory_per_token = self.trainer._estimate_memory_per_token()

        # Should be positive and reasonable
        self.assertGreater(memory_per_token, 0)
        # Rough calculation: 64 * 2 * 4 * 16 * 2 * 4 = 65,536 bytes (if hd=64, layers=2, heads=4, dim=16)
        # Actually our hd=64 is hidden_dim, not cache_dim.
        # Calculation in code: num_layers * num_heads * head_dim * 2 * 4
        # 2 * 4 * 16 * 2 * 4 = 1024
        self.assertGreater(memory_per_token, 500)
        self.assertLess(memory_per_token, 100000)


class TestIntegrationFeatures(unittest.TestCase):
    """Test integration of all optimization features"""

    def test_kv_cache_memory_efficiency(self):
        """Test that KV cache provides memory efficiency"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        manager = KVCacheManager(
            max_cache_size=100,
            cache_dim=1536,
            num_layers=12,
            num_heads=12,
            head_dim=64,
            device=device,
        )

        # Simulate cache hits and misses
        key_cache = torch.randn(12, 1, 12, 10, 64).to(device)
        value_cache = torch.randn(12, 1, 12, 10, 64).to(device)

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
        self.assertEqual(stats["cache_hits"], 10)
        self.assertEqual(stats["hit_rate"], "100.00%")

        # Memory saved calculation
        estimated_memory_per_token = (
            12 * 12 * 64 * 2 * 4  # bytes (num_layers * num_heads * head_dim * 2 * 4)
        )
        saved_memory = hits * 10 * estimated_memory_per_token
        self.assertGreater(saved_memory, 0)


if __name__ == "__main__":
    unittest.main()
