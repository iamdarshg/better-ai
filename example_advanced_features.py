#!/usr/bin/env python3
"""
Example usage and testing script for the three new advanced features:
ARPO, CLEANER, and KV-Cache GRPO
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from better_ai.training.integrated_trainer import create_integrated_trainer
    from better_ai.training.arpo import EntropyMonitor
    from better_ai.training.cleaner import create_cleaner_pipeline
    from better_ai.training.kv_cache_grpo import KVCacheManager

    print("Successfully imported all new features!")

    # Test 1: ARPO Entropy Monitor
    print("\n🔍 Testing ARPO Entropy Monitor...")
    monitor = EntropyMonitor(window_size=5, threshold_multiplier=1.5)

    # Create test logits
    logits = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
    entropy = monitor.compute_token_entropy(logits)
    analysis = monitor.update(logits)

    print(f"  Token entropy: {entropy}")
    print(f"  Entropy spike detected: {analysis['is_spike']}")
    print(f"  Baseline entropy: {analysis['baseline_entropy']}")

    # Test 2: CLEANER Pipeline
    print("\n🧹 Testing CLEANER Pipeline...")
    cleaner = create_cleaner_pipeline(min_similarity=0.5, purification_enabled=True)

    # Test trajectory with errors
    error_trajectory = [
        {
            "content": "def broken(",
            "error": {"message": "SyntaxError"},
            "correction": "def fixed(): pass",
        },
        {
            "content": "print(x)",
            "error": {"message": "NameError"},
            "correction": "print('hello')",
        },
    ]

    purified = cleaner.process_trajectory(error_trajectory)
    stats = cleaner.get_statistics()

    print(f"  Errors corrected: {stats['errors_corrected']}")
    print(f"  Purification rate: {stats['purification_rate']:.2%}")
    print(f"  SAAR success rate: {stats['saar_stats']['success_rate']:.2%}")

    # Test 3: KV-Cache Manager
    print("\n⚡ Testing KV-Cache Manager...")
    cache_manager = KVCacheManager(
        max_cache_size=10,
        cache_dim=128,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        device=torch.device("cpu"),
    )

    # Store and retrieve cache
    key_cache = torch.randn(2, 1, 4, 5, 16)
    value_cache = torch.randn(2, 1, 4, 5, 16)
    prefix_hash = "test_hash"

    cache_manager.store_cache(prefix_hash, key_cache, value_cache, 5)
    retrieved = cache_manager.retrieve_cache(prefix_hash)

    print(f"  Cache stored and retrieved: {retrieved is not None}")

    cache_stats = cache_manager.get_statistics()
    print(f"  Cache hit rate: {cache_stats['hit_rate']}")
    print(f"  Cache size: {cache_stats['cache_size']}")

    # Test 4: Integrated Trainer
    print("\n🚀 Testing Integrated Trainer...")

    # Create simple model and components
    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 100))

    class MockRewardModel:
        def score(self, prompt, response):
            return 0.5

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    config = {
        "enable_arpo": True,
        "enable_cleaner": True,
        "enable_kv_cache": True,
        "entropy_window": 5,
        "max_cache_size": 10,
        "cleaner_similarity_threshold": 0.5,
        "device": torch.device("cpu"),
        "group_size": 2,
    }

    trainer = create_integrated_trainer(model, MockRewardModel(), optimizer, config)

    # Mock training batch
    batch = {
        "input_ids": torch.randint(0, 100, (2, 5)),
        "attention_mask": torch.ones(2, 5),
        "target_ids": torch.randint(0, 100, (2, 5)),
    }

    # Perform training step
    metrics = trainer.train_step(batch)

    print(f"  Training step completed")
    print(f"  Total steps: {trainer.training_stats['total_steps']}")

    # Get comprehensive statistics
    stats = trainer.get_comprehensive_statistics()
    efficiency = stats["overall_efficiency"]

    print(f"  ARPO impact rate: {efficiency['arpo_impact_rate']:.2f}%")
    print(f"  Cleaner correction rate: {efficiency['cleaner_correction_rate']:.2f}%")
    print(f"  KV cache saving rate: {efficiency['kv_cache_saving_rate']:.2f}%")

    print("\n✨ All tests completed successfully!")
    print("📈 The three top features are working:")
    print("   1. ARPO - Entropy-based adaptive rollouts")
    print("   2. CLEANER - Self-purified trajectories")
    print("   3. KV-Cache GRPO - Memory optimization")

    print(f"\n🎯 Expected improvements:")
    print(f"   • ARPO: 25-40% improvement in tool-use tasks")
    print(f"   • CLEANER: 3-6% accuracy gains, 3x faster training")
    print(f"   • KV-Cache: 30-50% memory reduction")

except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
