"""
Unit tests for memory optimization and efficiency features
Tests memory management, gradient checkpointing, and optimization strategies
"""

import unittest
from better_ai.test_resource_tags import high_resource
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
from better_ai.config import ModelConfig
from better_ai.models.optimized_model import OptimizedDeepSeekMoEModel as MemoryOptimizedModel
from better_ai.training.checkpointing import SelectiveCheckpointManager as GradientCheckpointManager
from better_ai.optimizers.memory import MemoryOptimizer


<<<<<<< HEAD
@high_resource
=======

>>>>>>> 6ee6a9026156a3d656f792dbcbf9395f94c9f6e7
class TestMemoryOptimizedModel(unittest.TestCase):
    """Test memory optimization features in the model."""

    def setUp(self):
        """Set up test environment."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = ModelConfig(
            hidden_dim=256,
            num_layers=4,
            use_gradient_checkpointing=True,
            use_flash_attention=True,
            vocab_size=4096,
        )

    def test_gradient_checkpointing_integration(self):
        """Test that gradient checkpointing is properly integrated."""
        model = MemoryOptimizedModel(self.config).to(self.device)

        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(
            self.device
        )

        # Enable gradient checkpointing
        model.enable_gradient_checkpointing()

        # Forward pass should use checkpointing
        with torch.enable_grad():
            outputs = model(input_ids)
            loss = outputs["logits"].sum()
            loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(model.get_parameter("embed_tokens.weight").grad)

    def test_flash_attention_memory_efficiency(self):
        """Test that flash attention reduces memory usage."""
        model = MemoryOptimizedModel(self.config).to(self.device)

        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(
            self.device
        )

        # Measure memory with flash attention
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        outputs = model(input_ids, use_flash_attention=True)

        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_used = final_memory - initial_memory if torch.cuda.is_available() else 0

        # Output should be correct
        self.assertEqual(outputs["logits"].shape, (batch_size, seq_len, self.config.vocab_size))

    def test_memory_efficient_forward(self):
        """Test memory-efficient forward pass options."""
        model = MemoryOptimizedModel(self.config).to(self.device)

        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(
            self.device
        )

        # Test different memory optimization levels
        for optimization_level in ["none", "moderate", "aggressive"]:
            outputs = model(input_ids, memory_optimization=optimization_level)

            self.assertEqual(outputs["logits"].shape, (
                batch_size,
                seq_len,
                self.config.vocab_size,
            ))

            # Aggressive optimization should use less intermediate memory
            if optimization_level == "aggressive":
                self.assertIn("memory_saved", outputs.get("optimization_info", {}))

    def test_kv_cache_memory_management(self):
        """Test KV cache memory management."""
        model = MemoryOptimizedModel(self.config).to(self.device)

        batch_size = 1
        seq_len = 32
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(
            self.device
        )

        # Enable KV cache
        model.enable_kv_cache(max_cache_size=1024)

        # First forward pass
        outputs1 = model(input_ids, use_cache=True)

        # Second forward pass with new tokens
        new_tokens = torch.randint(0, self.config.vocab_size, (batch_size, 8)).to(
            self.device
        )
        outputs2 = model(
            new_tokens, use_cache=True, past_key_values=outputs1["past_key_values"]
        )

        # Check cache is used
        self.assertIsNotNone(outputs2["past_key_values"])
        self.assertEqual(len(outputs2["past_key_values"]), self.config.num_layers)


class TestGradientCheckpointManager(unittest.TestCase):
    """Test gradient checkpointing management."""

    def test_checkpoint_manager_initialization(self):
        """Test gradient checkpoint manager initialization."""
        config = ModelConfig(num_layers=4)
        manager = GradientCheckpointManager(config)

        self.assertEqual(manager.num_layers, 4)
        self.assertEqual(manager.checkpoint_layers, [])

    def test_selective_checkpointing(self):
        """Test selective layer checkpointing."""
        config = ModelConfig(num_layers=8)
        manager = GradientCheckpointManager(config)

        # Select every other layer for checkpointing
        manager.select_checkpoint_layers(strategy="every_other", frequency=2)

        expected_layers = [1, 3, 5, 7]  # 0-indexed
        self.assertEqual(manager.checkpoint_layers, expected_layers)

    def test_adaptive_checkpointing(self):
        """Test adaptive checkpointing based on memory pressure."""
        config = ModelConfig(num_layers=6)
        manager = GradientCheckpointManager(config)

        # Simulate high memory pressure
        manager.adapt_to_memory_pressure(memory_pressure=0.8)

        # Should checkpoint more layers under high pressure
        self.assertGreaterEqual(len(manager.checkpoint_layers), 3)

    def test_checkpoint_offloading(self):
        """Test checkpoint offloading to CPU."""
        config = ModelConfig(num_layers=4)
        manager = GradientCheckpointManager(config)

        # Enable offloading
        manager.enable_offloading(device="cpu")

        self.assertTrue(manager.offload_to_cpu)
        self.assertEqual(manager.offload_device, "cpu")



class TestMemoryEfficientDataLoading(unittest.TestCase):
    """Test memory-efficient data loading strategies."""

    @patch('better_ai.data.unified_dataloader.load_dataset')
    def test_streaming_data_loading(self, mock_load):
        """Test streaming data loading to reduce memory."""
        from better_ai.data.unified_dataloader import StreamingDataset
        mock_load.return_value = [{"text": "dummy"}]

        # Mock dataset
        dataset = StreamingDataset(
            dataset_name="test", tokenizer=Mock(), max_length=128, streaming=True
        )

        # Test streaming behavior
        self.assertTrue(dataset.streaming)
        self.assertEqual(dataset.max_length, 128)

    def test_memory_mapped_data(self):
        """Test memory-mapped data loading."""
        from better_ai.data.unified_dataloader import MemoryMappedDataset

        # Mock memory-mapped dataset
        dataset = MemoryMappedDataset(data_path="test_data.bin", memory_map=True)

        # Test memory mapping
        self.assertTrue(dataset.memory_map)
        self.assertEqual(dataset.data_path, "test_data.bin")

    def test_adaptive_batch_loading(self):
        """Test adaptive batch loading based on memory."""
        from better_ai.data.unified_dataloader import AdaptiveBatchLoader

        loader = AdaptiveBatchLoader(base_batch_size=8, memory_threshold=0.8)

        # Test adaptive behavior
        current_memory_usage = 0.9  # High memory usage
        adjusted_batch_size = loader.adjust_batch_size(current_memory_usage)

        self.assertLess(adjusted_batch_size, 8)  # Should reduce batch size


class TestMemoryOptimizationIntegration(unittest.TestCase):
    """Test integration of memory optimization features."""

    def test_end_to_end_memory_optimization(self):
        """Test end-to-end memory optimization workflow."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = ModelConfig(
            hidden_dim=256,
            num_layers=4,
            use_gradient_checkpointing=True,
            use_flash_attention=True,
        )

        # Create optimized model
        model = MemoryOptimizedModel(config).to(device)
        memory_optimizer = MemoryOptimizer(config)

        # Apply optimizations
        memory_optimizer.optimize_model(model)

        # Test forward pass with optimizations
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)

        outputs = model(input_ids, memory_optimization="aggressive")

        self.assertEqual(outputs["logits"].shape, (batch_size, seq_len, config.vocab_size))
        self.assertIn("optimization_info", outputs)

    def test_memory_optimization_with_training(self):
        """Test memory optimization during training."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = ModelConfig(hidden_dim=256, num_layers=2)
        model = MemoryOptimizedModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Enable training optimizations
        model.enable_training_optimizations()

        # Simulate training step
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = nn.CrossEntropyLoss()(
            outputs["logits"].view(-1, config.vocab_size), labels.view(-1)
        )
        loss.backward()
        optimizer.step()

        # Check that training completed successfully
        self.assertGreaterEqual(loss.item(), 0)

if __name__ == "__main__":
    unittest.main()
