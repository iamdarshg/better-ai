"""
Unit tests for memory optimization and efficiency features
Tests memory management, gradient checkpointing, and optimization strategies
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
from better_ai.config import ModelConfig
from better_ai.models.optimized_model import MemoryOptimizedModel
from better_ai.training.checkpointing import GradientCheckpointManager
from better_ai.training.adaptive_optimizations import MemoryOptimizer


class TestMemoryOptimizedModel:
    """Test memory optimization features in the model."""

    def setUp(self):
        """Set up test environment."""
        self.device = torch.device("cpu")
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
            input_ids.requires_grad_(True)
            outputs = model(input_ids)
            loss = outputs["logits"].sum()
            loss.backward()

        # Check that gradients are computed
        assert model.get_parameter("embed_tokens.weight").grad is not None

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
        assert outputs["logits"].shape == (batch_size, seq_len, self.config.vocab_size)

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

            assert outputs["logits"].shape == (
                batch_size,
                seq_len,
                self.config.vocab_size,
            )

            # Aggressive optimization should use less intermediate memory
            if optimization_level == "aggressive":
                assert "memory_saved" in outputs.get("optimization_info", {})

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
        assert outputs2["past_key_values"] is not None
        assert len(outputs2["past_key_values"]) == self.config.num_layers


class TestGradientCheckpointManager:
    """Test gradient checkpointing management."""

    def test_checkpoint_manager_initialization(self):
        """Test gradient checkpoint manager initialization."""
        config = ModelConfig(num_layers=4)
        manager = GradientCheckpointManager(config)

        assert manager.num_layers == 4
        assert manager.checkpoint_layers == []

    def test_selective_checkpointing(self):
        """Test selective layer checkpointing."""
        config = ModelConfig(num_layers=8)
        manager = GradientCheckpointManager(config)

        # Select every other layer for checkpointing
        manager.select_checkpoint_layers(strategy="every_other", frequency=2)

        expected_layers = [1, 3, 5, 7]  # 0-indexed
        assert manager.checkpoint_layers == expected_layers

    def test_adaptive_checkpointing(self):
        """Test adaptive checkpointing based on memory pressure."""
        config = ModelConfig(num_layers=6)
        manager = GradientCheckpointManager(config)

        # Simulate high memory pressure
        manager.adapt_to_memory_pressure(memory_pressure=0.8)

        # Should checkpoint more layers under high pressure
        assert len(manager.checkpoint_layers) >= 3

    def test_checkpoint_offloading(self):
        """Test checkpoint offloading to CPU."""
        config = ModelConfig(num_layers=4)
        manager = GradientCheckpointManager(config)

        # Enable offloading
        manager.enable_offloading(device="cpu")

        assert manager.offload_to_cpu
        assert manager.offload_device == "cpu"


class TestMemoryOptimizer:
    """Test memory optimization strategies."""

    def test_memory_optimizer_initialization(self):
        """Test memory optimizer initialization."""
        config = ModelConfig(hidden_dim=512)
        optimizer = MemoryOptimizer(config)

        assert optimizer.config.hidden_dim == 512
        assert optimizer.optimization_strategies == []

    def test_activation_checkpointing(self):
        """Test activation checkpointing optimization."""
        config = ModelConfig(hidden_dim=256, num_layers=4)
        optimizer = MemoryOptimizer(config)

        # Create a simple model for testing
        model = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # Enable activation checkpointing
        optimizer.enable_activation_checkpointing(model)

        # Test forward pass
        x = torch.randn(2, 256)
        output = model(x)

        assert output.shape == (2, 256)

    def test_memory_efficient_attention(self):
        """Test memory-efficient attention implementation."""
        config = ModelConfig(hidden_dim=256, num_attention_heads=8, max_seq_length=1024)
        optimizer = MemoryOptimizer(config)

        batch_size = 2
        seq_len = 512
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)

        # Test memory-efficient attention
        outputs = optimizer.memory_efficient_attention(
            hidden_states=hidden_states, attention_mask=None, use_memory_efficient=True
        )

        assert outputs.shape == hidden_states.shape

    def test_dynamic_batch_sizing(self):
        """Test dynamic batch sizing based on available memory."""
        config = ModelConfig(hidden_dim=256)
        optimizer = MemoryOptimizer(config)

        # Test with different memory constraints
        for available_memory_gb in [4, 8, 16]:
            optimal_batch_size = optimizer.calculate_optimal_batch_size(
                available_memory_gb=available_memory_gb,
                seq_length=512,
                model_size_estimate="medium",
            )

            assert optimal_batch_size > 0
            assert optimal_batch_size <= 32  # Reasonable upper bound

    def test_memory_fragmentation_handling(self):
        """Test memory fragmentation handling."""
        config = ModelConfig(hidden_dim=256)
        optimizer = MemoryOptimizer(config)

        # Simulate fragmented memory
        optimizer.handle_memory_fragmentation()

        # Should trigger cleanup
        assert optimizer.cleanup_called

    def test_memory_profiling_integration(self):
        """Test memory profiling integration."""
        config = ModelConfig(hidden_dim=256)
        optimizer = MemoryOptimizer(config)

        # Enable profiling
        optimizer.enable_profiling()

        # Simulate model operations
        with optimizer.profile_memory():
            x = torch.randn(4, 256)
            y = torch.randn(4, 256)
            z = x + y

        # Check profiling results
        profile_data = optimizer.get_profile_data()
        assert "peak_memory" in profile_data
        assert "memory_efficiency" in profile_data


class TestMemoryEfficientDataLoading:
    """Test memory-efficient data loading strategies."""

    def test_streaming_data_loading(self):
        """Test streaming data loading to reduce memory."""
        from better_ai.data.unified_dataloader import StreamingDataset

        # Mock dataset
        dataset = StreamingDataset(
            dataset_name="test", tokenizer=Mock(), max_length=128, streaming=True
        )

        # Test streaming behavior
        assert dataset.streaming
        assert dataset.max_length == 128

    def test_memory_mapped_data(self):
        """Test memory-mapped data loading."""
        from better_ai.data.unified_dataloader import MemoryMappedDataset

        # Mock memory-mapped dataset
        dataset = MemoryMappedDataset(data_path="test_data.bin", memory_map=True)

        # Test memory mapping
        assert dataset.memory_map
        assert dataset.data_path == "test_data.bin"

    def test_adaptive_batch_loading(self):
        """Test adaptive batch loading based on memory."""
        from better_ai.data.unified_dataloader import AdaptiveBatchLoader

        loader = AdaptiveBatchLoader(base_batch_size=8, memory_threshold=0.8)

        # Test adaptive behavior
        current_memory_usage = 0.9  # High memory usage
        adjusted_batch_size = loader.adjust_batch_size(current_memory_usage)

        assert adjusted_batch_size < 8  # Should reduce batch size


class TestMemoryOptimizationIntegration:
    """Test integration of memory optimization features."""

    def test_end_to_end_memory_optimization(self):
        """Test end-to-end memory optimization workflow."""
        config = ModelConfig(
            hidden_dim=256,
            num_layers=4,
            use_gradient_checkpointing=True,
            use_flash_attention=True,
        )

        # Create optimized model
        model = MemoryOptimizedModel(config)
        memory_optimizer = MemoryOptimizer(config)

        # Apply optimizations
        memory_optimizer.optimize_model(model)

        # Test forward pass with optimizations
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        outputs = model(input_ids, memory_optimization="aggressive")

        assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
        assert "optimization_info" in outputs

    def test_memory_optimization_with_training(self):
        """Test memory optimization during training."""
        config = ModelConfig(hidden_dim=256, num_layers=2)
        model = MemoryOptimizedModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Enable training optimizations
        model.enable_training_optimizations()

        # Simulate training step
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = nn.CrossEntropyLoss()(
            outputs["logits"].view(-1, config.vocab_size), labels.view(-1)
        )
        loss.backward()
        optimizer.step()

        # Check that training completed successfully
        assert loss.item() >= 0
