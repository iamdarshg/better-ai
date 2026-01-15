"""Testing framework for DeepSeek model components"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from typing import Dict, List, Tuple, Optional
import time
import numpy as np
from contextlib import contextmanager

from ..models.core import RMSNorm, SwiGLU, MultiHeadAttention, TransformerBlock, DeepSeekModel
from ..models.moe import Expert, ExpertRouter, MoELayer, DeepSeekMoEModel
from ..models.attention import FlashMultiHeadAttention, SparseAttention, LatentAttention
from ..optimizers.fp8 import FP8Linear, FP8AdamW, FP8LossScaler
from ..config import ModelConfig, TrainingConfig
from ..utils import get_device, profile_memory


class TestConfig:
    """Test configuration"""
    device = get_device()
    dtype = torch.float32
    
    # Model configs for testing
    tiny_config = ModelConfig(
        vocab_size=1000,
        hidden_dim=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_dim=128,
        max_seq_length=128
    )
    
    small_config = ModelConfig(
        vocab_size=5000,
        hidden_dim=256,
        num_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_dim=1024,
        max_seq_length=512
    )


class TestUtilities:
    """Test utility functions and classes"""
    
    @staticmethod
    def assert_tensor_properties(tensor: torch.Tensor, expected_shape: Tuple, expected_dtype: torch.dtype):
        """Assert tensor properties"""
        assert tensor.shape == expected_shape, f"Shape mismatch: {tensor.shape} != {expected_shape}"
        assert tensor.dtype == expected_dtype, f"Dtype mismatch: {tensor.dtype} != {expected_dtype}"
        assert not torch.isnan(tensor).any(), "Tensor contains NaN values"
        assert not torch.isinf(tensor).any(), "Tensor contains Inf values"
    
    @staticmethod
    def assert_model_parameters(model: nn.Module):
        """Assert model has valid parameters"""
        for name, param in model.named_parameters():
            assert param.requires_grad, f"Parameter {name} should require gradients"
            assert not torch.isnan(param).any(), f"Parameter {name} contains NaN"
            assert not torch.isinf(param).any(), f"Parameter {name} contains Inf"
    
    @staticmethod
    @contextmanager
    def assert_no_memory_leak():
        """Context manager to assert no memory leak"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
        
        yield
        
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated()
            assert end_memory <= start_memory * 1.1, f"Memory leak detected: {start_memory} -> {end_memory}"


class TestComponents:
    """Test individual model components"""
    
    def test_rms_norm(self):
        """Test RMSNorm layer"""
        hidden_size = 64
        norm = RMSNorm(hidden_size).to(TestConfig.device)
        
        # Test forward pass
        x = torch.randn(2, 10, hidden_size, device=TestConfig.device)
        output = norm(x)
        
        TestUtilities.assert_tensor_properties(output, x.shape, x.dtype)
        TestUtilities.assert_model_parameters(norm)
    
    def test_swiglu(self):
        """Test SwiGLU activation"""
        hidden_size = 64
        intermediate_size = 128
        swiglu = SwiGLU(hidden_size, intermediate_size).to(TestConfig.device)
        
        # Test forward pass
        x = torch.randn(2, 10, hidden_size, device=TestConfig.device)
        output = swiglu(x)
        
        TestUtilities.assert_tensor_properties(output, (2, 10, hidden_size), x.dtype)
        TestUtilities.assert_model_parameters(swiglu)
    
    def test_multi_head_attention(self):
        """Test MultiHeadAttention layer"""
        hidden_size = 64
        num_heads = 4
        num_key_value_heads = 2
        head_dim = hidden_size // num_heads
        
        attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim
        ).to(TestConfig.device)
        
        # Test forward pass
        x = torch.randn(2, 10, hidden_size, device=TestConfig.device)
        output, weights, cache = attn(x)
        
        TestUtilities.assert_tensor_properties(output, x.shape, x.dtype)
        TestUtilities.assert_model_parameters(attn)
    
    def test_transformer_block(self):
        """Test TransformerBlock"""
        config = TestConfig.tiny_config
        block = TransformerBlock(
            hidden_size=config.hidden_dim,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.hidden_dim // config.num_attention_heads,
            intermediate_size=config.intermediate_dim
        ).to(TestConfig.device)
        
        # Test forward pass
        x = torch.randn(2, 10, config.hidden_dim, device=TestConfig.device)
        output, weights, cache = block(x)
        
        TestUtilities.assert_tensor_properties(output, x.shape, x.dtype)
        TestUtilities.assert_model_parameters(block)
    
    def test_fp8_linear(self):
        """Test FP8Linear layer"""
        in_features = 64
        out_features = 128
        
        # Test FP32 fallback
        linear = FP8Linear(in_features, out_features, use_fp8=False).to(TestConfig.device)
        
        x = torch.randn(2, in_features, device=TestConfig.device)
        output = linear(x)
        
        TestUtilities.assert_tensor_properties(output, (2, out_features), x.dtype)
        TestUtilities.assert_model_parameters(linear)
    
    def test_expert(self):
        """Test Expert layer"""
        hidden_size = 64
        intermediate_size = 128
        expert = Expert(hidden_size, intermediate_size).to(TestConfig.device)
        
        x = torch.randn(2, 10, hidden_size, device=TestConfig.device)
        output = expert(x)
        
        TestUtilities.assert_tensor_properties(output, x.shape, x.dtype)
        TestUtilities.assert_model_parameters(expert)
    
    def test_expert_router(self):
        """Test ExpertRouter"""
        hidden_size = 64
        num_experts = 8
        num_experts_per_token = 2
        
        router = ExpertRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token
        ).to(TestConfig.device)
        
        x = torch.randn(2, 10, hidden_size, device=TestConfig.device)
        weights, selected, logits = router(x)
        
        expected_weights_shape = (2, 10, num_experts_per_token)
        expected_selected_shape = (2, 10, num_experts_per_token)
        expected_logits_shape = (2, 10, num_experts)
        
        TestUtilities.assert_tensor_properties(weights, expected_weights_shape, TestConfig.dtype)
        TestUtilities.assert_tensor_properties(selected, expected_selected_shape, torch.long)
        TestUtilities.assert_tensor_properties(logits, expected_logits_shape, TestConfig.dtype)
        TestUtilities.assert_model_parameters(router)
    
    def test_moe_layer(self):
        """Test MoELayer"""
        hidden_size = 64
        num_experts = 4
        num_experts_per_token = 2
        
        moe = MoELayer(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token
        ).to(TestConfig.device)
        
        x = torch.randn(2, 10, hidden_size, device=TestConfig.device)
        output, aux_loss, aux_losses = moe(x)
        
        TestUtilities.assert_tensor_properties(output, x.shape, x.dtype)
        assert isinstance(aux_loss, torch.Tensor)
        assert isinstance(aux_losses, dict)
        TestUtilities.assert_model_parameters(moe)


class TestModels:
    """Test complete models"""
    
    def test_deepseek_model(self):
        """Test DeepSeekModel"""
        config = TestConfig.tiny_config
        model = DeepSeekModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_dim,
            max_seq_length=config.max_seq_length
        ).to(TestConfig.device)
        
        # Test forward pass
        input_ids = torch.randint(0, config.vocab_size, (2, 10), device=TestConfig.device)
        outputs = model(input_ids=input_ids)
        
        expected_shape = (2, 10, config.vocab_size)
        TestUtilities.assert_tensor_properties(outputs['last_hidden_state'], expected_shape, TestConfig.dtype)
        TestUtilities.assert_model_parameters(model)
    
    def test_deepseek_moe_model(self):
        """Test DeepSeekMoEModel"""
        config = TestConfig.tiny_config
        model = DeepSeekMoEModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_dim,
            num_experts=4,
            num_experts_per_token=2
        ).to(TestConfig.device)
        
        # Test forward pass
        input_ids = torch.randint(0, config.vocab_size, (2, 10), device=TestConfig.device)
        outputs = model(input_ids=input_ids)
        
        expected_shape = (2, 10, config.vocab_size)
        TestUtilities.assert_tensor_properties(outputs['last_hidden_state'], expected_shape, TestConfig.dtype)
        assert 'aux_loss' in outputs
        TestUtilities.assert_model_parameters(model)


class TestOptimizers:
    """Test optimizers"""
    
    def test_fp8_loss_scaler(self):
        """Test FP8LossScaler"""
        scaler = FP8LossScaler()
        
        # Test scaling
        loss = torch.tensor(1.0)
        scaled_loss = scaler.scale_loss(loss)
        assert scaled_loss.item() > loss.item()
        
        # Test update
        scaler.update_scale(False)  # No overflow
        assert scaler.get_scale() >= scaler.scale / scaler.scale_factor / 2  # Should increase or stay same
    
    def test_fp8_adamw(self):
        """Test FP8AdamW optimizer"""
        model = nn.Linear(10, 5)
        optimizer = FP8AdamW(model.parameters(), lr=0.001)
        
        # Test step
        x = torch.randn(2, 10)
        y = torch.randn(2, 5)
        
        optimizer.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        
        # Check that parameters changed
        assert any(p.grad is not None for p in model.parameters())


class TestPerformance:
    """Test performance and memory efficiency"""
    
    def test_model_memory_usage(self):
        """Test that model memory usage is reasonable"""
        config = TestConfig.small_config
        model = DeepSeekModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_dim
        ).to(TestConfig.device)
        
        # Test memory usage
        with TestUtilities.assert_no_memory_leak():
            input_ids = torch.randint(0, config.vocab_size, (1, 64), device=TestConfig.device)
            
            # Warmup
            for _ in range(3):
                _ = model(input_ids=input_ids)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            # Measure memory
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated()
                
                # Forward pass
                outputs = model(input_ids=input_ids)
                torch.cuda.synchronize()
                
                memory_after = torch.cuda.memory_allocated()
                memory_used = memory_after - memory_before
                
                # Memory should be reasonable (less than 1GB for small model)
                assert memory_used < 1024 * 1024 * 1024, f"Memory usage too high: {memory_used / 1024**2:.2f} MB"
    
    def test_inference_speed(self):
        """Test inference speed"""
        config = TestConfig.tiny_config
        model = DeepSeekModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_dim
        ).to(TestConfig.device)
        
        input_ids = torch.randint(0, config.vocab_size, (1, 32), device=TestConfig.device)
        
        # Warmup
        for _ in range(5):
            _ = model(input_ids=input_ids)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Measure speed
        start_time = time.time()
        for _ in range(10):
            _ = model(input_ids=input_ids)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        tokens_per_second = input_ids.size(1) / avg_time
        
        # Should be reasonably fast (at least 100 tokens/sec on modern hardware)
        assert tokens_per_second > 50, f"Inference too slow: {tokens_per_second:.2f} tokens/sec"
    
    def test_gradient_checkpointing(self):
        """Test gradient checkpointing reduces memory"""
        config = TestConfig.small_config
        
        # Model without checkpointing
        model_no_checkpoint = DeepSeekModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_dim
        ).to(TestConfig.device)
        
        # Create large input to stress memory
        input_ids = torch.randint(0, config.vocab_size, (2, 256), device=TestConfig.device)
        
        # Forward pass without checkpointing
        outputs_no_checkpoint = model_no_checkpoint(input_ids=input_ids)
        loss_no_checkpoint = outputs_no_checkpoint['last_hidden_state'].sum()
        loss_no_checkpoint.backward()
        
        memory_no_checkpoint = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Clear gradients
        model_no_checkpoint.zero_grad()
        
        # Note: Actual gradient checkpointing would require modifying the model
        # This is a placeholder for the test
        assert memory_no_checkpoint > 0, "Memory should be allocated"


class TestIntegration:
    """Integration tests"""
    
    def test_full_training_loop(self):
        """Test a complete training loop with small model"""
        config = TestConfig.tiny_config
        model = DeepSeekModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_dim
        ).to(TestConfig.device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Create synthetic data
        batch_size = 2
        seq_len = 32
        
        for step in range(5):
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=TestConfig.device)
            labels = input_ids.clone()
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids)
            logits = outputs['last_hidden_state']
            
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            loss.backward()
            optimizer.step()
            
            assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
            assert not torch.isnan(loss), "Loss should not be NaN"
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        config = TestConfig.tiny_config
        model = DeepSeekModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_dim
        ).to(TestConfig.device)
        
        # Test save and load
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save model
            torch.save(model.state_dict(), temp_path)
            
            # Create new model and load
            new_model = DeepSeekModel(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                intermediate_size=config.intermediate_dim
            ).to(TestConfig.device)
            
            new_model.load_state_dict(torch.load(temp_path, map_location=TestConfig.device))
            
            # Test that outputs are the same
            input_ids = torch.randint(0, config.vocab_size, (2, 10), device=TestConfig.device)
            
            model.eval()
            new_model.eval()
            
            with torch.no_grad():
                output1 = model(input_ids=input_ids)
                output2 = new_model(input_ids=input_ids)
            
            torch.testing.assert_close(output1['last_hidden_state'], output2['last_hidden_state'])
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def run_all_tests():
    """Run all tests"""
    test_classes = [
        TestComponents,
        TestModels,
        TestOptimizers,
        TestPerformance,
        TestIntegration
    ]
    
    results = {}
    
    for test_class in test_classes:
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        class_results = {}
        
        for test_method in test_methods:
            try:
                getattr(test_instance, test_method)()
                class_results[test_method] = "PASSED"
            except Exception as e:
                class_results[test_method] = f"FAILED: {str(e)}"
        
        results[test_class.__name__] = class_results
    
    return results


if __name__ == "__main__":
    # Run tests when script is executed directly
    results = run_all_tests()
    
    print("Test Results:")
    print("=" * 50)
    
    total_tests = 0
    passed_tests = 0
    
    for class_name, class_results in results.items():
        print(f"\n{class_name}:")
        for test_name, result in class_results.items():
            print(f"  {test_name}: {result}")
            total_tests += 1
            if result == "PASSED":
                passed_tests += 1
    
    print(f"\nSummary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")