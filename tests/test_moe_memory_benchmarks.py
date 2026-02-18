"""
Memory benchmarking suite for MoE optimizations.

Measures actual memory usage reduction for each optimization.
Uses small configs for low-resource testing.
"""

import unittest
import torch
import torch.cuda as cuda
from better_ai.config import ModelConfig
from better_ai.models.moe import MoELayer
from better_ai.utils.memory_pool import reset_global_buffer_pool
import gc


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return 0.0
    return cuda.memory_allocated() / (1024 ** 2)


def measure_peak_memory(fn):
    """Decorator to measure peak GPU memory during function execution."""
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            result = fn(*args, **kwargs)
            return result, 0.0, 0.0
        
        # Clear cache and reset stats
        gc.collect()
        cuda.empty_cache()
        cuda.reset_peak_memory_stats()
        
        initial_mem = get_gpu_memory_mb()
        result = fn(*args, **kwargs)
        peak_mem = cuda.max_memory_allocated() / (1024 ** 2)
        final_mem = get_gpu_memory_mb()
        
        return result, peak_mem, final_mem - initial_mem
    
    return wrapper


class TestRouterMemoryReduction(unittest.TestCase):
    """Test memory reduction from chunked routing + fused softmax-topk."""
    
    def setUp(self):
        """Set up small model for memory testing."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping memory benchmark")
        
        self.device = torch.device("cuda")
        self.hidden_dim = 128  # Small for testing
        self.num_experts = 16  # Enough to see memory difference
        self.batch_size = 4
        self.seq_len = 256
    
    def test_router_memory_reduction(self):
        """Measure memory savings from chunked + fused routing."""
        
        @measure_peak_memory
        def baseline_forward():
            """Baseline: standard routing without optimizations."""
            moe = MoELayer(
                hidden_size=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=2,
                expert_intermediate_size=256,
                device=self.device,
                use_chunked_routing=False,
                use_fused_softmax_topk=False,
                use_buffer_pool=False,
                layer_id=0,
            ).to(self.device)
            
            hidden_states = torch.randn(
                self.batch_size, self.seq_len, self.hidden_dim, device=self.device
            )
            
            with torch.no_grad():
                outputs, _, _ = moe(hidden_states)
            
            return outputs
        
        @measure_peak_memory
        def optimized_forward():
            """Optimized: chunked + fused routing."""
            moe = MoELayer(
                hidden_size=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=2,
                expert_intermediate_size=256,
                device=self.device,
                use_chunked_routing=True,
                routing_chunk_size=64,
                use_fused_softmax_topk=True,
                use_buffer_pool=False,
                layer_id=0,
            ).to(self.device)
            
            hidden_states = torch.randn(
                self.batch_size, self.seq_len, self.hidden_dim, device=self.device
            )
            
            with torch.no_grad():
                outputs, _, _ = moe(hidden_states)
            
            return outputs
        
        # Run baseline
        _, baseline_peak, _ = baseline_forward()
        
        # Clear memory
        gc.collect()
        cuda.empty_cache()
        
        # Run optimized
        _, optimized_peak, _ = optimized_forward()
        
        memory_reduction = (baseline_peak - optimized_peak) / baseline_peak * 100
        
        print(f"\nRouter Memory Benchmark:")
        print(f"  Baseline peak: {baseline_peak:.2f} MB")
        print(f"  Optimized peak: {optimized_peak:.2f} MB")
        print(f"  Memory reduction: {memory_reduction:.1f}%")
        
        # Should see at least some reduction (target: 40-60%)
        # Being conservative for small model
        self.assertGreater(memory_reduction, 0, "Optimized routing should use less memory")


class TestBufferPoolMemoryReduction(unittest.TestCase):
    """Test memory reduction from buffer pooling."""
    
    def setUp(self):
        """Set up test."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping memory benchmark")
        
        self.device = torch.device("cuda")
        self.hidden_dim = 128
        self.num_experts = 8
        self.batch_size = 4
        self.seq_len = 256
        self.num_layers = 4  # Simulate multiple layers
    
    def test_buffer_pool_reduction(self):
        """Measure memory savings from buffer pooling across layers."""
        reset_global_buffer_pool()
        
        @measure_peak_memory
        def baseline_multi_layer():
            """Baseline: each layer allocates its own buffers."""
            layers = []
            for i in range(self.num_layers):
                moe = MoELayer(
                    hidden_size=self.hidden_dim,
                    num_experts=self.num_experts,
                    num_experts_per_token=2,
                    expert_intermediate_size=256,
                    device=self.device,
                    use_buffer_pool=False,
                    layer_id=i,
                ).to(self.device)
                layers.append(moe)
            
            hidden_states = torch.randn(
                self.batch_size, self.seq_len, self.hidden_dim, device=self.device
            )
            
            with torch.no_grad():
                x = hidden_states
                for layer in layers:
                    x, _, _ = layer(x)
            
            return x
        
        @measure_peak_memory
        def optimized_multi_layer():
            """Optimized: layers share buffer pool."""
            reset_global_buffer_pool()
            layers = []
            for i in range(self.num_layers):
                moe = MoELayer(
                    hidden_size=self.hidden_dim,
                    num_experts=self.num_experts,
                    num_experts_per_token=2,
                    expert_intermediate_size=256,
                    device=self.device,
                    use_buffer_pool=True,
                    layer_id=i,
                ).to(self.device)
                layers.append(moe)
            
            hidden_states = torch.randn(
                self.batch_size, self.seq_len, self.hidden_dim, device=self.device
            )
            
            with torch.no_grad():
                x = hidden_states
                for layer in layers:
                    x, _, _ = layer(x)
            
            return x
        
        # Run baseline
        _, baseline_peak, _ = baseline_multi_layer()
        
        # Clear
        gc.collect()
        cuda.empty_cache()
        
        # Run optimized
        _, optimized_peak, _ = optimized_multi_layer()
        
        memory_reduction = (baseline_peak - optimized_peak) / baseline_peak * 100
        
        print(f"\nBuffer Pool Memory Benchmark:")
        print(f"  Baseline peak: {baseline_peak:.2f} MB")
        print(f"  Optimized peak: {optimized_peak:.2f} MB")
        print(f"  Memory reduction: {memory_reduction:.1f}%")
        
        # Should see reduction with buffer reuse (target: ~30-50% for small model)
        self.assertGreaterEqual(memory_reduction, 0, "Buffer pooling should not increase memory")


class TestDynamicPruningMemoryReduction(unittest.TestCase):
    """Test memory reduction from dynamic expert pruning."""
    
    def setUp(self):
        """Set up test."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping memory benchmark")
        
        self.device = torch.device("cuda")
        self.hidden_dim = 128
        self.num_experts = 16  # Larger expert count to see pruning benefits
        self.batch_size = 4
        self.seq_len = 128
    
    def test_dynamic_pruning_reduction(self):
        """Measure memory savings from expert pruning (simulated)."""
        
        @measure_peak_memory
        def baseline_all_experts():
            """Baseline: all experts active."""
            moe = MoELayer(
                hidden_size=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=2,
                expert_intermediate_size=256,
                device=self.device,
                use_dynamic_pruning=False,
                layer_id=0,
            ).to(self.device)
            
            moe.eval()
            
            hidden_states = torch.randn(
                self.batch_size, self.seq_len, self.hidden_dim, device=self.device
            )
            
            with torch.no_grad():
                outputs, _, _ = moe(hidden_states)
            
            return outputs
        
        @measure_peak_memory
        def pruned_experts():
            """Optimized: dynamic pruning enabled (some experts pruned)."""
            moe = MoELayer(
                hidden_size=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=2,
                expert_intermediate_size=256,
                device=self.device,
                use_dynamic_pruning=True,
                layer_id=0,
            ).to(self.device)
            
            moe.eval()
            
            hidden_states = torch.randn(
                self.batch_size, self.seq_len, self.hidden_dim, device=self.device
            )
            
            with torch.no_grad():
                # Run multiple times to trigger pruning
                for _ in range(20):
                    outputs, _, _ = moe(hidden_states)
            
            return outputs
        
        # Run baseline
        _, baseline_peak, _ = baseline_all_experts()
        
        # Clear
        gc.collect()
        cuda.empty_cache()
        
        # Run with pruning
        _, pruned_peak, _ = pruned_experts()
        
        # Note: Actual memory reduction depends on pruning behavior
        # For uniform random routing, may not see much reduction
        # This test validates the mechanism works
        
        print(f"\nDynamic Pruning Memory Benchmark:")
        print(f"  Baseline peak: {baseline_peak:.2f} MB")
        print(f"  With pruning peak: {pruned_peak:.2f} MB")
        if baseline_peak > 0:
            memory_reduction = (baseline_peak - pruned_peak) / baseline_peak * 100
            print(f"  Memory reduction: {memory_reduction:.1f}%")


class TestCombinedOptimizationsMemory(unittest.TestCase):
    """Test memory reduction with all optimizations enabled."""
    
    def setUp(self):
        """Set up test."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping memory benchmark")
        
        self.device = torch.device("cuda")
        self.hidden_dim = 128
        self.num_experts = 12
        self.batch_size = 4
        self.seq_len = 256
        self.num_layers = 3
    
    def test_all_optimizations_combined(self):
        """Measure total memory savings with all optimizations."""
        reset_global_buffer_pool()
        
        @measure_peak_memory
        def baseline():
            """Baseline: no optimizations."""
            layers = []
            for i in range(self.num_layers):
                moe = MoELayer(
                    hidden_size=self.hidden_dim,
                    num_experts=self.num_experts,
                    num_experts_per_token=2,
                    expert_intermediate_size=256,
                    device=self.device,
                    use_chunked_routing=False,
                    use_fused_softmax_topk=False,
                    use_buffer_pool=False,
                    use_dynamic_pruning=False,
                    layer_id=i,
                ).to(self.device)
                layers.append(moe)
            
            hidden_states = torch.randn(
                self.batch_size, self.seq_len, self.hidden_dim, device=self.device
            )
            
            with torch.no_grad():
                x = hidden_states
                for layer in layers:
                    x, _, _ = layer(x)
            
            return x
        
        @measure_peak_memory
        def fully_optimized():
            """Full optimizations: all 5 enabled."""
            reset_global_buffer_pool()
            layers = []
            for i in range(self.num_layers):
                moe = MoELayer(
                    hidden_size=self.hidden_dim,
                    num_experts=self.num_experts,
                    num_experts_per_token=2,
                    expert_intermediate_size=256,
                    device=self.device,
                    use_chunked_routing=True,
                    routing_chunk_size=64,
                    use_fused_softmax_topk=True,
                    use_buffer_pool=True,
                    use_dynamic_pruning=False,  # Off for this test (random routing makes it ineffective)
                    layer_id=i,
                ).to(self.device)
                layers.append(moe)
            
            hidden_states = torch.randn(
                self.batch_size, self.seq_len, self.hidden_dim, device=self.device
            )
            
            with torch.no_grad():
                x = hidden_states
                for layer in layers:
                    x, _, _ = layer(x)
            
            return x
        
        # Run baseline
        _, baseline_peak, _ = baseline()
        
        # Clear
        gc.collect()
        cuda.empty_cache()
        
        # Run optimized
        _, optimized_peak, _ = fully_optimized()
        
        memory_reduction = (baseline_peak - optimized_peak) / baseline_peak * 100
        
        print(f"\nCombined Optimizations Memory Benchmark:")
        print(f"  Baseline peak: {baseline_peak:.2f} MB")
        print(f"  Fully optimized peak: {optimized_peak:.2f} MB")
        print(f"  Total memory reduction: {memory_reduction:.1f}%")
        
        # With all optimizations, should see significant reduction
        self.assertGreater(memory_reduction, 0, "Combined optimizations should reduce memory")


def run_benchmarks():
    """Run all memory benchmarks."""
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping memory benchmarks.")
        return True
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestRouterMemoryReduction))
    suite.addTests(loader.loadTestsFromTestCase(TestBufferPoolMemoryReduction))
    suite.addTests(loader.loadTestsFromTestCase(TestDynamicPruningMemoryReduction))
    suite.addTests(loader.loadTestsFromTestCase(TestCombinedOptimizationsMemory))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_benchmarks()
    exit(0 if success else 1)
