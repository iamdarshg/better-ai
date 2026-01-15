"""Selective Gradient Checkpointing for Memory-Efficient MoE Training"""

import torch
import torch.nn as nn
from typing import Callable, Optional, Dict, List, Tuple, Any
import time
import gc
from contextlib import contextmanager


class SelectiveCheckpointManager:
    """
    Memory-aware selective gradient checkpointing for MoE models
    Decides which layers to checkpoint based on memory pressure and layer importance
    """
    
    def __init__(
        self,
        memory_threshold: float = 0.7,
        checkpoint_frequency: int = 2,  # Every N layers
        checkpoint_large_layers: bool = True,
        layer_size_threshold: float = 0.05,  # 5% of total model size
        adaptive_checkpointing: bool = True,
        device: torch.device = torch.device('cpu')
    ):
        self.memory_threshold = memory_threshold
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_large_layers = checkpoint_large_layers
        self.layer_size_threshold = layer_size_threshold
        self.adaptive_checkpointing = adaptive_checkpointing
        self.device = device
        
        # Tracking
        self.layer_memory_usage = {}
        self.layer_sizes = {}
        self.checkpoint_decisions = {}
        self.memory_history = []
        self.current_memory_pressure = 0.0
        
        # Performance tracking
        self.checkpoint_times = []
        self.recomputation_times = []
        
    def analyze_model_memory(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model memory usage patterns"""
        total_params = sum(p.numel() for p in model.parameters())
        layer_sizes = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'parameters'):
                layer_params = sum(p.numel() for p in module.parameters())
                layer_sizes[name] = layer_params / total_params
        
        self.layer_sizes = layer_sizes
        return {
            'total_parameters': total_params,
            'layer_sizes': layer_sizes,
            'large_layers': [name for name, size in layer_sizes.items() if size > self.layer_size_threshold]
        }
    
    def get_memory_pressure(self) -> float:
        """Get current memory pressure (0.0 to 1.0)"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            
            current_pressure = allocated / total
            self.current_memory_pressure = current_pressure
            self.memory_history.append(current_pressure)
            
            return min(current_pressure, 1.0)
        return 0.0
    
    def should_checkpoint_layer(
        self,
        layer_name: str,
        layer_idx: int,
        force_checkpoint: bool = False
    ) -> bool:
        """Decide whether to checkpoint a specific layer"""
        
        # Force checkpoint if requested
        if force_checkpoint:
            return True
        
        # Check memory pressure
        memory_pressure = self.get_memory_pressure()
        
        # High memory pressure - checkpoint more aggressively
        if memory_pressure > self.memory_threshold:
            if self.checkpoint_large_layers and layer_name in self.layer_sizes:
                return self.layer_sizes[layer_name] > self.layer_size_threshold
            return True
        
        # Adaptive checkpointing based on patterns
        if self.adaptive_checkpointing:
            # Checkpoint every Nth layer
            if layer_idx % self.checkpoint_frequency == 0:
                return True
            
            # Checkpoint large layers under moderate pressure
            if (memory_pressure > 0.5 and 
                self.checkpoint_large_layers and 
                layer_name in self.layer_sizes and
                self.layer_sizes[layer_name] > self.layer_size_threshold):
                return True
        
        # Default decision
        return False
    
    @contextmanager
    def checkpoint_context(
        self,
        layer_name: str,
        layer_idx: int,
        use_checkpoint: Optional[bool] = None
    ):
        """Context manager for selective checkpointing"""
        
        # Decide whether to checkpoint
        if use_checkpoint is None:
            should_checkpoint = self.should_checkpoint_layer(layer_name, layer_idx)
        else:
            should_checkpoint = use_checkpoint
        
        self.checkpoint_decisions[layer_name] = should_checkpoint
        
        if should_checkpoint:
            # Measure checkpointing overhead
            start_time = time.time()
            
            try:
                # Use gradient checkpointing
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    result = yield
            finally:
                checkpoint_time = time.time() - start_time
                self.checkpoint_times.append(checkpoint_time)
                
                # Log decision
                memory_pressure = self.get_memory_pressure()
                if len(self.checkpoint_times) % 50 == 0:  # Log every 50 checkpoints
                    print(f"🔄 Checkpoint {layer_name} (mem: {memory_pressure:.2f}, time: {checkpoint_time:.3f}s)")
        else:
            # No checkpointing
            start_time = time.time()
            try:
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    result = yield
            finally:
                forward_time = time.time() - start_time
                if layer_idx % 10 == 0:  # Log occasionally
                    memory_pressure = self.get_memory_pressure()
                    print(f"⚡ Forward {layer_name} (mem: {memory_pressure:.2f}, time: {forward_time:.3f}s)")
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get checkpointing statistics"""
        checkpoint_ratio = 0.0
        if self.checkpoint_decisions:
            checkpointed = sum(1 for v in self.checkpoint_decisions.values() if v)
            checkpoint_ratio = checkpointed / len(self.checkpoint_decisions)
        
        return {
            'checkpoint_ratio': checkpoint_ratio,
            'total_checkpoints': len(self.checkpoint_times),
            'avg_checkpoint_time': sum(self.checkpoint_times) / len(self.checkpoint_times) if self.checkpoint_times else 0,
            'current_memory_pressure': self.current_memory_pressure,
            'memory_pressure_trend': self.memory_history[-10:] if len(self.memory_history) > 0 else [],
            'layer_checkpoint_decisions': self.checkpoint_decisions.copy()
        }


def selective_checkpoint(
    checkpoint_manager: SelectiveCheckpointManager,
    layer_name: str,
    layer_idx: int
):
    """Decorator for selective gradient checkpointing"""
    
    def decorator(function: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Check if we should checkpoint this function
            with checkpoint_manager.checkpoint_context(layer_name, layer_idx):
                if checkpoint_manager.checkpoint_decisions.get(layer_name, False):
                    # Use torch.utils.checkpoint
                    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
                else:
                    # Normal forward pass
                    return function(*args, **kwargs)
        return wrapper
    
    return decorator


class MemoryEfficientMoELayer(nn.Module):
    """
    MoE layer with selective gradient checkpointing and memory optimization
    """
    
    def __init__(
        self,
        original_moe_layer: nn.Module,
        checkpoint_manager: SelectiveCheckpointManager,
        layer_idx: int,
        layer_name: str = "moe_layer"
    ):
        super().__init__()
        self.original_moe_layer = original_moe_layer
        self.checkpoint_manager = checkpoint_manager
        self.layer_idx = layer_idx
        self.layer_name = layer_name
        
        # Wrap expert forward with selective checkpointing
        self._setup_checkpointing()
    
    def _setup_checkpointing(self):
        """Setup selective checkpointing for experts"""
        if hasattr(self.original_moe_layer, 'experts'):
            # Wrap each expert with selective checkpointing
            for i, expert in enumerate(self.original_moe_layer.experts):
                expert_name = f"{self.layer_name}_expert_{i}"
                self.original_moe_layer.experts[i] = self._wrap_expert(
                    expert, expert_name, i
                )
    
    def _wrap_expert(self, expert: nn.Module, expert_name: str, expert_idx: int) -> nn.Module:
        """Wrap individual expert with checkpointing"""
        
        class CheckpointedExpert(nn.Module):
            def __init__(self, original_expert, checkpoint_mgr, name, idx):
                super().__init__()
                self.original_expert = original_expert
                self.checkpoint_mgr = checkpoint_mgr
                self.name = name
                self.idx = idx
            
            def forward(self, x):
                # Decide whether to checkpoint this expert
                should_checkpoint = self.checkpoint_mgr.should_checkpoint_layer(
                    self.name, self.idx
                )
                
                if should_checkpoint:
                    return torch.utils.checkpoint.checkpoint(
                        self.original_expert.forward, x
                    )
                else:
                    return self.original_expert.forward(x)
        
        return CheckpointedExpert(expert, self.checkpoint_manager, expert_name, expert_idx)
    
    def forward(self, *args, **kwargs):
        """Forward pass with selective checkpointing"""
        # Use context manager for the entire MoE layer
        with self.checkpoint_manager.checkpoint_context(
            self.layer_name, self.layer_idx
        ):
            if self.checkpoint_manager.checkpoint_decisions.get(self.layer_name, False):
                # Checkpoint the entire MoE layer
                return torch.utils.checkpoint.checkpoint(
                    self.original_moe_layer.forward, *args, **kwargs
                )
            else:
                # Normal forward pass
                return self.original_moe_layer.forward(*args, **kwargs)


class AdaptiveMemoryManager:
    """
    Adaptive memory management for MoE training
    """
    
    def __init__(
        self,
        cleanup_frequency: int = 50,
        memory_target: float = 0.8,  # Target 80% GPU utilization
        enable_dynamic_batching: bool = True
    ):
        self.cleanup_frequency = cleanup_frequency
        self.memory_target = memory_target
        self.enable_dynamic_batching = enable_dynamic_batching
        
        self.step_counter = 0
        self.original_batch_size = None
        self.current_batch_size = None
        
    def step(self, current_loss: float, grad_norm: float):
        """Called after each training step"""
        self.step_counter += 1
        
        # Periodic memory cleanup
        if self.step_counter % self.cleanup_frequency == 0:
            self._cleanup_memory()
        
        # Dynamic batch size adjustment
        if self.enable_dynamic_batching:
            self._adjust_batch_size_if_needed(current_loss, grad_norm)
    
    def _cleanup_memory(self):
        """Perform memory cleanup"""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Log memory status
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            
            print(f"🧹 Memory Cleanup - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def _adjust_batch_size_if_needed(self, loss: float, grad_norm: float):
        """Dynamically adjust batch size based on training conditions"""
        if not torch.cuda.is_available():
            return
        
        memory_pressure = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        
        # If memory pressure is too high, suggest smaller batch size
        if memory_pressure > self.memory_target:
            if self.original_batch_size is None:
                self.original_batch_size = self.current_batch_size or 32
            
            suggested_batch_size = max(1, int(self.current_batch_size * 0.8))
            print(f"⚠️  High memory pressure ({memory_pressure:.2f})")
            print(f"   Consider reducing batch size: {self.current_batch_size} -> {suggested_batch_size}")
        
        # If training is unstable (high gradients), suggest adjustments
        elif grad_norm > 10.0:
            print(f"⚠️  High gradient norm ({grad_norm:.2f})")
            print(f"   Consider reducing learning rate or gradient clipping")
    
    def set_batch_size(self, batch_size: int):
        """Set current batch size"""
        self.current_batch_size = batch_size
        if self.original_batch_size is None:
            self.original_batch_size = batch_size
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        if not torch.cuda.is_available():
            return {}
        
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        
        return {
            'allocated_gb': allocated / 1024**3,
            'reserved_gb': reserved / 1024**3,
            'total_gb': total / 1024**3,
            'utilization': allocated / total,
            'pressure': allocated / total
        }


# Export classes and functions
__all__ = [
    'SelectiveCheckpointManager',
    'selective_checkpoint',
    'MemoryEfficientMoELayer',
    'AdaptiveMemoryManager'
]