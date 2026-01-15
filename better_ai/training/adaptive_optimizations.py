"""Dynamic Expert Capacity and Adaptive Attention Selection for MoE Training"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
from collections import deque


class DynamicExpertCapacityManager:
    """
    Dynamically adjusts expert capacity based on real-time load patterns
    Optimizes expert utilization and prevents overload/collapse
    """
    
    def __init__(
        self,
        num_experts: int,
        base_capacity_factor: float = 1.25,
        min_capacity_factor: float = 0.5,
        max_capacity_factor: float = 3.0,
        adjustment_frequency: int = 50,
        load_balance_threshold: float = 0.1,
        device: torch.device = torch.device('cpu')
    ):
        self.num_experts = num_experts
        self.base_capacity_factor = base_capacity_factor
        self.min_capacity_factor = min_capacity_factor
        self.max_capacity_factor = max_capacity_factor
        self.adjustment_frequency = adjustment_frequency
        self.load_balance_threshold = load_balance_threshold
        self.device = device
        
        # Dynamic capacity tracking
        self.current_capacity_factors = torch.full((num_experts,), base_capacity_factor, device=device)
        self.expert_load_history = deque(maxlen=200)
        self.expert_overload_history = deque(maxlen=100)
        self.expert_underload_history = deque(maxlen=100)
        
        # Adjustment parameters
        self.overload_penalty = 0.05  # Increase capacity by 5% on overload
        self.underload_penalty = 0.02  # Decrease capacity by 2% on underload
        self.smoothing_factor = 0.8  # Smooth capacity adjustments
        
        # Tracking
        self.adjustment_step = 0
        self.last_adjustment_loads = None
        self.capacity_adjustment_history = deque(maxlen=500)
        
    def update_expert_loads(self, expert_loads: torch.Tensor, expert_overflows: Optional[torch.Tensor] = None):
        """Update expert load tracking and compute capacity adjustments"""
        expert_loads = expert_loads.to(self.device)
        self.expert_load_history.append(expert_loads.cpu())
        
        # Track overflows/underflows
        if expert_overflows is not None:
            overflow_count = expert_overflows.sum().item()
            self.expert_overload_history.append(overflow_count)
            
            underflow_count = (expert_loads < 0.3).sum().item()  # Less than 30% utilization
            self.expert_underload_history.append(underflow_count)
        
        # Check if we should adjust capacity
        self.adjustment_step += 1
        if self.adjustment_step % self.adjustment_frequency == 0:
            self._adjust_capacities()
    
    def _adjust_capacities(self):
        """Adjust expert capacities based on recent load patterns"""
        if len(self.expert_load_history) < 10:
            return
        
        # Get recent load patterns
        recent_loads = torch.stack(list(self.expert_load_history)[-10:])
        avg_loads = recent_loads.mean(dim=0)
        load_variance = recent_loads.var(dim=0)
        
        # Identify overloaded and underloaded experts
        overloaded = avg_loads > (1.0 / self.num_experts + self.load_balance_threshold)
        underloaded = avg_loads < (1.0 / self.num_experts - self.load_balance_threshold)
        
        # Compute capacity adjustments
        adjustments = torch.zeros_like(self.current_capacity_factors)
        
        # Increase capacity for overloaded experts
        if overloaded.any():
            adjustments[overloaded] = 1.0 + self.overload_penalty
        
        # Decrease capacity for underloaded experts  
        if underloaded.any():
            adjustments[underloaded] = 1.0 - self.underload_penalty
        
        # Apply smoothing
        if self.last_adjustment_loads is not None:
            adjustment_diff = adjustments - self.last_adjustment_loads
            adjustments = self.last_adjustment_loads + self.smoothing_factor * adjustment_diff
        
        # Apply adjustments with bounds
        new_capacities = self.current_capacity_factors * adjustments
        new_capacities = torch.clamp(
            new_capacities,
            self.min_capacity_factor,
            self.max_capacity_factor
        )
        
        self.current_capacity_factors = new_capacities
        self.last_adjustment_loads = adjustments
        
        # Store adjustment history
        self.capacity_adjustment_history.append({
            'step': self.adjustment_step,
            'capacities': new_capacities.cpu().numpy().tolist(),
            'avg_loads': avg_loads.cpu().numpy().tolist(),
            'adjustments': adjustments.cpu().numpy().tolist()
        })
        
        # Log significant adjustments
        capacity_change = torch.abs(new_capacities - self.base_capacity_factor).sum()
        if capacity_change > 0.1:  # Significant change
            print(f"🔧 Dynamic Expert Capacity Adjustment (Step {self.adjustment_step}):")
            print(f"   Capacity Range: {new_capacities.min().item():.3f} - {new_capacities.max().item():.3f}")
            print(f"   Overloaded Experts: {overloaded.nonzero(as_tuple=True)[0].tolist()}")
            print(f"   Underloaded Experts: {underloaded.nonzero(as_tuple=True)[0].tolist()}")
    
    def get_expert_capacities(self) -> torch.Tensor:
        """Get current expert capacity factors"""
        return self.current_capacity_factors.clone()
    
    def get_capacity_stats(self) -> Dict[str, Union[float, List[float]]]:
        """Get capacity adjustment statistics"""
        if len(self.capacity_adjustment_history) == 0:
            return {}
        
        recent_adjustments = list(self.capacity_adjustment_history)[-20:]
        avg_capacity = np.mean([adj['capacities'] for adj in recent_adjustments], axis=0)
        capacity_variance = np.var([adj['capacities'] for adj in recent_adjustments], axis=0)
        
        return {
            'current_capacities': self.current_capacity_factors.cpu().numpy().tolist(),
            'base_capacity': self.base_capacity_factor,
            'avg_recent_capacities': avg_capacity.tolist(),
            'capacity_variance': capacity_variance.tolist(),
            'adjustment_frequency': self.adjustment_frequency,
            'total_adjustments': len(self.capacity_adjustment_history)
        }
    
    def reset_capacities(self):
        """Reset all capacities to base values"""
        self.current_capacity_factors.fill_(self.base_capacity_factor)
        self.last_adjustment_loads = None
        self.capacity_adjustment_history.clear()
        print("🔄 Expert capacities reset to base values")


class AdaptiveAttentionSelector:
    """
    Automatically selects optimal attention mechanism based on context and performance
    Switches between Standard, MLA, and DSA based on conditions
    """
    
    def __init__(
        self,
        seq_length_threshold_mla: int = 2048,  # Use MLA for long sequences
        seq_length_threshold_dsa: int = 4096,  # Use DSA for very long sequences
        memory_threshold_mla: float = 0.6,    # Use MLA under memory pressure
        sparsity_threshold_dsa: float = 0.3,  # Use DSA for sparse patterns
        adaptation_frequency: int = 100,
        performance_window: int = 50,
        device: torch.device = torch.device('cpu')
    ):
        self.seq_length_threshold_mla = seq_length_threshold_mla
        self.seq_length_threshold_dsa = seq_length_threshold_dsa
        self.memory_threshold_mla = memory_threshold_mla
        self.sparsity_threshold_dsa = sparsity_threshold_dsa
        self.adaptation_frequency = adaptation_frequency
        self.performance_window = performance_window
        self.device = device
        
        # Attention types
        self.attention_types = ['standard', 'mla', 'dsa']
        self.current_attention_type = 'standard'
        
        # Performance tracking
        self.attention_performance = {
            'standard': deque(maxlen=performance_window),
            'mla': deque(maxlen=performance_window),
            'dsa': deque(maxlen=performance_window)
        }
        
        # Usage patterns
        self.seq_length_history = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)
        self.attention_selection_history = deque(maxlen=200)
        
        # Adaptation parameters
        self.adaptation_step = 0
        self.performance_weights = {'throughput': 0.4, 'memory': 0.3, 'accuracy': 0.3}
        self.adaptation_smoothing = 0.7
        
    def select_attention_type(
        self,
        seq_length: int,
        memory_usage: Optional[float] = None,
        attention_patterns: Optional[Dict] = None,
        force_type: Optional[str] = None
    ) -> str:
        """Select optimal attention type based on current conditions"""
        
        if force_type is not None and force_type in self.attention_types:
            selected_type = force_type
        else:
            # Rule-based selection
            if seq_length > self.seq_length_threshold_dsa:
                selected_type = 'dsa'
            elif seq_length > self.seq_length_threshold_mla:
                selected_type = 'mla'
            elif memory_usage and memory_usage > self.memory_threshold_mla:
                selected_type = 'mla'
            elif attention_patterns and self._should_use_dsa(attention_patterns):
                selected_type = 'dsa'
            else:
                selected_type = 'standard'
        
        # Update history
        self.seq_length_history.append(seq_length)
        if memory_usage is not None:
            self.memory_usage_history.append(memory_usage)
        self.attention_selection_history.append(selected_type)
        
        self.current_attention_type = selected_type
        
        # Log selection
        if self.adaptation_step % 10 == 0:  # Log every 10 selections
            memory_str = f"{memory_usage:.2f}" if memory_usage else "N/A"
            print(f"Attention Selection: {selected_type.upper()} (seq_len={seq_length}, mem={memory_str})")
        
        return selected_type
    
    def _should_use_dsa(self, attention_patterns: Dict) -> bool:
        """Check if conditions favor DSA"""
        # Check attention sparsity
        sparsity_score = attention_patterns.get('sparsity_score', 0.0)
        if sparsity_score > self.sparsity_threshold_dsa:
            return True
        
        # Check local attention dominance
        local_attention_ratio = attention_patterns.get('local_attention_ratio', 0.0)
        if local_attention_ratio > 0.7:  # 70% local attention
            return True
        
        # Check token clustering
        token_clustering = attention_patterns.get('token_clustering_score', 0.0)
        if token_clustering > 0.5:
            return True
        
        return False
    
    def update_performance(
        self,
        attention_type: str,
        throughput: float,
        memory_usage: float,
        accuracy_score: Optional[float] = None
    ):
        """Update performance metrics for attention type"""
        if attention_type not in self.attention_types:
            return
        
        performance_score = (
            self.performance_weights['throughput'] * throughput +
            self.performance_weights['memory'] * (1.0 - memory_usage) +  # Lower memory is better
            (self.performance_weights['accuracy'] * (accuracy_score or 0.5))
        )
        
        self.attention_performance[attention_type].append({
            'throughput': throughput,
            'memory_usage': memory_usage,
            'accuracy': accuracy_score,
            'combined_score': performance_score,
            'step': self.adaptation_step
        })
    
    def _adapt_selection_thresholds(self):
        """Adapt selection thresholds based on performance history"""
        if self.adaptation_step < self.adaptation_frequency:
            return
        
        for att_type in self.attention_types:
            if len(self.attention_performance[att_type]) < 10:
                continue
            
            # Get recent performance
            recent_perf = list(self.attention_performance[att_type])[-20:]
            avg_performance = np.mean([p['combined_score'] for p in recent_perf])
            
            # Adapt thresholds based on performance
            if att_type == 'dsa' and avg_performance > 0.7:  # DSA performing well
                # Be more aggressive in using DSA
                self.sparsity_threshold_dsa = max(
                    0.1, self.sparsity_threshold_dsa * 0.9
                )
                self.seq_length_threshold_dsa = max(
                    2048, self.seq_length_threshold_dsa * 0.8
                )
            
            elif att_type == 'mla' and avg_performance > 0.7:
                # Be more aggressive in using MLA
                self.memory_threshold_mla = max(
                    0.4, self.memory_threshold_mla * 0.9
                )
                self.seq_length_threshold_mla = max(
                    1024, self.seq_length_threshold_mla * 0.8
                )
        
        print(f"Adapted thresholds: DSA@{self.seq_length_threshold_dsa}, MLA@{self.seq_length_threshold_mla}")
    
    def get_attention_recommendations(self) -> Dict[str, any]:
        """Get recommendations for attention mechanism usage"""
        recommendations = {}
        
        if len(self.attention_selection_history) > 50:
            # Selection frequency
            recent_selections = list(self.attention_selection_history)[-100:]
            selection_counts = {}
            for att_type in self.attention_types:
                selection_counts[att_type] = recent_selections.count(att_type) / len(recent_selections)
            
            recommendations['selection_frequency'] = selection_counts
            recommendations['most_used'] = max(selection_counts, key=selection_counts.get)
        
        # Performance comparison
        avg_performance = {}
        for att_type in self.attention_types:
            if len(self.attention_performance[att_type]) > 0:
                recent_perf = list(self.attention_performance[att_type])[-20:]
                avg_performance[att_type] = np.mean([p['combined_score'] for p in recent_perf])
        
        if avg_performance:
            recommendations['performance_ranking'] = sorted(
                avg_performance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            recommendations['best_performing'] = recommendations['performance_ranking'][0][0]
        
        # Context recommendations
        if len(self.seq_length_history) > 10:
            avg_seq_length = np.mean(list(self.seq_length_history)[-50:])
            recommendations['avg_seq_length'] = avg_seq_length
            
            if avg_seq_length > self.seq_length_threshold_dsa:
                recommendations['recommended_for_context'] = 'dsa'
            elif avg_seq_length > self.seq_length_threshold_mla:
                recommendations['recommended_for_context'] = 'mla'
            else:
                recommendations['recommended_for_context'] = 'standard'
        
        return recommendations
    
    def get_current_config(self) -> Dict[str, any]:
        """Get current attention selection configuration"""
        return {
            'current_type': self.current_attention_type,
            'thresholds': {
                'mla_seq_length': self.seq_length_threshold_mla,
                'dsa_seq_length': self.seq_length_threshold_dsa,
                'mla_memory': self.memory_threshold_mla,
                'dsa_sparsity': self.sparsity_threshold_dsa
            },
            'adaptation_step': self.adaptation_step,
            'performance_weights': self.performance_weights.copy()
        }
    
    def step(self):
        """Call at each training step for adaptation"""
        self.adaptation_step += 1
        
        # Periodic threshold adaptation
        if self.adaptation_step % self.adaptation_frequency == 0:
            self._adapt_selection_thresholds()


# Export classes
__all__ = [
    'DynamicExpertCapacityManager',
    'AdaptiveAttentionSelector'
]