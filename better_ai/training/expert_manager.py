"""Expert Specialization Manager for MoE Training"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict, deque
import time


class ExpertSpecializationManager:
    """
    Tracks and optimizes expert specialization during MoE training
    Enables better expert utilization and specialization patterns
    """
    
    def __init__(
        self,
        num_experts: int,
        num_languages: Optional[int] = None,
        history_length: int = 1000,
        specialization_threshold: float = 0.7,
        rebalance_frequency: int = 100,
        device: torch.device = torch.device('cpu')
    ):
        self.num_experts = num_experts
        self.num_languages = num_languages
        self.history_length = history_length
        self.specialization_threshold = specialization_threshold
        self.rebalance_frequency = rebalance_frequency
        self.device = device
        
        # Expert tracking matrices
        self.expert_language_counts = torch.zeros(num_experts, num_languages or 10, device=device)
        self.expert_load_history = deque(maxlen=history_length)
        self.expert_complexity_scores = torch.zeros(num_experts, device=device)
        self.expert_activation_patterns = torch.zeros(num_experts, num_experts, device=device)
        
        # Language to index mapping
        self.language_to_idx = {
            'python': 0, 'c': 1, 'rust': 2, 'javascript': 3,
            'java': 4, 'cpp': 5, 'go': 6, 'rust': 7,
            'typescript': 8, 'other': 9
        }
        
        # Specialization metrics
        self.expert_specialization_scores = torch.zeros(num_experts, device=device)
        self.expert_efficiency_scores = torch.zeros(num_experts, device=device)
        
        # Training step tracking
        self.current_step = 0
        self.last_rebalance_step = 0
        
        # Expert load balancing
        self.target_load_per_expert = 1.0 / num_experts
        self.load_balance_history = deque(maxlen=100)
        
    def update_expert_stats(
        self,
        expert_ids: torch.Tensor,
        language_tokens: Optional[torch.Tensor] = None,
        token_complexity: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None
    ):
        """Update expert statistics based on current batch"""
        if expert_ids is None:
            return
            
        # Count expert activations
        expert_counts = torch.bincount(expert_ids.flatten(), minlength=self.num_experts)
        expert_loads = expert_counts.float() / expert_ids.numel()
        
        # Update load history
        self.expert_load_history.append(expert_loads.cpu())
        
        # Update language specialization if provided
        if language_tokens is not None and self.num_languages is not None:
            for i, expert_id in enumerate(expert_ids.flatten()):
                if i < len(language_tokens):
                    lang = language_tokens[i].lower() if isinstance(language_tokens[i], str) else 'other'
                    lang_idx = self.language_to_idx.get(lang, 9)  # Default to 'other'
                    self.expert_language_counts[expert_id, lang_idx] += 1
        
        # Update complexity scores if provided
        if token_complexity is not None:
            for i, expert_id in enumerate(expert_ids.flatten()):
                if i < len(token_complexity):
                    self.expert_complexity_scores[expert_id] += token_complexity[i].float()
        
        # Update activation patterns if routing weights provided
        if routing_weights is not None:
            # Track which experts tend to activate together
            batch_expert_ids = expert_ids.flatten()
            for i in range(len(batch_expert_ids)):
                for j in range(i + 1, len(batch_expert_ids)):
                    expert_i, expert_j = batch_expert_ids[i], batch_expert_ids[j]
                    self.expert_activation_patterns[expert_i, expert_j] += 1
                    self.expert_activation_patterns[expert_j, expert_i] += 1
        
        self.current_step += 1
        
        # Check for rebalancing
        if (self.current_step - self.last_rebalance_step) >= self.rebalance_frequency:
            self._compute_specialization_scores()
            self._detect_expert_collapse()
            self.last_rebalance_step = self.current_step
    
    def _compute_specialization_scores(self):
        """Compute expert specialization and efficiency scores"""
        # Language specialization (entropy-based)
        language_dist = self.expert_language_counts + 1e-8  # Avoid division by zero
        language_dist = language_dist / language_dist.sum(dim=1, keepdim=True)
        language_entropy = -(language_dist * torch.log(language_dist + 1e-8)).sum(dim=1)
        max_entropy = torch.log(torch.tensor(self.num_languages or 10, dtype=torch.float32))
        language_specialization = 1.0 - (language_entropy / max_entropy)
        
        # Load efficiency (how close to target load)
        if len(self.expert_load_history) > 0:
            recent_loads = torch.stack(list(self.expert_load_history)[-10:])  # Last 10 steps
            avg_loads = recent_loads.mean(dim=0)
            load_efficiency = 1.0 - torch.abs(avg_loads - self.target_load_per_expert)
        else:
            load_efficiency = torch.ones(self.num_experts, device=self.device)
        
        # Complexity efficiency (normalized complexity per activation)
        total_activations = self.expert_language_counts.sum(dim=1) + 1e-8
        complexity_efficiency = self.expert_complexity_scores / total_activations
        
        # Combined specialization score
        self.expert_specialization_scores = (
            0.4 * language_specialization +
            0.3 * load_efficiency +
            0.3 * complexity_efficiency
        )
        
        # Store efficiency scores
        self.expert_efficiency_scores = load_efficiency
        
        # Update load balance history
        current_load_balance = torch.std(avg_loads).item() if len(self.expert_load_history) > 0 else 0.0
        self.load_balance_history.append(current_load_balance)
    
    def _detect_expert_collapse(self):
        """Detect if any expert has collapsed (becoming inactive)"""
        if len(self.expert_load_history) < 10:
            return
            
        recent_loads = torch.stack(list(self.expert_load_history)[-10:])
        avg_loads = recent_loads.mean(dim=0)
        
        collapsed_experts = (avg_loads < 0.01).nonzero(as_tuple=True)[0]
        
        if len(collapsed_experts) > 0:
            print(f"⚠️  Expert Collapse Detected: {collapsed_experts.tolist()}")
            print(f"   Consider increasing load_balance_loss_weight or reducing capacity_factor")
    
    def get_expert_recommendations(self) -> Dict[str, torch.Tensor]:
        """Get recommendations for expert routing optimization"""
        recommendations = {}
        
        # Under-utilized experts
        if len(self.expert_load_history) > 0:
            recent_loads = torch.stack(list(self.expert_load_history)[-5:]).mean(dim=0)
            underutilized = (recent_loads < 0.5 * self.target_load_per_expert).nonzero(as_tuple=True)[0]
            overutilized = (recent_loads > 1.5 * self.target_load_per_expert).nonzero(as_tuple=True)[0]
            
            recommendations['underutilized_experts'] = underutilized
            recommendations['overutilized_experts'] = overutilized
        
        # Highly specialized experts
        highly_specialized = (self.expert_specialization_scores > self.specialization_threshold).nonzero(as_tuple=True)[0]
        recommendations['specialized_experts'] = highly_specialized
        
        # Expert pairs that activate together
        if self.expert_activation_patterns.sum() > 0:
            activation_patterns = self.expert_activation_patterns / self.expert_activation_patterns.sum()
            strong_pairs = []
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    if activation_patterns[i, j] > 0.1:  # Threshold for "strong" pairs
                        strong_pairs.append((i, j, activation_patterns[i, j].item()))
            recommendations['strong_expert_pairs'] = strong_pairs
        
        return recommendations
    
    def get_expert_stats(self) -> Dict[str, torch.Tensor]:
        """Get comprehensive expert statistics"""
        stats = {
            'specialization_scores': self.expert_specialization_scores,
            'efficiency_scores': self.expert_efficiency_scores,
            'complexity_scores': self.expert_complexity_scores,
            'language_distribution': self.expert_language_counts,
            'activation_patterns': self.expert_activation_patterns
        }
        
        if len(self.expert_load_history) > 0:
            recent_loads = torch.stack(list(self.expert_load_history)[-10:])
            stats['recent_loads'] = recent_loads.mean(dim=0)
            stats['load_std'] = recent_loads.std(dim=0)
        
        if len(self.load_balance_history) > 0:
            stats['load_balance_trend'] = torch.tensor(self.load_balance_history)
        
        return stats
    
    def reset_history(self):
        """Reset tracking history (useful for new training phases)"""
        self.expert_load_history.clear()
        self.load_balance_history.clear()
        self.expert_language_counts.zero_()
        self.expert_complexity_scores.zero_()
        self.expert_activation_patterns.zero_()
        self.expert_specialization_scores.zero_()
        self.expert_efficiency_scores.zero_()
        self.current_step = 0
        self.last_rebalance_step = 0
    
    def get_language_specialization_matrix(self) -> torch.Tensor:
        """Get expert-language specialization matrix normalized"""
        if self.expert_language_counts.sum() == 0:
            return torch.zeros_like(self.expert_language_counts)
        
        # Normalize per expert
        expert_totals = self.expert_language_counts.sum(dim=1, keepdim=True)
        normalized = self.expert_language_counts / (expert_totals + 1e-8)
        return normalized
    
    def log_specialization_progress(self, step: int):
        """Log current specialization progress"""
        if len(self.expert_load_history) == 0:
            return
            
        recent_loads = torch.stack(list(self.expert_load_history)[-10:]).mean(dim=0)
        load_balance_score = (1.0 - torch.std(recent_loads)).item()
        
        avg_specialization = self.expert_specialization_scores.mean().item()
        max_specialization = self.expert_specialization_scores.max().item()
        
        print(f"\\n📊 Expert Specialization Progress (Step {step}):")
        print(f"   Load Balance: {load_balance_score:.3f}")
        print(f"   Avg Specialization: {avg_specialization:.3f}")
        print(f"   Max Specialization: {max_specialization:.3f}")
        
        # Show top specialized experts
        top_experts = torch.topk(self.expert_specialization_scores, 3)
        print(f"   Top Specialized: {top_experts.indices.tolist()}")
        print(f"   Specialization Scores: {top_experts.values.tolist():.3f}")


class MoETrainingMonitor:
    """Enhanced monitoring for MoE training with specialization tracking"""
    
    def __init__(
        self,
        num_experts: int,
        num_languages: Optional[int] = None,
        log_frequency: int = 100,
        save_frequency: int = 1000,
        log_dir: str = "./logs"
    ):
        self.num_experts = num_experts
        self.log_frequency = log_frequency
        self.save_frequency = save_frequency
        self.log_dir = log_dir
        
        # Expert specialization manager
        self.expert_manager = ExpertSpecializationManager(
            num_experts=num_experts,
            num_languages=num_languages
        )
        
        # Training metrics
        self.training_metrics = {
            'loss': deque(maxlen=1000),
            'aux_loss': deque(maxlen=1000),
            'learning_rate': deque(maxlen=1000),
            'gradient_norm': deque(maxlen=1000),
            'expert_utilization': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000)
        }
        
        # Performance tracking
        self.batch_times = deque(maxlen=100)
        self.step_times = deque(maxlen=1000)
        self.last_log_time = time.time()
        
    def update_step(
        self,
        step: int,
        loss: float,
        aux_loss: float,
        learning_rate: float,
        gradient_norm: float,
        expert_ids: Optional[torch.Tensor] = None,
        language_tokens: Optional[List[str]] = None,
        memory_usage: Optional[float] = None,
        batch_time: Optional[float] = None
    ):
        """Update training step metrics"""
        current_time = time.time()
        
        # Update basic metrics
        self.training_metrics['loss'].append(loss)
        self.training_metrics['aux_loss'].append(aux_loss)
        self.training_metrics['learning_rate'].append(learning_rate)
        self.training_metrics['gradient_norm'].append(gradient_norm)
        
        if memory_usage is not None:
            self.training_metrics['memory_usage'].append(memory_usage)
        
        if batch_time is not None:
            self.batch_times.append(batch_time)
        
        # Update expert statistics
        if expert_ids is not None:
            self.expert_manager.update_expert_stats(
                expert_ids=expert_ids,
                language_tokens=language_tokens
            )
        
        # Log progress
        if step % self.log_frequency == 0:
            self._log_progress(step)
        
        # Save detailed statistics
        if step % self.save_frequency == 0:
            self._save_detailed_stats(step)
    
    def _log_progress(self, step: int):
        """Log training progress"""
        recent_loss = np.mean(list(self.training_metrics['loss'])[-20:]) if len(self.training_metrics['loss']) > 0 else 0
        recent_aux_loss = np.mean(list(self.training_metrics['aux_loss'])[-20:]) if len(self.training_metrics['aux_loss']) > 0 else 0
        recent_lr = list(self.training_metrics['learning_rate'])[-1] if len(self.training_metrics['learning_rate']) > 0 else 0
        
        # Performance metrics
        if len(self.batch_times) > 0:
            avg_batch_time = np.mean(list(self.batch_times)[-10:])
            tokens_per_sec = self._estimate_tokens_per_sec(avg_batch_time)
        else:
            avg_batch_time = 0
            tokens_per_sec = 0
        
        print(f"\nStep {step} Training Progress:")
        print(f"   Loss: {recent_loss:.6f} | Aux Loss: {recent_aux_loss:.6f}")
        print(f"   Learning Rate: {recent_lr:.2e} | Batch Time: {avg_batch_time:.3f}s")
        print(f"   Tokens/sec: {tokens_per_sec:.0f}")
        
        # Expert specialization progress
        self.expert_manager.log_specialization_progress(step)
    
    def _estimate_tokens_per_sec(self, batch_time: float) -> float:
        """Estimate tokens per second based on batch time"""
        # Rough estimate based on typical batch sizes
        typical_batch_size = 32  # tokens per batch
        typical_seq_length = 1024  # sequence length
        total_tokens = typical_batch_size * typical_seq_length
        
        return total_tokens / max(batch_time, 0.001)
    
    def _save_detailed_stats(self, step: int):
        """Save detailed statistics to file"""
        import json
        import os
        
        stats = {
            'step': step,
            'timestamp': time.time(),
            'training_metrics': {
                'recent_loss': np.mean(list(self.training_metrics['loss'])[-20:]) if len(self.training_metrics['loss']) > 0 else 0,
                'recent_aux_loss': np.mean(list(self.training_metrics['aux_loss'])[-20:]) if len(self.training_metrics['aux_loss']) > 0 else 0,
                'current_lr': list(self.training_metrics['learning_rate'])[-1] if len(self.training_metrics['learning_rate']) > 0 else 0,
                'recent_gradient_norm': np.mean(list(self.training_metrics['gradient_norm'])[-20:]) if len(self.training_metrics['gradient_norm']) > 0 else 0,
                'memory_usage_gb': list(self.training_metrics['memory_usage'])[-1] if len(self.training_metrics['memory_usage']) > 0 else 0,
            },
            'expert_stats': {
                'specialization_scores': self.expert_manager.expert_specialization_scores.tolist(),
                'efficiency_scores': self.expert_manager.expert_efficiency_scores.tolist(),
                'language_specialization': self.expert_manager.get_language_specialization_matrix().tolist(),
                'recommendations': self.expert_manager.get_expert_recommendations()
            }
        }
        
        # Save to file
        os.makedirs(self.log_dir, exist_ok=True)
        with open(f"{self.log_dir}/expert_stats_step_{step}.json", 'w') as f:
            json.dump(stats, f, indent=2)
    
    def get_training_summary(self) -> Dict[str, any]:
        """Get comprehensive training summary"""
        return {
            'expert_specialization': self.expert_manager.get_expert_stats(),
            'training_metrics': {
                k: list(v)[-100:] if len(v) > 0 else [] 
                for k, v in self.training_metrics.items()
            },
            'performance': {
                'avg_batch_time': np.mean(list(self.batch_times)) if len(self.batch_times) > 0 else 0,
                'recent_tokens_per_sec': self._estimate_tokens_per_sec(
                    np.mean(list(self.batch_times)[-10:]) if len(self.batch_times) > 0 else 1
                )
            }
        }


# Export classes
__all__ = [
    'ExpertSpecializationManager', 
    'MoETrainingMonitor'
]