"""Coherence-based Scheduler for Adaptive MoE Training"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from collections import deque
import math
import time


class CoherenceCalculator:
    """
    Calculates various coherence metrics for MoE training
    Measures model stability, expert specialization, and convergence quality
    """
    
    def __init__(
        self,
        window_size: int = 50,
        stability_threshold: float = 0.1,
        specialization_weight: float = 0.3,
        convergence_weight: float = 0.4,
        stability_weight: float = 0.3,
        device: torch.device = torch.device('cpu')
    ):
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.specialization_weight = specialization_weight
        self.convergence_weight = convergence_weight
        self.stability_weight = stability_weight
        self.device = device
        
        # Tracking history
        self.loss_history = deque(maxlen=window_size * 2)
        self.aux_loss_history = deque(maxlen=window_size * 2)
        self.expert_util_history = deque(maxlen=window_size * 2)
        self.gradient_norm_history = deque(maxlen=window_size * 2)
        self.expert_specialization_history = deque(maxlen=window_size)
        
        # Coherence components
        self.current_stability = 0.0
        self.current_specialization = 0.0
        self.current_convergence = 0.0
        self.current_coherence = 0.0
        
        # Trend analysis
        self.loss_trend = 0.0
        self.expert_util_trend = 0.0
        self.gradient_trend = 0.0
        
    def update_metrics(
        self,
        loss: float,
        aux_loss: float,
        expert_utilization: float,
        gradient_norm: float,
        expert_specialization: Optional[float] = None
    ) -> Dict[str, float]:
        """Update coherence metrics with new training step data"""
        
        # Add to history
        self.loss_history.append(loss)
        self.aux_loss_history.append(aux_loss)
        self.expert_util_history.append(expert_utilization)
        self.gradient_norm_history.append(gradient_norm)
        
        if expert_specialization is not None:
            self.expert_specialization_history.append(expert_specialization)
        
        # Calculate current coherence components
        self._calculate_stability()
        self._calculate_specialization()
        self._calculate_convergence()
        self._calculate_trends()
        
        # Combine into overall coherence
        self.current_coherence = (
            self.specialization_weight * self.current_specialization +
            self.convergence_weight * self.current_convergence +
            self.stability_weight * self.current_stability
        )
        
        return {
            'coherence': self.current_coherence,
            'stability': self.current_stability,
            'specialization': self.current_specialization,
            'convergence': self.current_convergence,
            'loss_trend': self.loss_trend,
            'expert_util_trend': self.expert_util_trend,
            'gradient_trend': self.gradient_trend
        }
    
    def _calculate_stability(self):
        """Calculate training stability metric"""
        if len(self.loss_history) < 10:
            self.current_stability = 0.5
            return
        
        recent_losses = list(self.loss_history)[-self.window_size:]
        recent_grads = list(self.gradient_norm_history)[-self.window_size:]
        
        # Loss stability (coefficient of variation)
        if len(recent_losses) > 1:
            loss_mean = np.mean(recent_losses)
            loss_std = np.std(recent_losses)
            loss_stability = max(0.0, 1.0 - (loss_std / max(loss_mean, 1e-8)))
        else:
            loss_stability = 0.5
        
        # Gradient stability
        if len(recent_grads) > 1:
            grad_mean = np.mean(recent_grads)
            grad_std = np.std(recent_grads)
            grad_stability = max(0.0, 1.0 - (grad_std / max(grad_mean, 1e-8)))
        else:
            grad_stability = 0.5
        
        # Combine stability metrics
        self.current_stability = 0.6 * loss_stability + 0.4 * grad_stability
    
    def _calculate_specialization(self):
        """Calculate expert specialization coherence"""
        if len(self.expert_specialization_history) < 5:
            self.current_specialization = 0.5
            return
        
        recent_specialization = list(self.expert_specialization_history)[-self.window_size:]
        
        # Specialization consistency
        if len(recent_specialization) > 1:
            spec_mean = np.mean(recent_specialization)
            spec_std = np.std(recent_specialization)
            spec_consistency = max(0.0, 1.0 - (spec_std / max(spec_mean, 1e-8)))
        else:
            spec_consistency = 0.5
        
        # Specialization progress (improvement over time)
        if len(recent_specialization) >= 10:
            early_spec = np.mean(recent_specialization[:5])
            recent_spec = np.mean(recent_specialization[-5:])
            if early_spec > 0:
                spec_progress = min(1.0, recent_spec / early_spec)
            else:
                spec_progress = 0.5 + 0.5 * spec_progress
        else:
            spec_progress = 0.5
        
        # Expert utilization balance
        if len(self.expert_util_history) > 10:
            recent_utils = list(self.expert_util_history)[-self.window_size:]
            util_balance = 1.0 - min(1.0, np.std(recent_utils) / max(np.mean(recent_utils), 1e-8))
        else:
            util_balance = 0.5
        
        # Combine specialization metrics
        self.current_specialization = (
            0.4 * spec_consistency +
            0.3 * spec_progress +
            0.3 * util_balance
        )
    
    def _calculate_convergence(self):
        """Calculate convergence quality"""
        if len(self.loss_history) < 20:
            self.current_convergence = 0.5
            return
        
        recent_losses = list(self.loss_history)[-self.window_size:]
        
        # Loss reduction rate
        if len(recent_losses) >= 10:
            early_loss = np.mean(recent_losses[:10])
            recent_loss = np.mean(recent_losses[-10:])
            
            if early_loss > 0:
                loss_reduction = max(0.0, (early_loss - recent_loss) / early_loss)
            else:
                loss_reduction = 0.0
        else:
            loss_reduction = 0.0
        
        # Loss plateau detection
        if len(recent_losses) >= 20:
            recent_var = np.var(recent_losses[-10:])
            early_var = np.var(recent_losses[:10])
            plateau_score = max(0.0, 1.0 - (recent_var / max(early_var, 1e-8)))
        else:
            plateau_score = 0.5
        
        # Gradient norm convergence
        if len(self.gradient_norm_history) >= 10:
            recent_grads = list(self.gradient_norm_history)[-10:]
            early_grads = list(self.gradient_norm_history)[-20:-10]
            
            if len(early_grads) > 0:
                grad_convergence = max(0.0, 1.0 - (np.mean(recent_grads) / max(np.mean(early_grads), 1e-8)))
            else:
                grad_convergence = 0.5
        else:
            grad_convergence = 0.5
        
        # Combine convergence metrics
        self.current_convergence = (
            0.4 * loss_reduction +
            0.3 * plateau_score +
            0.3 * grad_convergence
        )
    
    def _calculate_trends(self):
        """Calculate various trend metrics"""
        if len(self.loss_history) >= 5:
            self.loss_trend = self._calculate_linear_trend(list(self.loss_history))
        
        if len(self.expert_util_history) >= 5:
            self.expert_util_trend = self._calculate_linear_trend(list(self.expert_util_history))
        
        if len(self.gradient_norm_history) >= 5:
            self.gradient_trend = self._calculate_linear_trend(list(self.gradient_norm_history))
    
    def _calculate_linear_trend(self, values: List[float]) -> float:
        """Calculate linear trend (negative = decreasing, positive = increasing)"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        if np.std(y) > 1e-8:
            slope = np.polyfit(x, y, 1)[0]
            # Normalize slope based on data range
            y_range = np.max(y) - np.min(y)
            if y_range > 1e-8:
                normalized_slope = slope / y_range
                return np.clip(normalized_slope, -1.0, 1.0)
        
        return 0.0
    
    def get_coherence_status(self) -> str:
        """Get human-readable coherence status"""
        if self.current_coherence > 0.8:
            return "EXCELLENT"
        elif self.current_coherence > 0.6:
            return "GOOD"
        elif self.current_coherence > 0.4:
            return "MODERATE"
        elif self.current_coherence > 0.2:
            return "POOR"
        else:
            return "VERY_POOR"
    
    def should_adjust_training(self) -> Dict[str, bool]:
        """Determine if training parameters should be adjusted"""
        
        recommendations = {
            'reduce_lr': False,
            'increase_lr': False,
            'adjust_expert_capacity': False,
            'change_attention_mechanism': False,
            'early_stop': False,
            'change_batch_size': False
        }
        
        # Learning rate adjustments
        if self.loss_trend > 0.05:  # Loss increasing significantly
            recommendations['reduce_lr'] = True
        elif self.loss_trend < -0.1 and self.current_stability > 0.7:  # Loss decreasing steadily
            recommendations['increase_lr'] = True
        
        # Expert capacity adjustments
        if self.current_stability < 0.4:  # Unstable training
            recommendations['adjust_expert_capacity'] = True
        
        # Attention mechanism changes
        if (self.gradient_trend > 0.05 or 
            (self.loss_trend > 0.02 and self.current_convergence < 0.3)):
            recommendations['change_attention_mechanism'] = True
        
        # Early stopping
        if (self.current_coherence > 0.7 and 
            self.loss_trend < -0.01 and 
            len(self.loss_history) > 100):
            recommendations['early_stop'] = True
        
        # Batch size adjustments
        if self.current_stability < 0.3:  # Very unstable
            recommendations['change_batch_size'] = True
        
        return recommendations


class CoherenceBasedScheduler:
    """
    Adaptive scheduler that adjusts training parameters based on coherence metrics
    Replaces static LR schedules with dynamic coherence-driven adjustments
    """
    
    def __init__(
        self,
        base_lr: float = 1e-4,
        min_lr: float = 1e-6,
        max_lr: float = 1e-3,
        coherence_target: float = 0.7,
        adjustment_frequency: int = 50,
        adjustment_factor: float = 0.1,
        patience: int = 200,
        device: torch.device = torch.device('cpu')
    ):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.coherence_target = coherence_target
        self.adjustment_frequency = adjustment_frequency
        self.adjustment_factor = adjustment_factor
        self.patience = patience
        self.device = device
        
        # Coherence calculator
        self.coherence_calc = CoherenceCalculator(device=device)
        
        # Current learning rate and training state
        self.current_lr = base_lr
        self.step_count = 0
        self.last_improvement_step = 0
        self.stagnation_count = 0
        
        # Adjustment history
        self.lr_history = deque(maxlen=500)
        self.adjustment_history = deque(maxlen=100)
        
        # Training phases
        self.current_phase = "WARMUP"
        self.phase_steps = {
            "WARMUP": 0,
            "TRAINING": 0,
            "STABILIZING": 0,
            "FINE_TUNING": 0
        }
        
    def step(
        self,
        loss: float,
        aux_loss: float,
        expert_utilization: float,
        gradient_norm: float,
        expert_specialization: Optional[float] = None,
        step: Optional[int] = None
    ) -> Dict[str, Union[float, str, bool]]:
        """Perform one scheduler step and update learning rate"""
        
        if step is not None:
            self.step_count = step
        
        # Update coherence metrics
        coherence_metrics = self.coherence_calc.update_metrics(
            loss=loss,
            aux_loss=aux_loss,
            expert_utilization=expert_utilization,
            gradient_norm=gradient_norm,
            expert_specialization=expert_specialization
        )
        
        # Determine training phase
        self._update_training_phase(coherence_metrics)
        
        # Adjust learning rate based on coherence
        if self.step_count % self.adjustment_frequency == 0:
            self._adjust_learning_rate(coherence_metrics)
        
        # Check for early stopping
        should_stop = self._should_early_stop(coherence_metrics)
        
        return {
            'current_lr': self.current_lr,
            'coherence': coherence_metrics['coherence'],
            'phase': self.current_phase,
            'adjusted': self.step_count % self.adjustment_frequency == 0,
            'should_stop': should_stop,
            'recommendations': self.coherence_calc.should_adjust_training()
        }
    
    def _update_training_phase(self, coherence_metrics: Dict[str, float]):
        """Update training phase based on coherence"""
        coherence = coherence_metrics['coherence']
        stability = coherence_metrics['stability']
        convergence = coherence_metrics['convergence']
        
        old_phase = self.current_phase
        
        # Phase determination logic
        if self.step_count < 500:  # First 500 steps
            self.current_phase = "WARMUP"
        elif coherence < 0.3 or stability < 0.3:
            self.current_phase = "STABILIZING"
        elif coherence > 0.7 and convergence > 0.6:
            self.current_phase = "FINE_TUNING"
        else:
            self.current_phase = "TRAINING"
        
        # Update phase step counts
        if self.current_phase != old_phase:
            self.phase_steps[self.current_phase] = 0
        self.phase_steps[self.current_phase] += 1
    
    def _adjust_learning_rate(self, coherence_metrics: Dict[str, float]):
        """Adjust learning rate based on coherence metrics"""
        coherence = coherence_metrics['coherence']
        stability = coherence_metrics['stability']
        convergence = coherence_metrics['convergence']
        loss_trend = coherence_metrics['loss_trend']
        
        old_lr = self.current_lr
        adjustment_factor = 1.0
        
        # Coherence-based adjustments
        if coherence > self.coherence_target:
            # Good coherence - can be more aggressive
            if convergence > 0.7 and loss_trend < -0.01:
                # Converging well - increase LR slightly
                adjustment_factor = 1.0 + self.adjustment_factor * 0.5
            elif stability > 0.8:
                # Very stable - maintain LR
                adjustment_factor = 1.0
            else:
                # Good but not converging - slight LR increase
                adjustment_factor = 1.0 + self.adjustment_factor * 0.25
        else:
            # Poor coherence - be conservative
            if loss_trend > 0.02:
                # Loss increasing - reduce LR
                adjustment_factor = 1.0 - self.adjustment_factor * 2.0
            elif stability < 0.4:
                # Unstable - reduce LR significantly
                adjustment_factor = 1.0 - self.adjustment_factor * 3.0
            else:
                # Poor but not catastrophic - moderate reduction
                adjustment_factor = 1.0 - self.adjustment_factor
        
        # Phase-specific adjustments
        if self.current_phase == "WARMUP":
            # Gradual LR increase during warmup
            warmup_progress = min(1.0, self.step_count / 500)
            adjustment_factor *= (0.5 + 0.5 * warmup_progress)
        elif self.current_phase == "STABILIZING":
            # Conservative during stabilization
            adjustment_factor *= 0.7
        elif self.current_phase == "FINE_TUNING":
            # Very conservative during fine-tuning
            adjustment_factor *= 0.5
        
        # Apply adjustment with bounds
        new_lr = self.current_lr * adjustment_factor
        new_lr = max(self.min_lr, min(self.max_lr, new_lr))
        
        self.current_lr = new_lr
        self.lr_history.append(new_lr)
        
        # Record adjustment
        self.adjustment_history.append({
            'step': self.step_count,
            'old_lr': old_lr,
            'new_lr': new_lr,
            'adjustment_factor': adjustment_factor,
            'coherence': coherence,
            'phase': self.current_phase,
            'reason': self._get_adjustment_reason(coherence_metrics, adjustment_factor)
        })
        
        # Log significant adjustments
        if abs(new_lr - old_lr) / old_lr > 0.05:  # >5% change
            print(f"LR Adjustment: {old_lr:.2e} -> {new_lr:.2e} ({self._get_adjustment_reason(coherence_metrics, adjustment_factor)})")
    
    def _get_adjustment_reason(self, coherence_metrics: Dict[str, float], factor: float) -> str:
        """Get human-readable reason for LR adjustment"""
        coherence = coherence_metrics['coherence']
        stability = coherence_metrics['stability']
        
        if factor > 1.1:
            return "LR_INCREASE_COHERENCE_GOOD"
        elif factor < 0.9:
            if coherence < 0.3:
                return "LR_DECREASE_COHERENCE_POOR"
            elif stability < 0.4:
                return "LR_DECREASE_INSTABILITY"
            elif coherence_metrics['loss_trend'] > 0.02:
                return "LR_DECREASE_LOSS_INCREASING"
            else:
                return "LR_DECREASE_CONSERVATIVE"
        else:
            return "LR_MAINTAIN_STABLE"
    
    def _should_early_stop(self, coherence_metrics: Dict[str, float]) -> bool:
        """Determine if training should be stopped early"""
        coherence = coherence_metrics['coherence']
        convergence = coherence_metrics['convergence']
        
        # Early stopping criteria
        if (coherence > 0.8 and 
            convergence > 0.8 and 
            self.step_count > 1000 and
            len(self.coherence_calc.loss_history) > 200):
            
            # Check if loss has plateaued
            recent_losses = list(self.coherence_calc.loss_history)[-100:]
            if len(recent_losses) >= 50:
                early_loss = np.mean(recent_losses[:50])
                recent_loss = np.mean(recent_losses[-50:])
                
                # If very little improvement and high coherence
                improvement = (early_loss - recent_loss) / early_loss
                if improvement < 0.01:  # Less than 1% improvement
                    return True
        
        return False
    
    def get_current_phase_info(self) -> Dict[str, any]:
        """Get information about current training phase"""
        return {
            'current_phase': self.current_phase,
            'phase_steps': self.phase_steps[self.current_phase],
            'phase_duration': self.phase_steps[self.current_phase],
            'coherence_target': self.coherence_target,
            'current_coherence': self.coherence_calc.current_coherence,
            'coherence_status': self.coherence_calc.get_coherence_status()
        }
    
    def get_scheduler_stats(self) -> Dict[str, any]:
        """Get comprehensive scheduler statistics"""
        return {
            'current_lr': self.current_lr,
            'base_lr': self.base_lr,
            'lr_range': (self.min_lr, self.max_lr),
            'total_adjustments': len(self.adjustment_history),
            'step_count': self.step_count,
            'coherence_metrics': {
                'current': self.coherence_calc.current_coherence,
                'stability': self.coherence_calc.current_stability,
                'specialization': self.coherence_calc.current_specialization,
                'convergence': self.coherence_calc.current_convergence,
                'trends': {
                    'loss': self.coherence_calc.loss_trend,
                    'expert_util': self.coherence_calc.expert_util_trend,
                    'gradient': self.coherence_calc.gradient_trend
                }
            },
            'training_phase': {
                'current': self.current_phase,
                'durations': self.phase_steps.copy()
            },
            'recommendations': self.coherence_calc.should_adjust_training()
        }
    
    def export_coherence_report(self, filepath: str):
        """Export detailed coherence report to file"""
        import json
        
        report = {
            'timestamp': time.time(),
            'scheduler_config': {
                'base_lr': self.base_lr,
                'min_lr': self.min_lr,
                'max_lr': self.max_lr,
                'coherence_target': self.coherence_target,
                'adjustment_frequency': self.adjustment_frequency
            },
            'final_state': self.get_scheduler_stats(),
            'adjustment_history': list(self.adjustment_history),
            'coherence_calculator': {
                'window_size': self.coherence_calc.window_size,
                'weights': {
                    'specialization': self.coherence_calc.specialization_weight,
                    'convergence': self.coherence_calc.convergence_weight,
                    'stability': self.coherence_calc.stability_weight
                }
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Coherence report exported to {filepath}")


# Export classes
__all__ = [
    'CoherenceCalculator',
    'CoherenceBasedScheduler'
]