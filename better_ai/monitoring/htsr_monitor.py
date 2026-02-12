"""HTSR Monitoring Module for Grokking Detection

This module provides functionality to monitor the Hurst exponent (α) of weight matrices
during training to detect grokking patterns and prevent excessive memorization.

α > 4.5 indicates over-grokking / excessive memorization
High α variance across layers indicates instability
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import kstest
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import logging
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log classification levels for HTSR monitoring."""
    SEVERE = "SEVERE"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    STATUS = "STATUS"


@dataclass
class HTSRLog:
    """Structured log entry for HTSR monitoring."""
    level: LogLevel
    message: str
    timestamp: float
    step: int
    details: Dict[str, Any] = None
    layer_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'level': self.level.value,
            'message': self.message,
            'timestamp': self.timestamp,
            'step': self.step,
            'details': self.details or {},
            'layer_name': self.layer_name
        }


class CommunicationStub:
    """Stub for sending communications when severe alerts are detected."""
    
    def __init__(self):
        self.enabled = True
        self.notification_queue: List[HTSRLog] = []
        
    def send_severe_alert(self, log: HTSRLog) -> bool:
        """Send severe alert via available channels.
        
        Args:
            log: The severe log entry to send
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False
            
        # Queue for sending
        self.notification_queue.append(log)
        
        # Stub implementations for different channels:
        # - Email (SMTP)
        # - Slack/Discord webhook
        # - PagerDuty/OpsGenie
        # - SMS/Twilio
        # - Push notification
        
        logger.info(f"[COMMUNICATION STUB] Severe alert queued: {log.message}")
        
        # TODO: Implement actual communication channels:
        # self._send_email(log)
        # self._send_slack(log)
        # self._send_pagerduty(log)
        
        return True
    
    def get_queued_notifications(self) -> List[Dict[str, Any]]:
        """Get queued notifications for display/debugging."""
        return [log.to_dict() for log in self.notification_queue]
    
    def clear_queue(self):
        """Clear the notification queue."""
        self.notification_queue.clear()


# Layer patterns that have little/no effect on grokking metrics
LAYER_PATTERNS_TO_EXCLUDE = [
    'bias', 'norm', 'activation', 'embed', 'positional', 'layer_norm', 
    'batch_norm', 'dropout', 'pool', 'fc_norm', 'layernorm', 'batchnorm',
    'activation_fn', 'act', 'softmax', 'sigmoid', 'tanh', 'relu', 'gelu',
    'embedding', 'head', 'lm_head', 'wte', 'wpe'
]


# Threshold constants for grokking detection
ALPHA_OVER_GROKKING_THRESHOLD = 4.5
ALPHA_VARIANCE_THRESHOLD = 0.5
ALPHA_VARIANCE_HISTORY_SIZE = 10


def compute_alpha_from_esd(W: np.ndarray) -> float:
    """Compute PL exponent α from empirical spectral density.
    
    Uses power-law fitting on the singular value distribution to estimate
    the spectral exponent α. Research shows α > 4.5 indicates over-grokking.
    
    Args:
        W: Weight matrix as numpy array
        
    Returns:
        Estimated PL exponent α
    """
    if W is None or len(W.shape) < 2:
        return 2.0
    
    if W.shape[0] < 10 or W.shape[1] < 10:
        return 2.0
    
    try:
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        
        if len(s) == 0 or np.sum(s) == 0:
            return 2.0
        
        s_normalized = s / np.sum(s)
        
        if np.sum(s_normalized) == 0:
            return 2.0
        
        s_positive = s_normalized[s_normalized > 0]
        
        if len(s_positive) < 5:
            return 2.0
        
        log_s = np.log(s_positive)
        
        if len(log_s) < 5 or np.std(log_s) == 0:
            return 2.0
        
        n_tail = max(int(len(s) * 0.3), 3)
        s_tail = np.sort(s)[-n_tail:]
        
        if len(s_tail) < 3:
            return 2.0
        
        log_s_tail = np.log(s_tail)
        ranks = np.arange(1, len(s_tail) + 1)
        log_ranks = np.log(ranks)
        
        try:
            slope, intercept = np.polyfit(log_ranks, log_s_tail, 1)
            
            if slope >= 0:
                return 2.0
            
            alpha = -1.0 / slope
            alpha = max(1.0, min(10.0, alpha))
            
            return alpha
            
        except (np.linalg.LinAlgError, ValueError):
            if np.mean(s) > 0:
                ratio = np.max(s) / np.mean(s)
                alpha = min(10.0, max(1.0, 2.0 + np.log10(ratio)))
                return alpha
            
            return 2.0
            
    except Exception as e:
        logger.debug(f"Error computing α: {e}")
        return 2.0


def detect_correlation_traps(W: np.ndarray, n_random_trials: int = 10) -> Tuple[bool, float]:
    """Detect outlier singular values via randomization test.
    
    Args:
        W: Weight matrix as numpy array
        n_random_trials: Number of randomization trials
        
    Returns:
        Tuple of (has_outliers: bool, p_value: float)
    """
    if W is None or len(W.shape) < 2:
        return False, 1.0
    
    if W.shape[0] < 10 or W.shape[1] < 10:
        return False, 1.0
    
    try:
        U, s_orig, Vt = np.linalg.svd(W, full_matrices=False)
        
        s_random_collection = []
        for _ in range(n_random_trials):
            W_random = np.random.permutation(W.flatten()).reshape(W.shape)
            _, s_random, _ = np.linalg.svd(W_random, full_matrices=False)
            s_random_collection.append(s_random)
        
        s_random_flat = np.concatenate(s_random_collection)
        
        if np.sum(s_orig) > 0:
            s_orig_norm = s_orig / np.sum(s_orig)
        else:
            return False, 1.0
            
        if np.sum(s_random_flat) > 0:
            s_random_norm = s_random_flat / np.sum(s_random_flat)
        else:
            return False, 1.0
        
        try:
            statistic, pvalue = kstest(s_orig_norm, s_random_flat)
        except (ValueError, FloatingPointError):
            pvalue = 0.5
        
        return pvalue < 0.05, pvalue
        
    except Exception as e:
        logger.debug(f"Error in correlation trap detection: {e}")
        return False, 1.0


class LayerAlphaAnalyzer:
    """Analyzes α for individual layers with filtering."""
    
    def __init__(
        self, 
        alpha_upper_threshold: float = ALPHA_OVER_GROKKING_THRESHOLD
    ):
        self.alpha_upper_threshold = alpha_upper_threshold
        self.layer_history: Dict[str, deque] = {}
        self.layer_variance: Dict[str, float] = {}
        
    def should_monitor_layer(self, layer_name: str, module: nn.Module) -> bool:
        """Check if a layer should be monitored based on patterns."""
        name_lower = layer_name.lower()
        
        for pattern in LAYER_PATTERNS_TO_EXCLUDE:
            if pattern in name_lower:
                return False
        
        if not hasattr(module, 'weight') or module.weight is None:
            return False
        
        if len(module.weight.shape) < 2:
            return False
        
        if module.weight.shape[0] < 10 or module.weight.shape[1] < 10:
            return False
        
        return True
    
    def analyze_layer(self, layer_name: str, module: nn.Module) -> Optional[Dict[str, Any]]:
        """Analyze a single layer's weight matrix."""
        if not self.should_monitor_layer(layer_name, module):
            return None
        
        try:
            W = module.weight.data.cpu().numpy()
            
            alpha = compute_alpha_from_esd(W)
            is_trap, p_value = detect_correlation_traps(W)
            
            result = {
                'layer_name': layer_name,
                'alpha': alpha,
                'is_correlation_trap': is_trap,
                'p_value': p_value,
                'shape': list(W.shape),
                'timestamp': time.time()
            }
            
            if layer_name not in self.layer_history:
                self.layer_history[layer_name] = deque(maxlen=100)
            
            self.layer_history[layer_name].append(alpha)
            
            # Calculate variance
            if len(self.layer_history[layer_name]) >= 3:
                history_list = list(self.layer_history[layer_name])
                self.layer_variance[layer_name] = np.var(history_list)
            else:
                self.layer_variance[layer_name] = 0.0
            
            return result
            
        except Exception as e:
            logger.debug(f"Error analyzing layer {layer_name}: {e}")
            return None
    
    def get_over_grokking_layers(self, analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """Get layers with α above threshold (over-grokking)."""
        over_grokking = {}
        for layer_name, result in analysis_results.items():
            if result and result['alpha'] > self.alpha_upper_threshold:
                over_grokking[layer_name] = result['alpha']
        return over_grokking
    
    def get_high_variance_layers(self, variance_threshold: float = ALPHA_VARIANCE_THRESHOLD) -> Dict[str, float]:
        """Get layers with high α variance."""
        high_variance = {}
        for layer_name, variance in self.layer_variance.items():
            if variance > variance_threshold:
                high_variance[layer_name] = variance
        return high_variance


class HTSRLogger:
    """Structured logger for HTSR monitoring with classification."""
    
    def __init__(self, comm_stub: CommunicationStub = None):
        self.comm_stub = comm_stub or CommunicationStub()
        self.log_history: deque = deque(maxlen=500)
        
    def log(
        self,
        level: LogLevel,
        message: str,
        step: int,
        details: Dict[str, Any] = None,
        layer_name: str = None
    ) -> HTSRLog:
        """Create and store a log entry."""
        log_entry = HTSRLog(
            level=level,
            message=message,
            timestamp=time.time(),
            step=step,
            details=details,
            layer_name=layer_name
        )
        
        self.log_history.append(log_entry)
        
        # Log to standard logger
        log_method = getattr(logger, level.value.lower(), logger.info)
        log_method(f"[{level.value}] Step {step}: {message}")
        
        # Send severe alerts via communication stub
        if level == LogLevel.SEVERE:
            self.comm_stub.send_severe_alert(log_entry)
        
        return log_entry
    
    def severe(self, message: str, step: int, details: Dict[str, Any] = None, layer_name: str = None):
        """Log severe event."""
        self.log(LogLevel.SEVERE, message, step, details, layer_name)
    
    def major(self, message: str, step: int, details: Dict[str, Any] = None, layer_name: str = None):
        """Log major event."""
        self.log(LogLevel.MAJOR, message, step, details, layer_name)
    
    def minor(self, message: str, step: int, details: Dict[str, Any] = None, layer_name: str = None):
        """Log minor event."""
        self.log(LogLevel.MINOR, message, step, details, layer_name)
    
    def status(self, message: str, step: int, details: Dict[str, Any] = None):
        """Log status update."""
        self.log(LogLevel.STATUS, message, step, details)
    
    def get_logs_by_level(self, level: LogLevel) -> List[HTSRLog]:
        """Get all logs of a specific level."""
        return [log for log in self.log_history if log.level == level]
    
    def get_recent_logs(self, n: int = 10) -> List[HTSRLog]:
        """Get recent logs."""
        return list(self.log_history)[-n:]
    
    def get_all_logs(self) -> List[Dict[str, Any]]:
        """Get all logs as dictionaries."""
        return [log.to_dict() for log in self.log_history]


class GrokkingDetector:
    """Detects grokking patterns based on accumulated metrics."""
    
    def __init__(
        self,
        alpha_upper_threshold: float = ALPHA_OVER_GROKKING_THRESHOLD,
        variance_threshold: float = ALPHA_VARIANCE_THRESHOLD,
        variance_history_size: int = ALPHA_VARIANCE_HISTORY_SIZE
    ):
        self.alpha_upper_threshold = alpha_upper_threshold
        self.variance_threshold = variance_threshold
        self.variance_history_size = variance_history_size
        
        self.over_grokking_count = 0
        self.high_variance_count = 0
        self.alpha_history: deque = deque(maxlen=variance_history_size)
        self.alpha_variance_history: deque = deque(maxlen=variance_history_size)
        self.first_over_grokking_step = None
        
    def check(
        self, 
        alpha_metrics: Dict[str, float]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if grokking is detected.
        
        Args:
            alpha_metrics: Dict mapping layer names to α values
            
        Returns:
            Tuple of (is_detected: bool, details: Dict)
        """
        if not alpha_metrics:
            return False, {'reason': 'no_metrics'}
        
        valid_alphas = [α for α in alpha_metrics.values() if α != float('inf')]
        
        if not valid_alphas:
            return False, {'reason': 'no_valid_alphas'}
        
        model_alpha = np.mean(valid_alphas)
        alpha_variance = np.var(valid_alphas)
        
        # Track variance history
        self.alpha_history.append(model_alpha)
        self.alpha_variance_history.append(alpha_variance)
        
        # Find over-grokking layers
        over_grokking_layers = {
            name: α for name, α in alpha_metrics.items() 
            if α > self.alpha_upper_threshold
        }
        self.over_grokking_count = len(over_grokking_layers)
        
        # Find high variance layers
        high_variance_layers = {
            name: α for name, α in alpha_metrics.items()
            if np.abs(α - model_alpha) > 2 * np.sqrt(alpha_variance) if alpha_variance > 0 else False
        }
        self.high_variance_count = len(high_variance_layers)
        
        # Check conditions
        is_over_grokking = model_alpha > self.alpha_upper_threshold
        is_high_variance = alpha_variance > self.variance_threshold
        
        # Both conditions trigger detection
        is_detected = is_over_grokking or is_high_variance
        
        if is_over_grokking and self.first_over_grokking_step is None:
            self.first_over_grokking_step = len(self.alpha_history)
        
        details = {
            'detected': is_detected,
            'model_alpha': model_alpha,
            'alpha_variance': alpha_variance,
            'over_grokking_layers': over_grokking_layers,
            'high_variance_layers': high_variance_layers,
            'over_grokking_count': self.over_grokking_count,
            'high_variance_count': self.high_variance_count,
            'is_over_grokking': is_over_grokking,
            'is_high_variance': is_high_variance
        }
        
        return is_detected, details
    
    def get_variance_trend(self) -> float:
        """Get trend in α variance (positive = increasing)."""
        if len(self.alpha_variance_history) < 3:
            return 0.0
        
        history = list(self.alpha_variance_history)
        recent = history[-3:]
        older = history[:-3]
        
        if len(older) == 0:
            return 0.0
        
        return np.mean(recent) - np.mean(older)


class HTSRMonitor:
    """Main HTSR monitoring interface for grokking detection."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        alpha_upper_threshold: float = ALPHA_OVER_GROKKING_THRESHOLD,
        monitor_interval: int = 75,
        variance_threshold: float = ALPHA_VARIANCE_THRESHOLD,
        verbose: bool = True
    ):
        self.model = model
        self.device = device or torch.device('cpu')
        self.alpha_upper_threshold = alpha_upper_threshold
        self.monitor_interval = monitor_interval
        self.variance_threshold = variance_threshold
        self.verbose = verbose
        
        self.layer_analyzer = LayerAlphaAnalyzer(alpha_upper_threshold)
        self.detector = GrokkingDetector(alpha_upper_threshold, variance_threshold)
        self.logger = HTSRLogger()
        
        self.alpha_history: List[Dict[str, Any]] = []
        self.step_history: List[int] = []
        self.timestamp_history: List[float] = []
        
        self.computation_time = 0.0
        self.total_checks = 0
        
        self.current_alpha_metrics: Dict[str, float] = {}
        self.current_detector_state: Dict[str, Any] = {}
        
    def compute_all_layer_alphas(self) -> Dict[str, Any]:
        """Compute α for all monitorable layers in the model."""
        all_results = {}
        
        total_start = time.time()
        
        for name, module in self.model.named_modules():
            if self.layer_analyzer.should_monitor_layer(name, module):
                result = self.layer_analyzer.analyze_layer(name, module)
                if result:
                    all_results[name] = result
        
        self.computation_time = time.time() - total_start
        self.total_checks += 1
        
        self.current_alpha_metrics = {
            name: result['alpha'] 
            for name, result in all_results.items()
        }
        
        self.alpha_history.append(self.current_alpha_metrics.copy())
        self.step_history.append(self.total_checks)
        self.timestamp_history.append(time.time())
        
        is_detected, details = self.detector.check(self.current_alpha_metrics)
        self.current_detector_state = details
        
        # Log based on detection results
        if details.get('is_over_grokking'):
            self.logger.severe(
                f"Over-grokking detected! Model α={details.get('model_alpha', 0):.2f} > {self.alpha_upper_threshold}",
                self.total_checks,
                {
                    'model_alpha': details.get('model_alpha'),
                    'over_grokking_layers': len(details.get('over_grokking_layers', {})),
                    'layers': list(details.get('over_grokking_layers', {}).keys())
                }
            )
        elif details.get('is_high_variance'):
            self.logger.severe(
                f"High α variance detected! Variance={details.get('alpha_variance', 0):.4f} > {self.variance_threshold}",
                self.total_checks,
                {
                    'variance': details.get('alpha_variance'),
                    'variance_layers': len(details.get('high_variance_layers', {}))
                }
            )
        elif details.get('over_grokking_count', 0) > 0:
            self.logger.major(
                f"{details.get('over_grokking_count')} layers approaching over-grokking threshold",
                self.total_checks,
                {'layer_count': details.get('over_grokking_count')}
            )
        elif details.get('high_variance_count', 0) > 0:
            self.logger.minor(
                f"{details.get('high_variance_count')} layers show variance",
                self.total_checks,
                {'layer_count': details.get('high_variance_count')}
            )
        else:
            self.logger.status(
                f"Model α={details.get('model_alpha', 0):.2f}, variance={details.get('alpha_variance', 0):.4f}",
                self.total_checks
            )
        
        if is_detected and self.verbose:
            logger.warning(
                f"Grokking detected! Model α={details.get('model_alpha', 'N/A'):.2f}, "
                f"Over-grokking layers: {len(details.get('over_grokking_layers', {}))}"
            )
        
        return {
            'layer_results': all_results,
            'alpha_metrics': self.current_alpha_metrics,
            'detector_state': self.current_detector_state,
            'computation_time': self.computation_time,
            'total_checks': self.total_checks
        }
    
    def should_check(self, step: int) -> bool:
        """Determine if a check should be performed at this step."""
        return step % self.monitor_interval == 0
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for the HTML dashboard."""
        variance_trend = self.detector.get_variance_trend()
        
        return {
            'current_alphas': self.current_alpha_metrics,
            'detector_state': self.current_detector_state,
            'alpha_history': self.alpha_history[-100:] if self.alpha_history else [],
            'step_history': self.step_history[-100:] if self.step_history else [],
            'logs': self.logger.get_all_logs()[-100:],
            'statistics': {
                'total_checks': self.total_checks,
                'avg_computation_time': self.computation_time,
                'over_grokking_count': self.detector.over_grokking_count,
                'high_variance_count': self.detector.high_variance_count,
                'current_model_alpha': self.current_detector_state.get('model_alpha', 2.0),
                'current_alpha_variance': self.current_detector_state.get('alpha_variance', 0.0),
                'variance_trend': variance_trend
            },
            'thresholds': {
                'alpha_upper': self.alpha_upper_threshold,
                'variance': self.variance_threshold
            },
            'log_summary': {
                'severe': len(self.logger.get_logs_by_level(LogLevel.SEVERE)),
                'major': len(self.logger.get_logs_by_level(LogLevel.MAJOR)),
                'minor': len(self.logger.get_logs_by_level(LogLevel.MINOR)),
                'status': len(self.logger.get_logs_by_level(LogLevel.STATUS))
            }
        }
    
    def get_layer_summary(self) -> List[Dict[str, Any]]:
        """Get summary of layer α values for display."""
        summary = []
        for name, alpha in sorted(
            self.current_alpha_metrics.items(), 
            key=lambda x: x[1] if isinstance(x[1], (int, float)) else 2.0
        ):
            status = 'severe' if alpha > self.alpha_upper_threshold else 'warning' if alpha > 3.5 else 'healthy'
            summary.append({
                'layer': name,
                'alpha': alpha if alpha != float('inf') else 2.0,
                'status': status
            })
        return summary
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate a comprehensive health report."""
        valid_alphas = [α for α in self.current_alpha_metrics.values() if α != float('inf')]
        
        if not valid_alphas:
            return {'status': 'unknown', 'message': 'No valid α values computed'}
        
        model_alpha = np.mean(valid_alphas)
        alpha_variance = np.var(valid_alphas)
        
        over_grokking = len([α for α in valid_alphas if α > self.alpha_upper_threshold])
        near_threshold = len([α for α in valid_alphas if self.alpha_upper_threshold - 0.5 < α <= self.alpha_upper_threshold])
        
        if over_grokking > 0 or alpha_variance > self.variance_threshold:
            status = 'severe'
        elif near_threshold > len(valid_alphas) * 0.5:
            status = 'warning'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'model_alpha': model_alpha,
            'alpha_variance': alpha_variance,
            'variance_trend': self.detector.get_variance_trend(),
            'over_grokking_layers': over_grokking,
            'near_threshold_layers': near_threshold,
            'total_layers': len(valid_alphas),
            'detector_state': self.current_detector_state
        }
    
    def apply_intervention(
        self,
        intervention_type: str,
        lr_reduction_factor: float = 0.5,
        wd_increase_factor: float = 2.0,
        optimizer = None
    ) -> Dict[str, Any]:
        """Apply intervention to reduce grokking.
        
        Args:
            intervention_type: Type of intervention
            lr_reduction_factor: Factor to reduce learning rate
            wd_increase_factor: Factor to increase weight decay
            optimizer: PyTorch optimizer
            
        Returns:
            Dict with intervention details
        """
        intervention_details = {
            'type': intervention_type,
            'lr_reduction_factor': lr_reduction_factor,
            'wd_increase_factor': wd_increase_factor,
            'timestamp': time.time()
        }
        
        if optimizer is not None:
            current_lrs = []
            current_wds = []
            
            for param_group in optimizer.param_groups:
                current_lrs.append(param_group.get('lr', 0))
                current_wds.append(param_group.get('weight_decay', 0))
                
                param_group['lr'] = param_group.get('lr', 0) * lr_reduction_factor
                if 'weight_decay' in param_group:
                    param_group['weight_decay'] *= wd_increase_factor
            
            intervention_details['previous_lrs'] = current_lrs
            intervention_details['new_lrs'] = [pg.get('lr', 0) for pg in optimizer.param_groups]
            intervention_details['previous_wds'] = current_wds
            intervention_details['new_wds'] = [pg.get('weight_decay', 0) for pg in optimizer.param_groups]
        
        self.logger.major(
            f"Intervention applied: {intervention_type}",
            self.total_checks,
            intervention_details
        )
        
        return intervention_details
