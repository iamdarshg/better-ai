"""Enhanced TUI and Coherence-based Scheduler for MoE Training"""

import os
import sys
import time
import threading
from typing import Dict, List, Optional, Any, Callable
import torch
import torch.nn as nn
import time
import os
from typing import Dict, List, Optional, Any, Union
from collections import deque
import json
import numpy as np


class ColoredText:
    """ANSI color codes for terminal output"""
    
    # Check if terminal supports colors
    _supports_color = (
        hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and
        os.environ.get('TERM') != 'dumb' and
        not os.environ.get('NO_COLOR')
    )
    
    # Colors
    RESET = '\033[0m' if _supports_color else ''
    RED = '\033[91m' if _supports_color else ''
    GREEN = '\033[92m' if _supports_color else ''
    YELLOW = '\033[93m' if _supports_color else ''
    BLUE = '\033[94m' if _supports_color else ''
    MAGENTA = '\033[95m' if _supports_color else ''
    CYAN = '\033[96m' if _supports_color else ''
    WHITE = '\033[97m' if _supports_color else ''
    BOLD = '\033[1m' if _supports_color else ''
    
    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        return f"{color}{text}{cls.RESET}"
    
    @classmethod
    def success(cls, text: str) -> str:
        return cls.colorize(text, cls.GREEN)
    
    @classmethod
    def warning(cls, text: str) -> str:
        return cls.colorize(text, cls.YELLOW)
    
    @classmethod
    def error(cls, text: str) -> str:
        return cls.colorize(text, cls.RED)
    
    @classmethod
    def info(cls, text: str) -> str:
        return cls.colorize(text, cls.BLUE)
    
    @classmethod
    def header(cls, text: str) -> str:
        return cls.colorize(cls.BOLD + text, cls.CYAN)


class MoETrainingTUI:
    """Advanced Terminal User Interface for MoE Training"""
    
    def __init__(
        self,
        update_frequency: int = 1,
        save_frequency: int = 100,
        log_file: Optional[str] = None,
        show_plots: bool = False
    ):
        self.update_frequency = update_frequency
        self.save_frequency = save_frequency
        self.log_file = log_file
        self.show_plots = show_plots
        
        # Training state
        self.is_training = False
        self.should_exit = False
        self.is_paused = False
        
        # Metrics tracking
        self.metrics_history = {
            'loss': deque(maxlen=1000),
            'aux_loss': deque(maxlen=1000),
            'lr': deque(maxlen=1000),
            'expert_utilization': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'gradient_norm': deque(maxlen=1000),
            'throughput': deque(maxlen=200),
            'coherence_score': deque(maxlen=1000)
        }
        
        # Current state
        self.current_step = 0
        self.total_steps = 0
        self.current_epoch = 0
        self.batch_time = 0
        self.start_time = time.time()
        
        # Expert stats
        self.expert_specialization = {}
        self.expert_loads = {}
        
        # TUI state
        self.last_update_time = time.time()
        self.refresh_rate = 0.1  # 100ms refresh
        
    def start_training_ui(self, total_steps: int = 0):
        """Initialize and start the training UI"""
        self.total_steps = total_steps
        self.is_training = True
        self.start_time = time.time()
        
        # Clear screen and show header
        self._clear_screen()
        self._show_header()
        
        # Start background update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        # Setup keyboard handling
        self._setup_keyboard_handlers()
        
        try:
            print(ColoredText.success("✅ Training UI Started!"))
            print("Press 'h' for help, 'q' to quit, 'p' to pause/resume")
        except UnicodeEncodeError:
            # Fallback for Windows terminal with limited Unicode support
            print(ColoredText.success("[OK] Training UI Started!"))
            print("Press 'h' for help, 'q' to quit, 'p' to pause/resume")
    
    def update_metrics(
        self,
        step: int,
        loss: float,
        aux_loss: float,
        lr: float,
        expert_stats: Optional[Dict] = None,
        memory_usage: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        throughput: Optional[float] = None,
        coherence_score: Optional[float] = None,
        batch_time: Optional[float] = None
    ):
        """Update training metrics"""
        self.current_step = step
        self.batch_time = batch_time or 0
        
        # Update metrics history
        self.metrics_history['loss'].append(loss)
        self.metrics_history['aux_loss'].append(aux_loss)
        self.metrics_history['lr'].append(lr)
        self.metrics_history['gradient_norm'].append(gradient_norm or 0)
        self.metrics_history['memory_usage'].append(memory_usage or 0)
        self.metrics_history['throughput'].append(throughput or 0)
        self.metrics_history['coherence_score'].append(coherence_score or 0)
        
        # Update expert stats
        if expert_stats:
            self.expert_specialization = expert_stats.get('specialization', {})
            self.expert_loads = expert_stats.get('loads', {})
            
            # Calculate expert utilization
            if self.expert_loads:
                expert_util = np.mean(list(self.expert_loads.values())) if self.expert_loads else 0
                self.metrics_history['expert_utilization'].append(expert_util)
        
        # Calculate epoch
        if self.total_steps > 0:
            self.current_epoch = (step // (self.total_steps // 10)) + 1
        
        # Log to file if configured
        if self.log_file and step % self.save_frequency == 0:
            self._save_metrics_to_file()
    
    def _update_loop(self):
        """Background thread for updating the UI"""
        while self.is_training and not self.should_exit:
            try:
                if not self.is_paused:
                    self._refresh_display()
                time.sleep(self.refresh_rate)
            except Exception as e:
                # Don't let the thread crash on display errors
                if ColoredText._supports_color:
                    print(f"\r{ColoredText.error(f'TUI Error: {e}')}", flush=True)
                break
    
    def _clear_screen(self):
        """Clear the terminal screen"""
        if ColoredText._supports_color:
            os.system('cls' if os.name == 'nt' else 'clear')
        else:
            # Just print newlines when colors aren't supported
            print('\n' * 5)
    
    def _show_header(self):
        """Show fixed header section"""
        elapsed = time.time() - self.start_time
        progress = (self.current_step / max(self.total_steps, self.current_step)) * 100 if self.total_steps > 0 else 0
        
        header = f"""
{ColoredText.header('MoE Training Dashboard')}
{'=' * 80}
{ColoredText.info('Step:')} {self.current_step:>8} / {self.total_steps or '∞':>8} | {ColoredText.info('Progress:')} {progress:>6.1f}%
{ColoredText.info('Time:')} {self._format_time(elapsed):>10} | {ColoredText.info('Epoch:')} {self.current_epoch:>4}
{ColoredText.info('Status:')} {ColoredText.success('RUNNING' if not self.is_paused else ColoredText.warning('PAUSED'))}
{'=' * 80}
"""
        print(header, end='', flush=True)
    
    def _refresh_display(self):
        """Refresh the entire display"""
        # Clear screen and show header
        self._clear_screen()
        self._show_header()
        
        # Show metrics sections
        self._show_training_metrics()
        self._show_expert_stats()
        self._show_performance_metrics()
        self._show_coherence_metrics()
        self._show_controls()
    
    def _show_training_metrics(self):
        """Show core training metrics"""
        if len(self.metrics_history['loss']) == 0:
            return
        
        recent_loss = list(self.metrics_history['loss'])[-10:]
        recent_aux_loss = list(self.metrics_history['aux_loss'])[-10:]
        recent_lr = list(self.metrics_history['lr'])[-1] if self.metrics_history['lr'] else [0]
        recent_grad_norm = list(self.metrics_history['gradient_norm'])[-10:]
        
        # Calculate trends
        loss_trend = self._calculate_trend(recent_loss)
        grad_trend = self._calculate_trend(recent_grad_norm)
        
        metrics = f"""
{ColoredText.header('📊 Training Metrics')}
┌─────────────────────────────────────────────────────────────────────────────────┐
│ {ColoredText.info('Loss:')} {recent_loss[-1]:.6f} ({ColoredText.success('↓') if loss_trend < 0 else ColoredText.error('↑') if loss_trend > 0 else '→'} {abs(loss_trend):.2e})    
│ {ColoredText.info('Aux Loss:')} {recent_aux_loss[-1]:.6f}                                 
│ {ColoredText.info('Learning Rate:')} {recent_lr[0]:.2e}                                                  
│ {ColoredText.info('Grad Norm:')} {recent_grad_norm[-1]:.3f} ({ColoredText.success('↓') if grad_trend < 0 else ColoredText.error('↑') if grad_trend > 0 else '→'} {abs(grad_trend):.2e})  
└─────────────────────────────────────────────────────────────────────────────────┘
"""
        print(metrics, end='', flush=True)
    
    def _show_expert_stats(self):
        """Show expert specialization and utilization"""
        if not self.expert_loads:
            return
        
        # Expert utilization
        current_utilization = list(self.metrics_history['expert_utilization'])[-1] if self.metrics_history['expert_utilization'] else [0]
        
        # Simplified expert info without complex calculations
        expert_info = f"""
{ColoredText.header('Expert Statistics')}
┌─────────────────────────────────────────────────────────────────────────────────┐
│ {ColoredText.info('Utilization:')} {current_utilization[0]:.1%}                                                         
│ {ColoredText.info('Load Balance:')} {self._calculate_load_balance():.3f}                                                       
│ {ColoredText.info('Active Experts:')} {len(self.expert_loads)}/{len(self.expert_loads)}              
└─────────────────────────────────────────────────────────────────────────────────┘
"""
        
        print(expert_info, end='', flush=True)
    
    def _show_performance_metrics(self):
        """Show performance and system metrics"""
        recent_memory = list(self.metrics_history['memory_usage'])[-10:] if self.metrics_history['memory_usage'] else [0]
        recent_throughput = list(self.metrics_history['throughput'])[-10:] if self.metrics_history['throughput'] else [0]
        
        memory_gb = recent_memory[-1] if recent_memory else 0
        throughput = recent_throughput[-1] if recent_throughput else 0
        
        performance = f"""
{ColoredText.header('⚡ Performance Metrics')}
┌─────────────────────────────────────────────────────────────────────────────────┐
│ {ColoredText.info('Memory Usage:')} {memory_gb:.2f}GB {ColoredText.success('OK') if memory_gb < 8 else ColoredText.warning('WARN') if memory_gb < 12 else ColoredText.error('HIGH')}          
│ {ColoredText.info('Throughput:')} {throughput:.0f} tokens/sec                                             
│ {ColoredText.info('Batch Time:')} {self.batch_time:.3f}s                                                 
│ {ColoredText.info('GPU Memory:')} {self._get_gpu_memory_info()}                                                        
└─────────────────────────────────────────────────────────────────────────────────┘
"""
        print(performance, end='', flush=True)
    
    def _show_coherence_metrics(self):
        """Show coherence-based training metrics"""
        if len(self.metrics_history['coherence_score']) == 0:
            return
        
        recent_coherence = list(self.metrics_history['coherence_score'])[-20:]
        current_coherence = float(recent_coherence[-1])
        coherence_trend = self._calculate_trend(recent_coherence)
        
        # Coherence status
        if current_coherence > 0.8:
            coherence_status = ColoredText.success('EXCELLENT')
        elif current_coherence > 0.6:
            coherence_status = ColoredText.success('GOOD')
        elif current_coherence > 0.4:
            coherence_status = ColoredText.warning('MODERATE')
        else:
            coherence_status = ColoredText.error('LOW')
        
        coherence_info = f"""
{ColoredText.header('🧠 Coherence Metrics')}
┌─────────────────────────────────────────────────────────────────────────┐
│ {ColoredText.info('Coherence:')} {current_coherence:.3f} ({coherence_status}) ({ColoredText.success('↑') if coherence_trend > 0 else ColoredText.error('↓') if coherence_trend < 0 else '→'} {abs(coherence_trend):.3f})
│ {ColoredText.info('Training Phase:')} {self._get_training_phase(current_coherence)}                              
│ {ColoredText.info('Stability:')} {self._calculate_stability():.3f}                                         
│ {ColoredText.info('Convergence:')} {self._calculate_convergence():.3f}                                      
└─────────────────────────────────────────────────────────────────────────────────┘
"""
        print(coherence_info, end='', flush=True)
    
    def _show_controls(self):
        """Show control instructions"""
        controls = f"""
{ColoredText.header('🎮 Controls')}
┌─────────────────────────────────────────────────────────────────────────────────┐
│ [h] Help  [q] Quit  [p] Pause/Resume  [s] Save Stats  [r] Reset View              
│ [l] Toggle Logs  [e] Expert Details  [c] Coherence Details                              
└─────────────────────────────────────────────────────────────────────────────────┘
"""
        print(controls, end='', flush=True)
    
    def _setup_keyboard_handlers(self):
        """Setup keyboard input handlers"""
        if not sys.stdout.isatty():
            return
        import keyboard
        
        def on_key_press(event):
            if event.name == 'q':
                self.should_exit = True
                self.is_training = False
                print(f"\n{ColoredText.warning('Training stopped by user')}")
            elif event.name == 'p':
                self.is_paused = not self.is_paused
                status = ColoredText.warning('PAUSED') if self.is_paused else ColoredText.success('RESUMED')
                print(f"\nTraining {status}")
            elif event.name == 's':
                self._save_metrics_to_file()
                print(f"\n{ColoredText.success('Metrics saved to file')}")
            elif event.name == 'r':
                self._reset_view()
        
        # Start keyboard listener
        keyboard.on_press(on_key_press)
        # Note: Signal handlers already handle graceful shutdown
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (positive = increasing, negative = decreasing)"""
        if len(values) < 3:
            return 0.0
        
# Simple linear regression for trend
        x = np.arange(len(values))
        y = np.array(values)
        
        if len(y) > 1 and np.std(y) > 1e-8:
            slope = float(np.polyfit(x, y, 1)[0])
            return slope
        return 0.0
    
    def _calculate_load_balance(self) -> float:
        """Calculate expert load balance (1.0 = perfect)"""
        if not self.expert_loads:
            return 1.0
        
        loads = list(self.expert_loads.values())
        if len(loads) == 0:
            return 1.0
        
        mean_load = float(np.mean(loads))
        std_load = float(np.std(loads))
        
        # Load balance score (higher is better)
        if mean_load > 0:
            balance_score = 1.0 - (std_load / mean_load)
            return max(0.0, min(1.0, balance_score))
        return 0.0
    
    def _get_gpu_memory_info(self) -> str:
        """Get GPU memory usage info"""
        if not torch.cuda.is_available():
            return "N/A"
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        percentage = (allocated / total) * 100
        return f"{allocated:.1f}/{total:.1f}GB ({percentage:.1f}%)"
    
    def _get_training_phase(self, coherence: float) -> str:
        """Determine training phase based on coherence"""
        if coherence > 0.8:
            return "FINE_TUNING"
        elif coherence > 0.6:
            return "STABILIZING" 
        elif coherence > 0.4:
            return "LEARNING"
        else:
            return "EXPLORING"
    
    def _calculate_stability(self) -> float:
        """Calculate training stability based on recent metrics"""
        if len(self.metrics_history['loss']) < 20:
            return 0.0
        
        recent_losses = list(self.metrics_history['loss'])[-20:]
        if len(recent_losses) == 0:
            return 0.0
        
        # Stability = 1 - (variance / mean)
        mean_loss = np.mean(recent_losses)
        variance = np.var(recent_losses)
        
        if mean_loss > 0:
            stability = max(0.0, 1.0 - (variance / (mean_loss ** 2)))
            return float(stability)
        return 0.0
    
    def _calculate_convergence(self) -> float:
        """Calculate convergence score"""
        if len(self.metrics_history['loss']) < 50:
            return 0.0
        
        recent_losses = list(self.metrics_history['loss'])[-50:]
        if len(recent_losses) < 10:
            return 0.0
        
        # Convergence based on loss reduction over recent window
        old_losses = recent_losses[:25]
        new_losses = recent_losses[25:]
        
        if len(old_losses) > 0 and len(new_losses) > 0:
            old_mean = float(np.mean(old_losses))
            new_mean = float(np.mean(new_losses))
            
            if old_mean > 0:
                improvement = float((old_mean - new_mean) / old_mean)
                return max(0.0, min(1.0, improvement))
        return 0.0
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def _save_metrics_to_file(self):
        """Save current metrics to file"""
        if not self.log_file:
            return

        # Convert tensors to serializable types
        def convert_to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist() if obj.numel() > 1 else obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            elif isinstance(obj, (float, int, str, bool)):
                return obj
            elif obj is None:
                return None
            else:
                try:
                    return float(obj)
                except (ValueError, TypeError):
                    return str(obj)

        metrics = {
            'timestamp': time.time(),
            'step': self.current_step,
            'epoch': self.current_epoch,
            'training_metrics': {
                'loss': list(self.metrics_history['loss'])[-1] if self.metrics_history['loss'] else 0,
                'aux_loss': list(self.metrics_history['aux_loss'])[-1] if self.metrics_history['aux_loss'] else 0,
                'learning_rate': list(self.metrics_history['lr'])[-1] if self.metrics_history['lr'] else 0,
                'gradient_norm': list(self.metrics_history['gradient_norm'])[-1] if self.metrics_history['gradient_norm'] else 0,
            },
            'expert_stats': {
                'specialization': convert_to_serializable(self.expert_specialization),
                'loads': convert_to_serializable(self.expert_loads),
                'utilization': list(self.metrics_history['expert_utilization'])[-1] if self.metrics_history['expert_utilization'] else 0,
                'load_balance': self._calculate_load_balance()
            },
            'performance': {
                'memory_usage_gb': list(self.metrics_history['memory_usage'])[-1] if self.metrics_history['memory_usage'] else 0,
                'throughput': list(self.metrics_history['throughput'])[-1] if self.metrics_history['throughput'] else 0,
                'batch_time': self.batch_time
            },
            'coherence': {
                'score': list(self.metrics_history['coherence_score'])[-1] if self.metrics_history['coherence_score'] else 0,
                'phase': self._get_training_phase(
                    list(self.metrics_history['coherence_score'])[-1] if self.metrics_history['coherence_score'] else 0
                ),
                'stability': self._calculate_stability(),
                'convergence': self._calculate_convergence()
            }
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(convert_to_serializable(metrics), indent=2) + '\n')
    
    def _reset_view(self):
        """Reset the UI view"""
        self._clear_screen()
        self._show_header()
        print(f"{ColoredText.success('View reset!')}")
    
    def stop_training_ui(self):
        """Stop the training UI"""
        self.is_training = False
        self.should_exit = True
        
        if hasattr(self, 'update_thread') and self.update_thread.is_alive():
            # Give the thread more time to finish
            self.update_thread.join(timeout=2.0)
            if self.update_thread.is_alive():
                # Force cleanup if thread is still alive
                print(f"{ColoredText.warning('Warning: UI thread did not shut down cleanly')}")
        
        # Restore terminal state if colors were used
        if ColoredText._supports_color:
            print(ColoredText.RESET, end='', flush=True)
        
        print(f"\n{ColoredText.success('Training UI stopped')}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        return {
            'total_steps': self.current_step,
            'total_time': time.time() - self.start_time,
            'final_metrics': {
                'loss': list(self.metrics_history['loss'])[-1] if self.metrics_history['loss'] else 0,
                'coherence': list(self.metrics_history['coherence_score'])[-1] if self.metrics_history['coherence_score'] else 0,
                'expert_utilization': list(self.metrics_history['expert_utilization'])[-1] if self.metrics_history['expert_utilization'] else 0,
            },
            'training_efficiency': {
                'avg_throughput': np.mean(list(self.metrics_history['throughput'])) if self.metrics_history['throughput'] else 0,
                'avg_batch_time': np.mean([self.batch_time]) if self.batch_time > 0 else 0,
                'stability_score': self._calculate_stability(),
                'convergence_score': self._calculate_convergence()
            }
        }


# Export classes
__all__ = [
    'ColoredText',
    'MoETrainingTUI'
]
