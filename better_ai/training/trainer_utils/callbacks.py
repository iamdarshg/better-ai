
import torch
import time
import os
from typing import Dict, Any
from ..tui import ColoredText

def _should_log_step(self) -> bool:
    """Determine if current step should be logged"""
    log_frequency = getattr(self.config, 'log_every_n_steps', 50)
    return self.global_step % log_frequency == 0

def _should_early_stop(self) -> bool:
    """Enhanced early stopping criteria"""
    max_steps = getattr(self.config, 'max_steps', 10000)

    # Step limit
    if self.global_step >= max_steps:
        return True

    # Coherence-based early stopping
    if self.use_enhanced_features and self.early_stop_triggered:
        return True

    # Loss-based early stopping
    if len(self.metrics_history['loss']) > 100:
        recent_losses = list(self.metrics_history['loss'])[-20:]
        if all(abs(loss - recent_losses[-1]) < 1e-6 for loss in recent_losses[-5:]):
            print(f"{ColoredText.warning('Early stopping: loss plateaued')}")
            return True

    return False

def _enhanced_logging(self, batch_idx: int):
    """Enhanced logging with all metrics"""
    recent_loss = list(self.metrics_history['loss'])[-10:] if len(self.metrics_history['loss']) > 0 else [0]
    recent_aux_loss = list(self.metrics_history['aux_loss'])[-10:] if len(self.metrics_history['aux_loss']) > 0 else [0]
    avg_lr = list(self.metrics_history['learning_rate'])[-1] if self.metrics_history['learning_rate'] else [0]
    memory_gb = list(self.metrics_history['memory_usage'])[-1] if self.metrics_history['memory_usage'] else [0]
    throughput = list(self.metrics_history['throughput'])[-1] if self.metrics_history['throughput'] else [0]

    print(f"\\n{ColoredText.header('📊 Enhanced Training Progress (Step ' + str(self.global_step) + ')')}")
    print(f"{'='*80}")
    print(f"{ColoredText.info('Batch:')} {batch_idx:>6} | {ColoredText.info('Loss:')} {recent_loss[-1]:.6f} | {ColoredText.info('Aux Loss:')} {recent_aux_loss[-1]:.6f}")
    print(f"{ColoredText.info('LR:')} {avg_lr:.2e} | {ColoredText.info('Memory:')} {memory_gb:.2f}GB | {ColoredText.info('Throughput:')} {throughput:.0f} tok/s")

    if self.use_enhanced_features:
        # Expert specialization metrics
        expert_stats = self.expert_manager.get_expert_stats()
        if 'specialization_scores' in expert_stats:
            spec_scores = expert_stats['specialization_scores']
            avg_spec = torch.mean(spec_scores).item()
            max_spec = torch.max(spec_scores).item()
            print(f"{ColoredText.info('Expert Spec:')} {avg_spec:.3f} avg, {max_spec:.3f} max")

        # Coherence metrics
        if hasattr(self.coherence_scheduler, 'coherence_calc'):
            coherence = self.coherence_scheduler.coherence_calc.current_coherence
            coherence_status = self.coherence_scheduler.coherence_calc.get_coherence_status()
            print(f"{ColoredText.info('Coherence:')} {coherence:.3f} ({coherence_status})")

    print(f"{'='*80}")

def _get_final_results(self) -> Dict[str, Any]:
    """Get comprehensive training results"""
    total_time = time.time() - self.start_time

    results = {
        'total_steps': self.global_step,
        'total_time': total_time,
        'best_loss': self.best_loss,
        'early_stopped': self.early_stop_triggered,
        'final_metrics': {}
    }

    if len(self.metrics_history['loss']) > 0:
        results['final_metrics']['loss'] = list(self.metrics_history['loss'])[-1]
        results['final_metrics']['avg_loss'] = sum(list(self.metrics_history['loss'])) / len(list(self.metrics_history['loss']))

    if self.use_enhanced_features:
        results['enhanced_metrics'] = {
            'expert_specialization': self.expert_manager.get_expert_stats(),
            'checkpointing_stats': self.checkpoint_manager.get_checkpoint_stats(),
            'capacity_stats': self.capacity_manager.get_capacity_stats(),
            'attention_stats': self.attention_selector.get_attention_recommendations(),
            'coherence_stats': self.coherence_scheduler.get_scheduler_stats(),
            'memory_stats': self.memory_manager.get_memory_stats()
        }

        # Save comprehensive report
        self.training_ui.get_training_summary()

        # Export coherence report
        self.coherence_scheduler.export_coherence_report(
            f"{self.save_dir}/coherence_report.json"
        )

    return results

def save_checkpoint(self, filepath: str):
    """Enhanced checkpoint saving with optimization state"""
    checkpoint = {
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'global_step': self.global_step,
        'best_loss': self.best_loss,
        'config': self.config
    }

    if self.use_enhanced_features:
        checkpoint['enhanced_state'] = {
            'expert_manager': self.expert_manager.get_expert_stats(),
            'checkpointing_stats': self.checkpoint_manager.get_checkpoint_stats(),
            'capacity_manager': self.capacity_manager.get_capacity_stats(),
            'coherence_scheduler': self.coherence_scheduler.get_scheduler_stats(),
            'metrics_history': {
                key: list(values)[-100:] if len(values) > 0 else []
                for key, values in self.metrics_history.items()
            }
        }

    torch.save(checkpoint, filepath)
    print(f"{ColoredText.success(f'Checkpoint saved: {filepath}')}")

def load_checkpoint(self, filepath: str):
    """Enhanced checkpoint loading"""
    if not os.path.exists(filepath):
        print(f"{ColoredText.error(f'Checkpoint not found: {filepath}')}")
        return False

    checkpoint = torch.load(filepath, map_location=self.device)

    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.global_step = checkpoint.get('global_step', 0)
    self.best_loss = checkpoint.get('best_loss', float('inf'))

    # Load enhanced state
    if 'enhanced_state' in checkpoint and self.use_enhanced_features:
        enhanced_state = checkpoint['enhanced_state']

        if 'expert_manager' in enhanced_state:
            # Restore expert manager state (simplified)
            pass  # Would need proper state restoration

        if 'metrics_history' in enhanced_state:
            # Restore metrics history
            for key, values in enhanced_state['metrics_history'].items():
                self.metrics_history[key].extend(values)

    self.checkpoint_loaded = True
    print(f"{ColoredText.success(f'Checkpoint loaded: {filepath}')}")
    return True
