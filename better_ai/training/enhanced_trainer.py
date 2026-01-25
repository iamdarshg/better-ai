"""Enhanced Trainer with All MoE Optimizations Integrated"""

import torch
import torch.nn as nn
import time
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import deque
import json

from .expert_manager import ExpertSpecializationManager, MoETrainingMonitor
from .checkpointing import SelectiveCheckpointManager, AdaptiveMemoryManager
from .adaptive_optimizations import DynamicExpertCapacityManager, AdaptiveAttentionSelector
from .coherence_scheduler import CoherenceBasedScheduler
from .tui import MoETrainingTUI, ColoredText


class EnhancedMoETrainer:
    """
    Enhanced MoE trainer with all optimizations:
    - Expert specialization tracking
    - Selective gradient checkpointing
    - Dynamic expert capacity adjustment
    - Adaptive attention selection
    - Coherence-based scheduling
    - Real-time TUI monitoring
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        eval_dataloader,
        optimizer,
        scheduler,
        config,
        device: torch.device,
        use_enhanced_features: bool = True
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.use_enhanced_features = use_enhanced_features
        
        # Enhanced optimization managers
        if use_enhanced_features:
            # Expert specialization and monitoring
            self.expert_manager = ExpertSpecializationManager(
                num_experts=getattr(config, 'num_experts', 8),
                num_languages=3,
                device=device
            )
            
            self.training_monitor = MoETrainingMonitor(
                num_experts=getattr(config, 'num_experts', 8),
                num_languages=3,
                log_frequency=50,
                save_frequency=500,
                log_dir="./logs"
            )
            
            # Checkpointing and memory management
            self.checkpoint_manager = SelectiveCheckpointManager(
                memory_threshold=0.7,
                checkpoint_frequency=2,
                device=device
            )
            
            self.memory_manager = AdaptiveMemoryManager(
                cleanup_frequency=50,
                memory_target=0.8,
                enable_dynamic_batching=True
            )
            
            # Dynamic optimizations
            self.capacity_manager = DynamicExpertCapacityManager(
                num_experts=getattr(config, 'num_experts', 8),
                base_capacity_factor=getattr(config, 'expert_capacity_factor', 1.25),
                device=device
            )
            
            self.attention_selector = AdaptiveAttentionSelector(
                seq_length_threshold_mla=2048,
                seq_length_threshold_dsa=4096,
                memory_threshold_mla=0.6,
                device=device
            )
            
            # Coherence-based scheduler
            self.coherence_scheduler = CoherenceBasedScheduler(
                base_lr=getattr(config, 'learning_rate', 1e-4),
                coherence_target=0.7,
                adjustment_frequency=50,
                device=device
            )
            
            # Enhanced TUI
            self.training_ui = MoETrainingTUI(
                update_frequency=1,
                save_frequency=100,
                log_file="./logs/enhanced_training.json",
                show_plots=False
            )
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.early_stop_triggered = False
        
        # Metrics tracking
        self.metrics_history = {
            'loss': deque(maxlen=1000),
            'aux_loss': deque(maxlen=1000),
            'learning_rate': deque(maxlen=1000),
            'gradient_norm': deque(maxlen=1000),
            'expert_utilization': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'throughput': deque(maxlen=200),
            'coherence_score': deque(maxlen=1000)
        }
        
        # Performance tracking
        self.step_times = deque(maxlen=1000)
        self.start_time = time.time()
        
        # Checkpoint tracking
        self.checkpoint_loaded = False
        self.save_dir = getattr(config, 'output_dir', './checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train(self) -> Dict[str, Any]:
        """Enhanced training loop with all optimizations"""
        
        if self.use_enhanced_features:
            print(f"\n{ColoredText.success('Enhanced MoE Training Started!')}")
            print(f"{ColoredText.info('Features:')} Expert Specialization + Selective Checkpointing + Dynamic Optimization + Coherence Scheduler")
            print(f"{'='*80}")
            
            # Start TUI
            self.training_ui.start_training_ui(
                total_steps=getattr(self.config, 'max_steps', 10000)
            )
        
        try:
            self.model.train()
            
            # Handle iterable datasets properly - create continuous iterator
            data_iterator = iter(self.train_dataloader)
            batch_idx = 0
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Enhanced batch processing
                step_start_time = time.time()

                # Handle mixed data types (fix for batch collation error)
                batch = self._process_batch(batch)

                # Forward pass with optimizations
                loss, aux_loss, expert_ids = self._enhanced_forward_pass(batch)

                # Backward pass with gradient handling
                loss_total = loss + aux_loss
                loss_total.backward()

                # Gradient clipping and optimization
                grad_norm = self._handle_gradients_and_optimize()

                # Update all optimization managers
                self._update_optimization_managers(
                    loss, aux_loss, grad_norm, expert_ids, batch, step_start_time
                )

                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

                # Enhanced logging and early stopping
                if self._should_log_step():
                    self._enhanced_logging(batch_idx)

                if self._should_early_stop():
                    break

                # Coherence-based early stopping
                if self.use_enhanced_features:
                    coherence_result = self.coherence_scheduler.step(
                        loss=loss.item() if hasattr(loss, 'item') else float(loss),
                        aux_loss=aux_loss.item() if hasattr(aux_loss, 'item') else float(aux_loss),
                        expert_utilization=self._calculate_expert_utilization(expert_ids),
                        gradient_norm=grad_norm,
                        step=self.global_step
                    )
                    
                    if coherence_result['should_stop']:
                        self.early_stop_triggered = True
                        print(f"{ColoredText.warning('Early stopping triggered by coherence scheduler!')}")
                        break
                    
                    if coherence_result['adjusted']:
                        # Update learning rate based on coherence
                        if hasattr(self.optimizer, 'param_groups'):
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = coherence_result['current_lr']
        
        except KeyboardInterrupt:
            print(f"\n{ColoredText.warning('Training interrupted by user!')}")
        except Exception as e:
            print(f"{ColoredText.error(f'Training failed: {e}')}")
            import traceback
            traceback.print_exc()
        
        finally:
            if self.use_enhanced_features:
                self.training_ui.stop_training_ui()
            
            # Save final results
            return self._get_final_results()
    
    def _process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Process batch with type safety (fix for collation error)"""
        processed_batch = {}
        
        for key, value in batch.items():
            if hasattr(value, 'to') and callable(getattr(value, 'to')):
                # Tensor - move to device
                processed_batch[key] = value.to(self.device)
            elif isinstance(value, list):
                # List of strings or other non-tensor data
                processed_batch[key] = value
            else:
                # Try to convert to tensor if possible
                try:
                    if isinstance(value, (int, float)):
                        processed_batch[key] = torch.tensor([value], dtype=torch.long, device=self.device)
                    else:
                        processed_batch[key] = value
                except Exception:
                    processed_batch[key] = value
        
        return processed_batch
    
    def _enhanced_forward_pass(self, batch: Dict[str, Any]) -> tuple:
        """Enhanced forward pass with attention selection"""
        
        # Determine attention type based on sequence length
        input_ids = batch.get('input_ids')
        if input_ids is not None:
            seq_length = input_ids.size(1) if len(input_ids.shape) > 1 else input_ids.size(0)
            memory_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
            
            if self.use_enhanced_features:
                attention_type = self.attention_selector.select_attention_type(
                    seq_length=seq_length,
                    memory_usage=memory_usage
                )
                # Note: In a real implementation, you would switch the model's attention mechanism
                # For now, we just track the selection
                print(f"🧠 Attention Type: {attention_type.upper()} (seq_len={seq_length}, mem={memory_usage:.2f})")
        
        # Filter out labels and other non-model arguments
        model_batch = {k: v for k, v in batch.items() if k not in ['labels', 'pixel_values', 'label_ids']}
        
        # Standard forward pass
        outputs = self.model(**model_batch)
        
        # Extract loss and expert information
        if isinstance(outputs, dict):
            loss = outputs.get('loss', torch.tensor(0.0, device=self.device))
            aux_loss = outputs.get('aux_loss', torch.tensor(0.0, device=self.device))
            expert_ids = outputs.get('expert_ids')
        else:
            # Legacy format
            loss = outputs[0] if len(outputs) > 0 else torch.tensor(0.0, device=self.device)
            aux_loss = outputs[1] if len(outputs) > 1 else torch.tensor(0.0, device=self.device)
            expert_ids = None
        
        # Compute loss from logits if model doesn't compute it
        if loss.item() == 0.0 and 'labels' in batch:
            labels = batch['labels'].to(self.device)
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
        
        return loss, aux_loss, expert_ids
    
    def _handle_gradients_and_optimize(self) -> float:
        """Enhanced gradient handling with clipping"""
        
        # Gradient clipping
        grad_norm = 0.0
        if hasattr(self.config, 'clip_grad_norm') and self.config.clip_grad_norm:
            if hasattr(self.optimizer, 'param_groups'):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for group in self.optimizer.param_groups for p in group['params']],
                    self.config.clip_grad_norm
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()), 
                    self.config.clip_grad_norm
                )
        
        # Optimizer step
        if hasattr(self.optimizer, 'step'):
            if hasattr(self.optimizer, 'param_groups'):
                self.optimizer.step()
            else:
                self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        return grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
    
    def _update_optimization_managers(self, loss, aux_loss, grad_norm, expert_ids, batch, step_start_time):
        """Update all optimization managers with current step data"""
        if not self.use_enhanced_features:
            return
        
        step_time = time.time() - step_start_time
        memory_usage = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        throughput = self._estimate_throughput(batch, step_time)
        language_tokens = batch.get('language', [])
        
        # Convert tensors to python types for compatibility
        loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
        aux_loss_val = aux_loss.item() if hasattr(aux_loss, 'item') else float(aux_loss)
        grad_norm_val = float(grad_norm) if isinstance(grad_norm, (int, float)) else (grad_norm.item() if hasattr(grad_norm, 'item') else 0.0)
        
        # Update expert manager
        if expert_ids is not None:
            self.expert_manager.update_expert_stats(
                expert_ids=expert_ids,
                language_tokens=language_tokens
            )
        
        # Update training monitor
        self.training_monitor.update_step(
            step=self.global_step,
            loss=loss_val,
            aux_loss=aux_loss_val,
            learning_rate=self._get_current_lr(),
            gradient_norm=grad_norm_val,
            expert_ids=expert_ids,
            language_tokens=language_tokens,
            memory_usage=memory_usage,
            batch_time=step_time
        )
        
        # Update memory manager
        self.memory_manager.step(loss_val, grad_norm_val)
        
        # Update capacity manager
        if expert_ids is not None:
            expert_loads = self._calculate_expert_loads(expert_ids)
            # Convert dict to tensor if needed
            if hasattr(self.capacity_manager, 'update_expert_loads'):
                try:
                    # Try to pass as tensor
                    loads_tensor = torch.tensor(list(expert_loads.values()), device=self.device)
                    self.capacity_manager.update_expert_loads(loads_tensor)
                except Exception:
                    # Fallback to dict
                    self.capacity_manager.update_expert_loads(expert_loads)
        
        # Update TUI
        expert_stats = {
            'specialization': getattr(self.expert_manager, 'expert_specialization_scores', {}),
            'loads': self._calculate_expert_loads(expert_ids) if expert_ids is not None else {}
        }
        
        coherence_val = 0.5
        if hasattr(self.coherence_scheduler, 'coherence_calc'):
            coherence_val = getattr(self.coherence_scheduler.coherence_calc, 'current_coherence', 0.5)
        
        # Update TUI first
        try:
            self.training_ui.update_metrics(
                step=self.global_step,
                loss=loss_val,
                aux_loss=aux_loss_val,
                lr=self._get_current_lr(),
                expert_stats=expert_stats,
                memory_usage=memory_usage,
                gradient_norm=grad_norm_val,
                throughput=throughput,
                coherence_score=float(coherence_val),
                batch_time=step_time
            )
        except Exception as e:
            print(f"TUI update error (non-critical): {e}")
        
        # Update training monitor
        self.training_monitor.update_step(
            step=self.global_step,
            loss=loss_val,
            aux_loss=aux_loss_val,
            learning_rate=self._get_current_lr(),
            gradient_norm=grad_norm_val,
            expert_ids=expert_ids,
            language_tokens=language_tokens,
            memory_usage=memory_usage,
            batch_time=step_time
        )
        
        # Update memory manager
        self.memory_manager.step(loss_val, grad_norm_val)
        
        # Update capacity manager
        if expert_ids is not None:
            expert_loads = self._calculate_expert_loads(expert_ids)
            # Convert dict to tensor if needed
            if hasattr(self.capacity_manager, 'update_expert_loads'):
                try:
                    # Try to pass as tensor
                    loads_tensor = torch.tensor(list(expert_loads.values()), device=self.device)
                    self.capacity_manager.update_expert_loads(loads_tensor)
                except Exception:
                    # Fallback to dict
                    self.capacity_manager.update_expert_loads(expert_loads)
        
        # Update metrics history
        self.metrics_history['loss'].append(loss_val)
        self.metrics_history['aux_loss'].append(aux_loss_val)
        self.metrics_history['learning_rate'].append(self._get_current_lr())
        self.metrics_history['gradient_norm'].append(grad_norm_val)
        self.metrics_history['memory_usage'].append(memory_usage)
        self.metrics_history['throughput'].append(throughput)
        
        if expert_ids is not None:
            expert_loads = self._calculate_expert_loads(expert_ids)
            load_values = list(expert_loads.values())
            if len(load_values) > 1:
                expert_util = 1.0 - float(torch.std(torch.tensor(load_values)))
            else:
                expert_util = 0.5
            self.metrics_history['expert_utilization'].append(expert_util)
        
        # Update TUI
        expert_stats = {
            'specialization': self.expert_manager.expert_specialization_scores,
            'loads': self._calculate_expert_loads(expert_ids) if expert_ids is not None else {}
        }
        
        # Update TUI
        try:
            self.training_ui.update_metrics(
                step=self.global_step,
                loss=loss_val,
                aux_loss=aux_loss_val,
                lr=self._get_current_lr(),
                expert_stats=expert_stats,
                memory_usage=memory_usage,
                gradient_norm=grad_norm_val,
                throughput=throughput,
                coherence_score=float(coherence_val),
                batch_time=step_time
            )
        except Exception as e:
            print(f"TUI update error (non-critical): {e}")
    
    def _calculate_expert_loads(self, expert_ids: torch.Tensor) -> Dict[int, float]:
        """Calculate expert load distribution"""
        if expert_ids is None:
            return {}
        
        expert_counts = torch.bincount(expert_ids.flatten(), 
                                 minlength=getattr(self.config, 'num_experts', 8))
        total_tokens = expert_ids.numel() * getattr(self.config, 'num_experts_per_token', 2)
        
        loads = {}
        for i, count in enumerate(expert_counts):
            loads[i] = (count.item() / total_tokens) if total_tokens > 0 else 0.0
        
        return loads
    
    def _calculate_expert_utilization(self, expert_ids: torch.Tensor) -> float:
        """Calculate expert utilization metric"""
        loads = self._calculate_expert_loads(expert_ids)
        if not loads:
            return 0.5
        
        # Utilization = 1 - std(loads) (more balanced = higher utilization)
        load_values = list(loads.values())
        if len(load_values) > 1:
            std_val = torch.std(torch.tensor(load_values))
            return max(0.0, 1.0 - float(std_val))
        return 0.5
        
        # Utilization = 1 - std(loads) (more balanced = higher utilization)
        load_values = list(loads.values())
        if len(load_values) > 1:
            return max(0.0, 1.0 - torch.std(torch.tensor(load_values)).item())
        return 0.5
    
    def _estimate_throughput(self, batch: Dict[str, Any], step_time: float) -> float:
        """Estimate tokens per second"""
        if step_time <= 0:
            return 0.0
        
        input_ids = batch.get('input_ids')
        if input_ids is None:
            return 0.0
        
        batch_size = input_ids.size(0) if len(input_ids.shape) > 0 else input_ids.size(1)
        seq_length = input_ids.size(1) if len(input_ids.shape) > 1 else input_ids.size(0)
        
        total_tokens = batch_size * seq_length
        return total_tokens / step_time
    
    def _get_current_lr(self) -> float:
        """Get current learning rate"""
        if hasattr(self.optimizer, 'param_groups') and len(self.optimizer.param_groups) > 0:
            return self.optimizer.param_groups[0].get('lr', 0.0)
        elif hasattr(self.optimizer, 'lr'):
            return getattr(self.optimizer, 'lr', 0.0)
        return 0.0
    
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
                'checkpoint_manager': self.checkpoint_manager.get_checkpoint_stats(),
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


# Export the enhanced trainer
__all__ = ['EnhancedMoETrainer']