"""Enhanced Trainer with All MoE Optimizations Integrated"""

import torch
import torch.nn as nn
import time
import os
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import deque
import json

from .expert_manager import ExpertSpecializationManager, MoETrainingMonitor
from .checkpointing import SelectiveCheckpointManager, AdaptiveMemoryManager
from .adaptive_optimizations import DynamicExpertCapacityManager, AdaptiveAttentionSelector
from .coherence_scheduler import CoherenceBasedScheduler
from .tui import MoETrainingTUI, ColoredText
from .pruning import prune_expert_widths
from .trainer_utils.rl import rl_forward_pass
from .trainer_utils.data import process_batch
from .trainer_utils.optimization import handle_gradients_and_optimize, update_optimization_managers
from .trainer_utils.callbacks import _should_log_step, _should_early_stop, _enhanced_logging, _get_final_results, save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


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
        tokenizer=None,
        use_enhanced_features: bool = True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.use_enhanced_features = use_enhanced_features
        
        if self.config.use_ring_attention:
            # Ensure model has config attribute
            if hasattr(self.model, 'config'):
                self.model.config.use_ring_attention = True
            if hasattr(self.model, '_replace_with_ring_attention'):
                try:
                    self.model._replace_with_ring_attention(self.model.config, self.device)
                except Exception as e:
                    logger.warning(f"Failed to replace with ring attention: {e}")

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
    
    _rl_forward_pass = rl_forward_pass
    _process_batch = process_batch
    _handle_gradients_and_optimize = handle_gradients_and_optimize
    _update_optimization_managers = update_optimization_managers
    _should_log_step = _should_log_step
    _should_early_stop = _should_early_stop
    _enhanced_logging = _enhanced_logging
    _get_final_results = _get_final_results
    save_checkpoint = save_checkpoint
    load_checkpoint = load_checkpoint

    def _enhanced_forward_pass(self, batch: Dict[str, Any]) -> tuple:
        """Enhanced forward pass with attention selection and RLHF"""
        
        # Debug logging for batch validation
        logger.debug(f"Processing batch with keys: {list(batch.keys())}")
        
        if 'chosen' in batch and 'rejected' in batch:
             input_ids = batch['chosen_input_ids']
             labels = batch['chosen_labels']
             batch['input_ids'] = input_ids
             batch['labels'] = labels
             logger.debug(f"RLHF batch: input_ids shape {input_ids.shape if input_ids is not None else 'None'}")
        elif 'prompt' in batch and 'response' in batch:
            logger.debug("RLHF batch detected, using RL forward pass")
            return self._rl_forward_pass(batch)

        input_ids = batch.get('input_ids')
        if input_ids is not None:
            # Validate input_ids shape
            if len(input_ids.shape) != 2:
                logger.error(f"Invalid input_ids shape: {input_ids.shape}, expected 2D tensor")
                raise ValueError(f"Invalid input_ids shape: {input_ids.shape}")
            
            seq_length = input_ids.size(1)
            batch_size = input_ids.size(0)
            
            logger.debug(f"Batch size: {batch_size}, Sequence length: {seq_length}")
            
            # Validate sequence length
            if seq_length <= 0:
                logger.error(f"Invalid sequence length: {seq_length}")
                raise ValueError(f"Invalid sequence length: {seq_length}")
            
            memory_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0

            if self.use_enhanced_features:
                attention_type = self.attention_selector.select_attention_type(
                    seq_length=seq_length,
                    memory_usage=memory_usage
                )
                logger.info(f"🧠 Attention Type: {attention_type.upper()} (seq_len={seq_length}, mem={memory_usage:.2f})")

        model_batch = {k: v for k, v in batch.items() if k not in ['labels', 'pixel_values', 'label_ids']}

        # Debug logging for model batch
        logger.debug(f"Model batch keys: {list(model_batch.keys())}")
        for key, value in model_batch.items():
            if hasattr(value, 'shape'):
                logger.debug(f"  {key}: {value.shape}")

        try:
            outputs = self.model(**model_batch)
        except Exception as e:
            logger.error(f"Model forward pass failed: {e}")
            logger.error(f"Model batch: {model_batch}")
            raise

        if isinstance(outputs, dict):
            loss = outputs.get('loss', torch.tensor(0.0, device=self.device))
            aux_loss = outputs.get('aux_loss', torch.tensor(0.0, device=self.device))
            expert_ids = outputs.get('expert_ids')
        else:
            loss = outputs[0] if len(outputs) > 0 else torch.tensor(0.0, device=self.device)
            aux_loss = outputs[1] if len(outputs) > 1 else torch.tensor(0.0, device=self.device)
            expert_ids = None

        if loss.item() == 0.0 and 'labels' in batch:
            labels = batch['labels'].to(self.device)
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )

        logger.debug(f"Forward pass completed: loss={loss.item():.4f}, aux_loss={aux_loss.item():.4f}")
        return loss, aux_loss, expert_ids

    def _calculate_expert_utilization(self, expert_ids):
        """Calculate expert utilization for coherence scheduler"""
        if expert_ids is None:
            return 0.5  # Default utilization
        
        try:
            if hasattr(expert_ids, 'numel'):
                total_experts = expert_ids.numel()
                unique_experts = expert_ids.unique().numel()
                return unique_experts / max(total_experts, 1)
            return 0.5
        except:
            return 0.5

    def train(self, ) -> Dict[str, Any]:
        """Enhanced training loop with all optimizations"""
        
        if self.use_enhanced_features:
            print(f"\n{ColoredText.success('Enhanced MoE Training Started!')}")
            print(f"{ColoredText.info('Features:')} Expert Specialization + Selective Checkpointing + Dynamic Optimization + Coherence Scheduler")
            print(f"{'='*80}")
            
            # Start TUI
            self.training_ui.start_training_ui(
                total_steps=getattr(self.config, 'max_steps', 10)
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

                # Pruning
                if self.config.pruning_steps and self.global_step in self.config.pruning_steps:
                    prune_expert_widths(self.model, self.config.pruning_ratio, ["expert"])

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
