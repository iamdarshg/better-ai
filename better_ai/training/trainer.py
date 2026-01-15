"""Training utilities for DeepSeek model"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Callable
import time
import json
import os
from datetime import datetime
import logging

from ..config import TrainingConfig
from ..optimizers.fp8 import FP8AdamW, get_fp8_optimizer
from ..models.core import DeepSeekModel


class Trainer:
    """Main training loop for DeepSeek model"""
    
    def __init__(
        self,
        model: DeepSeekModel,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = get_fp8_optimizer(
                self.model,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                use_fp8=config.use_fp8,
                fp8_scaling_window=config.fp8_scaling_window,
                fp8_delayed_scaling=config.fp8_delayed_scaling
            )
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        if scheduler is None:
            self.scheduler = self._create_scheduler()
        else:
            self.scheduler = scheduler
        
        # Setup AMP scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.bf16 or config.fp16 else None
        
        # Setup logging
        self._setup_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        total_steps = len(self.train_dataloader) * self.config.max_steps
        
        if self.config.lr_schedule == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer.optimizer if hasattr(self.optimizer, 'optimizer') else self.optimizer,
                T_max=total_steps,
                eta_min=self.config.min_lr_ratio * self.config.learning_rate
            )
        elif self.config.lr_schedule == "linear":
            return optim.lr_scheduler.LinearLR(
                self.optimizer.optimizer if hasattr(self.optimizer, 'optimizer') else self.optimizer,
                start_factor=1.0,
                end_factor=self.config.min_lr_ratio,
                total_iters=total_steps
            )
        else:  # constant
            return optim.lr_scheduler.ConstantLR(
                self.optimizer.optimizer if hasattr(self.optimizer, 'optimizer') else self.optimizer,
                factor=1.0
            )
    
    def train(self):
        """Main training loop"""
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Total steps: {self.config.max_steps}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        
        start_time = time.time()
        
        try:
            while self.global_step < self.config.max_steps:
                self.model.train()
                
                for batch in self.train_dataloader:
                    # Move batch to device (handle mixed data types)
                    batch = {
                        k: v.to(self.device) if hasattr(v, 'to') and callable(getattr(v, 'to')) else v
                        for k, v in batch.items()
                    }
                    
                    # Forward pass
                    loss = self.training_step(batch)
                    
                    # Log metrics
                    if self.global_step % self.config.log_every_n_steps == 0:
                        self._log_metrics(loss, start_time)
                    
                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0 and self.eval_dataloader:
                        self._evaluate()
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint()
                    
                    # Check for early stopping
                    if self.global_step >= self.config.max_steps:
                        break
                
                self.epoch += 1
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        finally:
            self._save_checkpoint(final=True)
            self.logger.info("Training completed")
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform one training step"""
        input_ids = batch['input_ids']
        labels = batch['labels'] if 'labels' in batch else input_ids
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.config.bf16 or self.config.fp16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=batch.get('attention_mask')
            )
            logits = outputs['last_hidden_state']
            
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # Backward pass
        # Debug: Check label range before loss
        if 'labels' in batch and hasattr(batch['labels'], 'min'):
            print(f"DEBUG: Labels range: {batch['labels'].min().item()} - {batch['labels'].max().item()}")
        if self.config.bf16 or self.config.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.clip_grad_norm > 0:
                if self.config.bf16 or self.config.fp16:
                    self.scaler.unscale_(self.optimizer.optimizer if hasattr(self.optimizer, 'optimizer') else self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            
            # Optimizer step
            if self.config.bf16 or self.config.fp16:
                self.scaler.step(self.optimizer.optimizer if hasattr(self.optimizer, 'optimizer') else self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Scheduler step
            self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
        
        self.global_step += 1
        return loss.detach()
    
    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        
        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=self.config.bf16 or self.config.fp16):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask')
                )
                logits = outputs['last_hidden_state']
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch['labels'][..., 1:].contiguous() if 'labels' in batch else batch['input_ids'][..., 1:].contiguous()
                
# Use ignore_index=0 since synthetic data uses 1-9999 range (no 0 padding)
            loss = nn.CrossEntropyLoss(ignore_index=0)(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            
            total_loss += loss.item() * batch['input_ids'].size(0)
            total_samples += batch['input_ids'].size(0)
        
        avg_loss = total_loss / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Update best loss
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self._save_checkpoint(best=True)
        
        self.model.train()
        
        metrics = {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
            'best_loss': self.best_loss
        }
        
        self.logger.info(f"Validation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        return metrics
    
    def _log_metrics(self, loss: torch.Tensor, start_time: float):
        """Log training metrics"""
        current_lr = self.scheduler.get_last_lr()[0]
        elapsed_time = time.time() - start_time
        samples_per_sec = (self.global_step * self.config.batch_size) / elapsed_time
        
        metrics = {
            'step': self.global_step,
            'loss': loss.item(),
            'learning_rate': current_lr,
            'samples_per_sec': samples_per_sec,
            'time_elapsed': elapsed_time
        }
        
        self.logger.info(f"Step {self.global_step}: Loss = {loss.item():.4f}, LR = {current_lr:.2e}, "
                        f"Throughput = {samples_per_sec:.2f} samples/sec")
        
        # Log to file
        with open(os.path.join(self.config.log_dir, 'metrics.jsonl'), 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def _save_checkpoint(self, best: bool = False, final: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.optimizer.state_dict() if hasattr(self.optimizer, 'optimizer') else self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': self.config.__dict__
        }
        
        if best:
            checkpoint_path = os.path.join(self.config.output_dir, 'best_model.pt')
        elif final:
            checkpoint_path = os.path.join(self.config.output_dir, 'final_model.pt')
        else:
            checkpoint_path = os.path.join(self.config.output_dir, f'checkpoint_step_{self.global_step}.pt')
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space"""
        checkpoint_files = [f for f in os.listdir(self.config.output_dir) if f.startswith('checkpoint_step_')]
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # Keep only the latest N checkpoints
        max_checkpoints = self.config.save_total_limit
        if len(checkpoint_files) > max_checkpoints:
            for old_checkpoint in checkpoint_files[:-max_checkpoints]:
                os.remove(os.path.join(self.config.output_dir, old_checkpoint))
                self.logger.info(f"Removed old checkpoint: {old_checkpoint}")


def create_data_collator(config: TrainingConfig) -> Callable:
    """Create data collator function"""
    def collator(batch):
        """Collate function for training data"""
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item.get('attention_mask', torch.ones_like(item['input_ids'])) for item in batch]
        
        # Pad sequences to max length in batch
        max_length = min(max(len(seq) for seq in input_ids), config.max_seq_length)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids, mask in zip(input_ids, attention_mask):
            if len(ids) > max_length:
                # Truncate
                padded_ids = ids[:max_length]
                padded_mask = mask[:max_length]
            else:
                # Pad
                pad_length = max_length - len(ids)
                padded_ids = torch.cat([ids, torch.zeros(pad_length, dtype=ids.dtype)])
                padded_mask = torch.cat([mask, torch.zeros(pad_length, dtype=mask.dtype)])
            
            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_mask)
        }
    
    return collator


def get_model_size(model: nn.Module) -> Dict[str, int]:
    """Get model size statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate memory usage
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': total_size / 1024**2,
        'model_size_gb': total_size / 1024**3
    }