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
from .adaptive_optimizations import (
    DynamicExpertCapacityManager,
    AdaptiveAttentionSelector,
)
from .coherence_scheduler import CoherenceBasedScheduler
from .tui import MoETrainingTUI, ColoredText
from .pruning import prune_expert_widths
from .trainer_utils.rl import (
    rl_forward_pass,
    rl_stage2_forward_pass,
    compute_length_aware_dpo_loss,
)
from .trainer_utils.data import process_batch
from .trainer_utils.optimization import (
    handle_gradients_and_optimize,
    update_optimization_managers,
)
from .trainer_utils.callbacks import (
    _should_log_step,
    _should_early_stop,
    _enhanced_logging,
    _get_final_results,
    save_checkpoint,
    load_checkpoint,
)

from ..monitoring import (
    HTSRMonitor,
    HTMLDashboard,
    LogLevel,
    ObservabilityAdapter,
    collect_gpu_stats,
)

logger = logging.getLogger(__name__)


def setup_bf16_optimizer_states(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.Optimizer:
    """Convert optimizer states to bfloat16 for memory efficiency.

    BF16 optimizer states reduce memory usage while maintaining numerical stability.
    This is particularly beneficial for large models with many parameters.
    """
    try:
        import bfloat16

        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param.requires_grad and param.grad is not None:
                    state = optimizer.state.get(param)
                    if state is not None:
                        for key, value in state.items():
                            if torch.is_tensor(value) and value.dtype == torch.float32:
                                state[key] = value.to(dtype=torch.bfloat16)

        logger.info("Successfully converted optimizer states to bfloat16")
        return optimizer
    except ImportError:
        logger.warning("bfloat16 not available, keeping fp32 optimizer states")
        return optimizer


def convert_optimizer_for_bf16_training(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.Optimizer:
    """Prepare optimizer for BF16 training by adjusting param_groups.

    This function sets up the optimizer to work with BF16 training by:
    1. Creating master weights in FP32 for stability
    2. Using BF16 gradients for memory efficiency
    """
    try:
        from torch.distributed.algorithms._CheckpointWrapper import (
            checkpoint_wrapper,
        )

        for param_group in optimizer.param_groups:
            param_group["betas"] = param_group.get("betas", (0.9, 0.999))
            param_group["eps"] = param_group.get("eps", 1e-8)

        logger.info("Optimizer prepared for BF16 training")
        return optimizer
    except Exception as e:
        logger.warning(f"Failed to prepare optimizer for BF16: {e}")
        return optimizer


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
        use_enhanced_features: bool = True,
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
            if hasattr(self.model, "config"):
                self.model.config.use_ring_attention = True
            if hasattr(self.model, "_replace_with_ring_attention"):
                try:
                    self.model._replace_with_ring_attention(
                        self.model.config, self.device
                    )
                except Exception as e:
                    logger.warning(f"Failed to replace with ring attention: {e}")

        # Enhanced optimization managers
        if use_enhanced_features:
            # Expert specialization and monitoring
            self.expert_manager = ExpertSpecializationManager(
                num_experts=getattr(config, "num_experts", 8),
                num_languages=getattr(config, "num_languages", 3),
                device=device,
            )

            self.training_monitor = MoETrainingMonitor(
                num_experts=getattr(config, "num_experts", 8),
                num_languages=getattr(config, "num_languages", 3),
                log_frequency=getattr(config, "expert_monitor_log_frequency", 50),
                save_frequency=getattr(config, "expert_monitor_save_frequency", 500),
                log_dir=getattr(config, "log_dir", "./logs"),
            )

            # Checkpointing and memory management
            self.checkpoint_manager = SelectiveCheckpointManager(
                memory_threshold=getattr(config, "checkpoint_memory_threshold", 0.7),
                checkpoint_frequency=getattr(config, "checkpoint_frequency", 2),
                device=device,
            )

            self.memory_manager = AdaptiveMemoryManager(
                cleanup_frequency=getattr(config, "memory_cleanup_frequency", 50),
                memory_target=getattr(config, "memory_target", 0.8),
                enable_dynamic_batching=getattr(
                    config, "enable_dynamic_batching", True
                ),
            )

            # Dynamic optimizations
            self.capacity_manager = DynamicExpertCapacityManager(
                num_experts=getattr(config, "num_experts", 8),
                base_capacity_factor=getattr(config, "expert_capacity_factor", 1.25),
                device=device,
            )

            self.attention_selector = AdaptiveAttentionSelector(
                seq_length_threshold_mla=getattr(
                    config, "seq_length_threshold_mla", 2048
                ),
                seq_length_threshold_dsa=getattr(
                    config, "seq_length_threshold_dsa", 4096
                ),
                memory_threshold_mla=getattr(config, "memory_threshold_mla", 0.6),
                device=device,
            )

            # Coherence-based scheduler
            self.coherence_scheduler = CoherenceBasedScheduler(
                base_lr=getattr(config, "learning_rate", 1e-4),
                coherence_target=getattr(config, "coherence_target", 0.7),
                adjustment_frequency=getattr(
                    config, "coherence_adjustment_frequency", 50
                ),
                device=device,
            )

            # HTSR Monitoring for Grokking Detection (opt-in)
            htsr_config = getattr(config, "htsr", None)
            if htsr_config and getattr(htsr_config, "enable_htsr_monitoring", False):
                self.htsr_monitor = HTSRMonitor(
                    model=self.model,
                    device=device,
                    alpha_upper_threshold=getattr(
                        htsr_config, "htsr_alpha_upper_threshold", 4.5
                    ),
                    monitor_interval=getattr(htsr_config, "htsr_monitor_interval", 75),
                    variance_threshold=getattr(
                        htsr_config, "htsr_variance_threshold", 0.5
                    ),
                    verbose=True,
                )

                self.htsr_dashboard = HTMLDashboard(
                    config={
                        "port": getattr(htsr_config, "htsr_dashboard_port", 8050),
                        "host": getattr(htsr_config, "htsr_dashboard_host", "0.0.0.0"),
                    },
                    auth_users=getattr(htsr_config, "htsr_dashboard_users", {}),
                    auto_refresh_interval=getattr(
                        htsr_config, "htsr_dashboard_auto_refresh", 120
                    ),
                )

                # Configure communication channels if enabled
                if getattr(htsr_config, "htsr_comm_email_enabled", False):
                    self.htsr_dashboard.configure_communication_channel("email", {})
                if getattr(htsr_config, "htsr_comm_slack_enabled", False):
                    self.htsr_dashboard.configure_communication_channel("slack", {})
                if getattr(htsr_config, "htsr_comm_discord_enabled", False):
                    self.htsr_dashboard.configure_communication_channel("discord", {})
                if getattr(htsr_config, "htsr_comm_pagerduty_enabled", False):
                    self.htsr_dashboard.configure_communication_channel("pagerduty", {})

                # Set loss thresholds
                self.htsr_dashboard.set_loss_thresholds(
                    train_warning=getattr(htsr_config, "htsr_train_loss_warning", 1.0),
                    train_critical=getattr(
                        htsr_config, "htsr_train_loss_critical", 0.1
                    ),
                    val_warning=getattr(htsr_config, "htsr_val_loss_warning", 1.5),
                    val_critical=getattr(htsr_config, "htsr_val_loss_critical", 0.2),
                )

                self.htsr_dashboard.total_steps = getattr(config, "max_steps", 10000)
                self.htsr_dashboard.start()

                logger.info("HTSR Grokking Monitoring enabled")
            else:
                self.htsr_monitor = None
                self.htsr_dashboard = None

            # Enhanced TUI
            self.training_ui = MoETrainingTUI(
                update_frequency=getattr(config, "tui_update_frequency", 1),
                save_frequency=getattr(config, "tui_save_frequency", 100),
                log_file=getattr(
                    config, "tui_log_file", "./logs/enhanced_training.json"
                ),
                show_plots=getattr(config, "tui_show_plots", False),
            )

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.early_stop_triggered = False

        # BF16 optimizer states setup
        self.use_bf16_optimizer = getattr(config, "use_bf16_optimizer_states", False)
        if self.use_bf16_optimizer:
            try:
                self.optimizer = setup_bf16_optimizer_states(self.optimizer)
                logger.info("BF16 optimizer states enabled for memory efficiency")
            except Exception as e:
                logger.warning(f"Failed to setup BF16 optimizer states: {e}")
                self.use_bf16_optimizer = False

        # Metrics tracking
        self.metrics_history = {
            "loss": deque(maxlen=1000),
            "aux_loss": deque(maxlen=1000),
            "learning_rate": deque(maxlen=1000),
            "gradient_norm": deque(maxlen=1000),
            "gradient_noise_scale": deque(maxlen=1000),
            "expert_utilization": deque(maxlen=1000),
            "memory_usage": deque(maxlen=1000),
            "throughput": deque(maxlen=200),
            "coherence_score": deque(maxlen=1000),
        }

        # For GNS estimation
        self._grad_buffer = deque(maxlen=20)

        # Performance tracking
        self.step_times = deque(maxlen=1000)
        self.start_time = time.time()

        # Checkpoint tracking
        self.checkpoint_loaded = False
        self.save_dir = getattr(config, "output_dir", "./checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize reference model for DPO/RLHF
        self.ref_model = self._setup_ref_model()

        # Run observability (provider-agnostic, optional)
        self.observability = ObservabilityAdapter.from_config(config)

    def _setup_ref_model(self):
        """Create a frozen copy of the model as reference"""
        import copy

        ref_model = copy.deepcopy(self.model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model

    _rl_forward_pass = rl_forward_pass
    _rl_stage2_forward_pass = rl_stage2_forward_pass
    _compute_length_aware_dpo_loss = compute_length_aware_dpo_loss
    _process_batch = process_batch
    _handle_gradients_and_optimize = handle_gradients_and_optimize
    _update_optimization_managers = update_optimization_managers
    _should_log_step = _should_log_step
    _should_early_stop = _should_early_stop
    _enhanced_logging = _enhanced_logging
    _get_final_results = _get_final_results
    save_checkpoint = save_checkpoint
    load_checkpoint = load_checkpoint

    def _estimate_throughput(self, batch, step_time):
        """Estimate tokens per second"""
        if "input_ids" in batch:
            num_tokens = batch["input_ids"].numel()
        elif "chosen_input_ids" in batch:
            num_tokens = (
                batch["chosen_input_ids"].numel() + batch["rejected_input_ids"].numel()
            )
        else:
            num_tokens = 0
        return num_tokens / step_time if step_time > 0 else 0

    def _get_current_lr(self):
        """Get current learning rate"""
        if hasattr(self.optimizer, "param_groups"):
            return self.optimizer.param_groups[0]["lr"]
        return getattr(self.config, "learning_rate", 1e-4)

    def _calculate_expert_loads(self, expert_ids):
        """Calculate load per expert"""
        if expert_ids is None:
            return {}
        num_experts = getattr(self.config, "num_experts", 8)
        loads = {i: 0 for i in range(num_experts)}
        if torch.is_tensor(expert_ids):
            unique, counts = torch.unique(expert_ids, return_counts=True)
            for u, c in zip(unique, counts):
                if int(u) < num_experts:
                    loads[int(u)] = int(c)
        return loads

    def _enhanced_forward_pass(self, batch: Dict[str, Any]) -> tuple:
        """Enhanced forward pass with attention selection and RLHF"""

        # Debug logging for batch validation
        logger.debug(f"Processing batch with keys: {list(batch.keys())}")

        if "chosen_input_ids" in batch and "rejected_input_ids" in batch:
            logger.debug("DPO batch detected, computing length-aware DPO loss")
            loss = self._compute_length_aware_dpo_loss(
                self.model,
                getattr(self, "ref_model", self.model),  # Mock ref_model for now
                batch,
            )
            return loss, torch.tensor(0.0, device=self.device), None

        if "chosen" in batch and "rejected" in batch:
            input_ids = batch["chosen_input_ids"]
            labels = batch["chosen_labels"]
            batch["input_ids"] = input_ids
            batch["labels"] = labels
            logger.debug(
                f"RLHF batch: input_ids shape {input_ids.shape if input_ids is not None else 'None'}"
            )
        elif "prompt" in batch and "response" in batch:
            logger.debug(
                f"RLHF batch detected, using RL Stage {getattr(self.config, 'rl_stage', 1)} forward pass"
            )
            if getattr(self.config, "rl_stage", 1) == 2:
                return self._rl_stage2_forward_pass(batch)
            return self._rl_forward_pass(batch)

        input_ids = batch.get("input_ids")
        if input_ids is not None:
            # Validate input_ids shape
            if len(input_ids.shape) != 2:
                logger.error(
                    f"Invalid input_ids shape: {input_ids.shape}, expected 2D tensor"
                )
                raise ValueError(f"Invalid input_ids shape: {input_ids.shape}")

            seq_length = input_ids.size(1)
            batch_size = input_ids.size(0)

            logger.debug(f"Batch size: {batch_size}, Sequence length: {seq_length}")

            # Validate sequence length
            if seq_length <= 0:
                logger.error(f"Invalid sequence length: {seq_length}")
                raise ValueError(f"Invalid sequence length: {seq_length}")

            memory_usage = (
                torch.cuda.memory_allocated()
                / torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available()
                else 0
            )

            if self.use_enhanced_features:
                attention_type = self.attention_selector.select_attention_type(
                    seq_length=seq_length, memory_usage=memory_usage
                )
                logger.info(
                    f"🧠 Attention Type: {attention_type.upper()} (seq_len={seq_length}, mem={memory_usage:.2f})"
                )

        model_batch = {
            k: v
            for k, v in batch.items()
            if k not in ["labels", "pixel_values", "label_ids"]
        }

        # Debug logging for model batch
        logger.debug(f"Model batch keys: {list(model_batch.keys())}")
        for key, value in model_batch.items():
            if hasattr(value, "shape"):
                logger.debug(f"  {key}: {value.shape}")

        try:
            outputs = self.model(**model_batch)
        except Exception as e:
            logger.error(f"Model forward pass failed: {e}")
            logger.error(f"Model batch: {model_batch}")
            raise

        if isinstance(outputs, dict):
            loss = outputs.get("loss", torch.tensor(0.0, device=self.device))
            aux_loss = outputs.get("aux_loss", torch.tensor(0.0, device=self.device))
            expert_ids = outputs.get("expert_ids")
        else:
            loss = (
                outputs[0]
                if len(outputs) > 0
                else torch.tensor(0.0, device=self.device)
            )
            aux_loss = (
                outputs[1]
                if len(outputs) > 1
                else torch.tensor(0.0, device=self.device)
            )
            expert_ids = None

        if loss.item() == 0.0 and "labels" in batch:
            labels = batch["labels"].to(self.device)
            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
                )

        logger.debug(
            f"Forward pass completed: loss={loss.item():.4f}, aux_loss={aux_loss.item():.4f}"
        )
        return loss, aux_loss, expert_ids

    def _estimate_gradient_noise_scale(self) -> float:
        """
        Estimate Gradient Noise Scale (GNS) based on recent projected gradients.
        GNS = (sum(var(g_i))) / (norm(mean(g)))^2
        Uses random projection to maintain low memory footprint.
        """
        if len(self._grad_buffer) < 5:
            return 0.0

        try:
            # Stack recent projected gradients
            grads = torch.stack(list(self._grad_buffer))
            # GNS formula: E[|g - E[g]|^2] / |E[g]|^2
            mean_grad = grads.mean(dim=0)
            diff = grads - mean_grad
            var_sum = (diff**2).sum(dim=1).mean()
            mean_norm_sq = (mean_grad**2).sum()

            gns = var_sum / (mean_norm_sq + 1e-8)
            return float(gns)
        except Exception:
            return 0.0

    def _collect_grad_sample(self):
        """Collect a small projected sample of current gradients for GNS estimation"""
        try:
            # Use a subset of parameters to save memory/time
            grads = []
            for p in self.model.parameters():
                if p.requires_grad and p.grad is not None:
                    # Take a small random sample of each gradient
                    if p.grad.numel() > 100:
                        # Fixed stride for deterministic-ish sampling
                        grads.append(p.grad.flatten()[::p.grad.numel()//10].clone().detach())
                    else:
                        grads.append(p.grad.flatten().clone().detach())

            if grads:
                flat_grad = torch.cat(grads)
                self._grad_buffer.append(flat_grad.to(device='cpu', dtype=torch.float32))
        except Exception:
            pass

    def _calculate_expert_utilization(self, expert_ids):
        """Calculate expert utilization for coherence scheduler"""
        if expert_ids is None:
            return 0.5  # Default utilization

        try:
            if hasattr(expert_ids, "numel"):
                total_experts = expert_ids.numel()
                unique_experts = expert_ids.unique().numel()
                return unique_experts / max(total_experts, 1)
            return 0.5
        except:
            return 0.5

    def train(
        self,
    ) -> Dict[str, Any]:
        """Enhanced training loop with all optimizations"""

        if self.use_enhanced_features:
            print(f"\n{ColoredText.success('Enhanced MoE Training Started!')}")
            print(
                f"{ColoredText.info('Features:')} Expert Specialization + Selective Checkpointing + Dynamic Optimization + Coherence Scheduler"
            )
            print(f"{'=' * 80}")

            # Start TUI
            self.training_ui.start_training_ui(
                total_steps=getattr(self.config, "max_steps", 10)
            )

        try:
            self.model.train()
            self.observability.start_run(config=getattr(self.config, "__dict__", {}))

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

                step_time = time.time() - step_start_time
                throughput = self._estimate_throughput(batch, step_time)
                current_lr = self._get_current_lr()

                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

                metrics_payload = {
                    "train/loss": float(loss.item() if hasattr(loss, "item") else loss),
                    "train/aux_loss": float(aux_loss.item() if hasattr(aux_loss, "item") else aux_loss),
                    "train/lr": float(current_lr),
                    "train/grad_norm": float(grad_norm),
                    "train/tokens_per_sec": float(throughput),
                    "train/step_time_sec": float(step_time),
                    "train/epoch": float(self.current_epoch),
                }
                metrics_payload.update(collect_gpu_stats())
                self.observability.log_metrics(metrics_payload, step=self.global_step)

                # Pruning
                if (
                    self.config.pruning_steps
                    and self.global_step in self.config.pruning_steps
                ):
                    prune_expert_widths(
                        self.model, self.config.pruning_ratio, ["expert"]
                    )

                # Enhanced logging and early stopping
                if self._should_log_step():
                    self._enhanced_logging(batch_idx)

                if self._should_early_stop():
                    break

                # Coherence-based early stopping
                if self.use_enhanced_features:
                    coherence_result = self.coherence_scheduler.step(
                        loss=loss.item() if hasattr(loss, "item") else float(loss),
                        aux_loss=aux_loss.item()
                        if hasattr(aux_loss, "item")
                        else float(aux_loss),
                        expert_utilization=self._calculate_expert_utilization(
                            expert_ids
                        ),
                        gradient_norm=grad_norm,
                        step=self.global_step,
                    )

                    if coherence_result["should_stop"]:
                        self.early_stop_triggered = True
                        print(
                            f"{ColoredText.warning('Early stopping triggered by coherence scheduler!')}"
                        )
                        break

                    if coherence_result["adjusted"]:
                        # Update learning rate based on coherence
                        if hasattr(self.optimizer, "param_groups"):
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = coherence_result["current_lr"]

                # HTSR Monitoring (check every N steps)
                if (
                    self.htsr_monitor
                    and self.global_step % self.htsr_monitor.monitor_interval == 0
                ):
                    self._htsr_monitor_step(
                        loss.item() if hasattr(loss, "item") else float(loss)
                    )

        except KeyboardInterrupt:
            self.observability.finish_run(status="interrupted")
            print(f"\n{ColoredText.warning('Training interrupted by user!')}")
        except Exception as e:
            self.observability.finish_run(status="failed")
            print(f"{ColoredText.error(f'Training failed: {e}')}")
            import traceback

            traceback.print_exc()

        finally:
            self.observability.finish_run(status="completed")
            if self.use_enhanced_features:
                self.training_ui.stop_training_ui()

            # Stop HTSR dashboard
            if self.htsr_dashboard:
                self.htsr_dashboard.stop()

            # Save final results
            return self._get_final_results()

    def _htsr_monitor_step(self, train_loss: float = None):
        """Perform HTSR monitoring step.

        Args:
            train_loss: Current training loss for dashboard
        """
        if not self.htsr_monitor:
            return

        try:
            # Compute α for all layers
            result = self.htsr_monitor.compute_all_layer_alphas()

            alpha_metrics = result["alpha_metrics"]
            detector_state = result["detector_state"]

            # Get current LR
            current_lr = self._get_current_lr()
            lr_values = [pg.get("lr", current_lr) for pg in self.optimizer.param_groups]

            # Update dashboard
            if self.htsr_dashboard:
                self.htsr_dashboard.update_alpha_metrics(
                    layer_alphas=alpha_metrics,
                    model_alpha=detector_state.get("model_alpha", 2.0),
                    alpha_variance=detector_state.get("alpha_variance", 0.0),
                    over_grokking_layers=detector_state.get("over_grokking_layers", {}),
                    high_variance_layers=detector_state.get("high_variance_layers", {}),
                    step=self.global_step,
                )

                self.htsr_dashboard.update_lr(lr_values, self.global_step)

                if train_loss is not None:
                    self.htsr_dashboard.update_losses(
                        train_loss=train_loss, val_loss=None, step=self.global_step
                    )
                    self.observability.log_metrics({"eval/train_loss": float(train_loss)}, step=self.global_step)

            # Apply intervention if grokking detected
            htsr_config = getattr(self.config, "htsr", None)
            if htsr_config and detector_state.get("detected", False):
                intervention_applied = self._apply_htsr_intervention(
                    detector_state=detector_state,
                    lr_reduction_factor=getattr(
                        htsr_config, "htsr_lr_reduction_factor", 0.5
                    ),
                    wd_increase_factor=getattr(
                        htsr_config, "htsr_wd_increase_factor", 2.0
                    ),
                    auto_apply=getattr(htsr_config, "htsr_apply_intervention", True),
                )

                if intervention_applied and self.htsr_dashboard:
                    self.htsr_dashboard.log_intervention(
                        intervention_type="lr_reduction_wd_increase",
                        details={
                            "reason": "grokking_detected",
                            "model_alpha": detector_state.get("model_alpha"),
                            "alpha_variance": detector_state.get("alpha_variance"),
                            "lr_reduction_factor": getattr(
                                htsr_config, "htsr_lr_reduction_factor", 0.5
                            ),
                            "wd_increase_factor": getattr(
                                htsr_config, "htsr_wd_increase_factor", 2.0
                            ),
                        },
                        step=self.global_step,
                    )

            # Update additional system and MoE metrics requested by user
            if hasattr(self.model, "calculate_weight_entropy"):
                weight_entropy = self.model.calculate_weight_entropy()
                # Update model cache for entropic steering
                self.model._cached_weight_entropy = weight_entropy
            else:
                weight_entropy = self._calculate_weight_entropy()

            power_draw = self._estimate_power_draw()

            if self.htsr_dashboard:
                self.htsr_dashboard.update_system_metrics(
                    weight_entropy=weight_entropy,
                    power_draw=power_draw,
                    step=self.global_step
                )

                # MoE metrics (utilization and GNS)
                recent_util = self.metrics_history["expert_utilization"][-1] if self.metrics_history["expert_utilization"] else 0.5
                recent_gns = self.metrics_history["gradient_noise_scale"][-1] if self.metrics_history["gradient_noise_scale"] else 0.0

                self.htsr_dashboard.update_moe_metrics(
                    utilization=recent_util,
                    gns=recent_gns,
                    step=self.global_step
                )

            logger.debug(
                f"HTSR Check: α={detector_state.get('model_alpha', 'N/A'):.2f}, "
                f"variance={detector_state.get('alpha_variance', 0):.4f}, "
                f"over_grokking={len(detector_state.get('over_grokking_layers', {}))}, "
                f"entropy={weight_entropy:.4f}, power={power_draw:.1f}W"
            )

        except Exception as e:
            logger.warning(f"HTSR monitoring step failed: {e}")

    def _calculate_weight_entropy(self) -> float:
        """Calculate average entropy of weight distributions across linear layers."""
        total_entropy = 0.0
        count = 0
        try:
            for name, param in self.model.named_parameters():
                if "weight" in name and param.dim() >= 2 and param.numel() > 100:
                    w = param.detach().float()
                    # Standardize to get a distribution
                    w_min, w_max = w.min(), w.max()
                    hist = torch.histc(w, bins=50, min=float(w_min), max=float(w_max))
                    prob = hist / (hist.sum() + 1e-10)
                    entropy = -(prob * torch.log(prob + 1e-10)).sum()
                    total_entropy += entropy.item()
                    count += 1
        except Exception:
            return 0.0
        return total_entropy / count if count > 0 else 0.0

    def _estimate_power_draw(self) -> float:
        """Estimate current power draw in Watts (stub/approximation)."""
        if torch.cuda.is_available():
            # Rough approximation: base + dynamic part proportional to utilization if we could get it
            # Since we can't easily get GPU util without extra libs, we use a proxy
            return 150.0 + 150.0 * (0.7)  # Assume 70% load during training
        return 65.0  # Typical CPU load

    def _apply_htsr_intervention(
        self,
        detector_state: Dict[str, Any],
        lr_reduction_factor: float = 0.5,
        wd_increase_factor: float = 2.0,
        auto_apply: bool = True,
    ) -> bool:
        """Apply intervention to reduce grokking.

        Args:
            detector_state: Current detector state from HTSR monitor
            lr_reduction_factor: Factor to reduce learning rate
            wd_increase_factor: Factor to increase weight decay
            auto_apply: Whether to automatically apply interventions

        Returns:
            True if intervention was applied
        """
        if not auto_apply:
            return False

        try:
            # Apply intervention via monitor
            if self.htsr_monitor:
                intervention_details = self.htsr_monitor.apply_intervention(
                    intervention_type="lr_reduction_wd_increase",
                    lr_reduction_factor=lr_reduction_factor,
                    wd_increase_factor=wd_increase_factor,
                    optimizer=self.optimizer,
                )

                logger.warning(
                    f"HTSR Intervention: Reduced LR by {lr_reduction_factor}x, "
                    f"increased WD by {wd_increase_factor}x at step {self.global_step}"
                )

                return True

        except Exception as e:
            logger.warning(f"HTSR intervention failed: {e}")

        return False
