import torch
import time
from typing import Dict, Any


def handle_gradients_and_optimize(self) -> float:
    """Enhanced gradient handling with clipping and bf16 support"""

    if getattr(self, "use_deepspeed", False):
        self.model.step()
        return 0.0

    grad_norm = 0.0

    clip_norm = getattr(self.config, "gradient_clip_norm", None)
    if clip_norm is None:
        for module in self.model.modules():
            if hasattr(module, "gradient_clip_norm"):
                clip_norm = module.gradient_clip_norm
                break

    if clip_norm is not None and clip_norm > 0:
        if hasattr(self.optimizer, "param_groups"):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizer.param_groups for p in group["params"]],
                clip_norm,
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()), clip_norm
            )

    # Collect gradient sample for GNS estimation before optimizer step
    if hasattr(self, "_collect_grad_sample"):
        self._collect_grad_sample()

    if hasattr(self.optimizer, "step"):
        if hasattr(self.optimizer, "param_groups"):
            self.optimizer.step()
        else:
            self.optimizer.step()

    self.optimizer.zero_grad()

    return grad_norm.item() if hasattr(grad_norm, "item") else grad_norm


def update_optimization_managers(
    self, loss, aux_loss, grad_norm, expert_ids, batch, step_start_time
):
    """Update all optimization managers with current step data"""
    # Convert tensors to python types for compatibility
    loss_val = loss.item() if hasattr(loss, "item") else float(loss)
    aux_loss_val = aux_loss.item() if hasattr(aux_loss, "item") else float(aux_loss)
    grad_norm_val = (
        float(grad_norm)
        if isinstance(grad_norm, (int, float))
        else (grad_norm.item() if hasattr(grad_norm, "item") else 0.0)
    )

    step_time = time.time() - step_start_time
    throughput = self._estimate_throughput(batch, step_time)

    # Update basic metrics history regardless of enhanced features
    self.metrics_history["loss"].append(loss_val)
    self.metrics_history["aux_loss"].append(aux_loss_val)
    self.metrics_history["learning_rate"].append(self._get_current_lr())
    self.metrics_history["gradient_norm"].append(grad_norm_val)
    self.metrics_history["throughput"].append(throughput)

    # Update GNS metric
    if hasattr(self, "_estimate_gradient_noise_scale"):
        self.metrics_history["gradient_noise_scale"].append(self._estimate_gradient_noise_scale())

    if not self.use_enhanced_features:
        return

    memory_usage = (
        torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    )
    language_tokens = batch.get("language", [])

    # Update expert manager
    if expert_ids is not None:
        self.expert_manager.update_expert_stats(
            expert_ids=expert_ids, language_tokens=language_tokens
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
        batch_time=step_time,
    )

    # Update memory manager
    self.memory_manager.step(loss_val, grad_norm_val)

    # Update capacity manager
    if expert_ids is not None:
        expert_loads = self._calculate_expert_loads(expert_ids)
        # Convert dict to tensor if needed
        if hasattr(self.capacity_manager, "update_expert_loads"):
            try:
                # Try to pass as tensor
                loads_tensor = torch.tensor(
                    list(expert_loads.values()), device=self.device
                )
                self.capacity_manager.update_expert_loads(loads_tensor)
            except Exception:
                # Fallback to dict
                self.capacity_manager.update_expert_loads(expert_loads)

    # Update TUI
    expert_stats = {
        "specialization": getattr(
            self.expert_manager, "expert_specialization_scores", {}
        ),
        "loads": self._calculate_expert_loads(expert_ids)
        if expert_ids is not None
        else {},
    }

    coherence_val = 0.5
    if hasattr(self.coherence_scheduler, "coherence_calc"):
        coherence_val = getattr(
            self.coherence_scheduler.coherence_calc, "current_coherence", 0.5
        )

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
            batch_time=step_time,
        )
    except Exception as e:
        print(f"TUI update error (non-critical): {e}")

    # Update training monitor - restored once
    self.training_monitor.update_step(
        step=self.global_step,
        loss=loss_val,
        aux_loss=aux_loss_val,
        learning_rate=self._get_current_lr(),
        gradient_norm=grad_norm_val,
        expert_ids=expert_ids,
        language_tokens=language_tokens,
        memory_usage=memory_usage,
        batch_time=step_time,
    )

    # Update memory manager - restored once
    self.memory_manager.step(loss_val, grad_norm_val)

    # Update capacity manager
    if expert_ids is not None:
        expert_loads = self._calculate_expert_loads(expert_ids)
        if hasattr(self.capacity_manager, "update_expert_loads"):
            try:
                loads_tensor = torch.tensor(
                    list(expert_loads.values()), device=self.device
                )
                self.capacity_manager.update_expert_loads(loads_tensor)
            except Exception:
                self.capacity_manager.update_expert_loads(expert_loads)

    # Memory usage for enhanced features
    self.metrics_history["memory_usage"].append(memory_usage)

    if expert_ids is not None:
        expert_loads_dict = self._calculate_expert_loads(expert_ids)
        load_values = list(expert_loads_dict.values())
        if len(load_values) > 1:
            expert_util = 1.0 - float(torch.std(torch.tensor(load_values, dtype=torch.float32)))
        else:
            expert_util = 0.5
        self.metrics_history["expert_utilization"].append(expert_util)

    # Update TUI - restored once
    expert_stats = {
        "specialization": getattr(self.expert_manager, "expert_specialization_scores", {}),
        "loads": expert_loads_dict if expert_ids is not None else {},
    }

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
            batch_time=step_time,
        )
    except Exception as e:
        print(f"TUI update error (non-critical): {e}")
