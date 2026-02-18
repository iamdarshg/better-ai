"""Optimized Mixture of Experts (MoE) implementation with token-centric processing"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict
import torch.distributed as dist
from .core import RMSNorm
from .moe import LossFreeBalancing, router_z_loss


class Expert(nn.Module):
    """
    Optimized Single expert layer with Fused SwiGLU and FP8 support.

    Memory Optimizations:
    1. Fused Gate/Up Projections: Reduces kernel launches and overhead.
    2. FP8 Support: Directly supports FP8 linear layers if requested.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_fp8: bool = False
    ):
        super().__init__()

        # Fused gate and up projections to save memory overhead and improve speed
        if use_fp8:
            from ..optimizers.fp8 import FP8Linear
            self.gate_up_proj = FP8Linear(hidden_size, 2 * intermediate_size, bias=bias)
            self.down_proj = FP8Linear(intermediate_size, hidden_size, bias=bias)
        else:
            self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

        self.intermediate_size = intermediate_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused forward: gate and up are computed in one GEMM
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.dropout(self.down_proj(F.silu(gate) * up))


class OptimizedExpertRouter(nn.Module):
    """Optimized router with load-aware routing and Expert Choice support"""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        router_bias: bool = False,
        router_dtype: torch.dtype = torch.float32,
        capacity_factor: float = 1.25,
        routing_type: str = "topk" # "topk" or "expert_choice"
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.capacity_factor = capacity_factor
        self.routing_type = routing_type

        # Router projection
        self.router_linear = nn.Linear(
            hidden_size,
            num_experts,
            bias=router_bias,
            dtype=router_dtype
        )

        # Load balancing stats (updated lazily)
        self.register_buffer('expert_loads_ema', torch.zeros(num_experts))
        self.load_update_freq = 100
        self.update_counter = 0

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute only router logits"""
        return self.router_linear(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        num_tokens = batch_size * sequence_length

        # Compute router logits
        router_logits = self.get_logits(hidden_states) # [B, S, E]

        if self.routing_type == "expert_choice":
            # Expert Choice routing logic (ST-MoE)
            # Each expert selects top tokens based on capacity
            capacity = int((num_tokens * self.num_experts_per_token) / self.num_experts * self.capacity_factor)

            # router_logits_flat: [num_tokens, num_experts]
            router_logits_flat = router_logits.view(-1, self.num_experts)
            probs = F.softmax(router_logits_flat, dim=0) # Normalize across tokens for each expert?
            # Actually, standard Expert Choice uses scores directly or normalized across experts

            # Top-C tokens for each expert
            expert_scores, token_indices = torch.topk(router_logits_flat.t(), k=capacity, dim=-1) # [E, C]

            # We need to return in a format compatible with the MoE layer
            # For Expert Choice, we'll return a sparse representation or mapped weights
            # This is a stub for full implementation
            return expert_scores, token_indices, router_logits

        # Default Top-k selection
        routing_probs = F.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(
            routing_probs,
            self.num_experts_per_token,
            dim=-1
        )
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-6)

        return routing_weights, selected_experts, router_logits

    def update_load_stats(self, expert_counts: torch.Tensor):
        """Update EMA of expert loads"""
        self.update_counter += 1
        if self.update_counter % self.load_update_freq == 0:
            alpha = 0.1  # EMA decay
            self.expert_loads_ema = (
                alpha * expert_counts +
                (1 - alpha) * self.expert_loads_ema
            )


class OptimizedMoELayer(nn.Module):
    """Optimized MoE layer with token-centric processing and Switch-style load balancing"""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        expert_intermediate_size: Optional[int] = None,
        dropout: float = 0.0,
        capacity_factor: float = 1.25,
        load_balance_loss_weight: float = 0.01,
        router_bias: bool = False,
        router_dtype: torch.dtype = torch.float32,
        shared_experts: int = 1,
        loss_free_balancing: bool = True,
        routing_type: str = "topk"
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.capacity_factor = capacity_factor
        self.load_balance_loss_weight = load_balance_loss_weight
        self.shared_experts = shared_experts
        self.loss_free_balancing = loss_free_balancing

        if expert_intermediate_size is None:
            expert_intermediate_size = hidden_size * 4

        # Optimized router - only routes to non-shared experts
        self.router = OptimizedExpertRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            router_bias=router_bias,
            router_dtype=router_dtype,
            capacity_factor=capacity_factor,
            routing_type=routing_type
        )

        if self.loss_free_balancing:
            self.balancer = LossFreeBalancing(
                num_experts=num_experts,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        else:
            self.balancer = None

        # Experts
        use_fp8_experts = router_dtype == torch.float8_e4m3fn or router_dtype == torch.float8_e5m2

        self.experts = nn.ModuleList([
            Expert(
                hidden_size=hidden_size,
                intermediate_size=expert_intermediate_size,
                dropout=dropout,
                use_fp8=use_fp8_experts
            ) for _ in range(num_experts)
        ])

        # Shared experts (always active)
        if shared_experts > 0:
            self.shared_experts_layer = nn.ModuleList([
                Expert(
                    hidden_size=hidden_size,
                    intermediate_size=expert_intermediate_size,
                    dropout=dropout,
                    use_fp8=use_fp8_experts
                ) for _ in range(shared_experts)
            ])

    def _token_centric_expert_forward(
        self,
        hidden_states_flat: torch.Tensor,
        routing_weights_flat: torch.Tensor,
        selected_experts_flat: torch.Tensor,
        total_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized token-centric expert processing using vectorized indexing and index_add
        """
        device = hidden_states_flat.device
        expert_outputs = torch.zeros_like(hidden_states_flat)
        expert_loads = torch.zeros(self.num_experts, device=device)

        for expert_idx in range(self.num_experts):
            mask = (selected_experts_flat == expert_idx)
            if not mask.any():
                continue

            token_indices, k_indices = torch.where(mask)
            expert_input = hidden_states_flat[token_indices]
            weights = routing_weights_flat[token_indices, k_indices].unsqueeze(-1)

            # Use checkpointing for memory-critical training
            if self.training and getattr(self, "expert_checkpointing", True):
                expert_output = torch.utils.checkpoint.checkpoint(
                    self.experts[expert_idx],
                    expert_input,
                    use_reentrant=False
                )
            else:
                expert_output = self.experts[expert_idx](expert_input)

            expert_outputs.index_add_(0, token_indices, expert_output * weights)
            expert_loads[expert_idx] = token_indices.size(0)

        return expert_outputs, expert_loads

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # Route tokens - use balancer if available for noise reduction
        if self.loss_free_balancing and self.balancer is not None:
            # Use get_logits if available to ensure pre_router_net is handled
            if hasattr(self.router, 'get_logits'):
                router_logits = self.router.get_logits(hidden_states)
            else:
                router_logits = self.router.router_linear(hidden_states)

            routing_weights, selected_experts = self.balancer.update_and_route(
                router_logits, compute_loads=True, k=self.num_experts_per_token
            )
            routing_type = "topk"
        else:
            routing_weights, selected_experts, router_logits = self.router(hidden_states)
            routing_type = getattr(self.router, "routing_type", "topk")

        # Flatten for processing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        total_tokens = hidden_states_flat.size(0)

        if routing_type == "expert_choice":
            # routing_weights is [E, C], selected_experts is [E, C] (token indices)
            expert_outputs = torch.zeros_like(hidden_states_flat)
            expert_loads = torch.zeros(self.num_experts, device=hidden_states.device)

            for expert_idx in range(self.num_experts):
                token_indices = selected_experts[expert_idx]
                weights = routing_weights[expert_idx].unsqueeze(-1)

                expert_input = hidden_states_flat[token_indices]
                expert_output = self.experts[expert_idx](expert_input)

                expert_outputs.index_add_(0, token_indices, expert_output * weights)
                expert_loads[expert_idx] = token_indices.size(0)

            # For aux loss compatibility
            selected_experts_flat = None
        else:
            # Default Top-k path
            routing_weights_flat = routing_weights.view(-1, self.num_experts_per_token)
            selected_experts_flat = selected_experts.view(-1, self.num_experts_per_token)

            # Expert processing
            expert_outputs, expert_loads = self._token_centric_expert_forward(
                hidden_states_flat, routing_weights_flat, selected_experts_flat, total_tokens
            )

        # Shared experts
        if self.shared_experts > 0:
            device = hidden_states_flat.device
            shared_output = torch.zeros_like(hidden_states_flat)
            shared_expert_loads = torch.zeros(self.shared_experts, device=device)
            for shared_idx in range(self.shared_experts):
                shared_expert_output = self.shared_experts_layer[shared_idx](hidden_states_flat)
                shared_output += shared_expert_output / self.shared_experts
                shared_expert_loads[shared_idx] = total_tokens

            expert_outputs += shared_output
            expert_loads = torch.cat([expert_loads, shared_expert_loads])

        final_outputs = expert_outputs.view(batch_size, sequence_length, hidden_dim)
        self.router.update_load_stats(expert_loads)

        # Robust load balancing loss
        if self.loss_free_balancing and self.balancer is not None:
            # Noise-free balancing
            z_loss = router_z_loss(router_logits, 1e-3)
            aux_losses = {
                'load_balance_loss': torch.tensor(0.0, device=final_outputs.device),
                'router_z_loss': z_loss,
                'total_aux_loss': z_loss
            }
        else:
            # Switch-Transformer style auxiliary loss
            if selected_experts_flat is None:
                # Expert Choice aux loss (simplified or skipped)
                router_z_loss_val = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
                aux_losses = {
                    'load_balance_loss': torch.tensor(0.0, device=final_outputs.device),
                    'router_z_loss': router_z_loss_val,
                    'total_aux_loss': 0.001 * router_z_loss_val
                }
            else:
                aux_losses = self._compute_aux_losses(router_logits, selected_experts_flat, total_tokens)

        return final_outputs, aux_losses['total_aux_loss'], aux_losses

    def _compute_aux_losses(
        self,
        router_logits: torch.Tensor,
        selected_experts_flat: torch.Tensor,
        total_tokens: int
    ) -> Dict[str, torch.Tensor]:
        """Switch-Transformer style auxiliary loss"""

        # 1. Dispatch frequency (f_i): fraction of tokens dispatched to expert i
        # selected_experts_flat: [total_tokens, K]
        # we only count the top-1 for the standard Switch loss
        top1_experts = selected_experts_flat[:, 0]
        f_i = torch.zeros(self.num_experts, device=router_logits.device)
        f_i.scatter_add_(0, top1_experts, torch.ones_like(top1_experts, dtype=torch.float32))
        f_i = f_i / total_tokens

        # 2. Routing probability (P_i): average routing probability for expert i
        probs = F.softmax(router_logits.view(-1, router_logits.size(-1)), dim=-1)
        P_i = probs.mean(dim=0)

        # Switch loss = N * sum(f_i * P_i)
        load_balance_loss = self.num_experts * torch.sum(f_i * P_i)

        # Router z-loss for numerical stability
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()

        total_aux_loss = (
            self.load_balance_loss_weight * load_balance_loss +
            0.001 * router_z_loss
        )

        return {
            'load_balance_loss': load_balance_loss,
            'router_z_loss': router_z_loss,
            'total_aux_loss': total_aux_loss
        }


__all__ = ['OptimizedMoELayer', 'OptimizedExpertRouter']
