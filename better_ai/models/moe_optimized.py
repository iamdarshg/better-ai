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
    """Single expert layer with SwiGLU activation"""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.dropout(self.down_proj(F.silu(gate) * up))


class OptimizedExpertRouter(nn.Module):
    """Optimized router with load-aware routing"""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        router_bias: bool = False,
        router_dtype: torch.dtype = torch.float32,
        capacity_factor: float = 1.25
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.capacity_factor = capacity_factor
        
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
        
        # Compute router logits
        router_logits = self.get_logits(hidden_states)
        
        # Apply softmax with temperature for better routing
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k selection
        routing_weights, selected_experts = torch.topk(
            routing_probs, 
            self.num_experts_per_token, 
            dim=-1
        )
        
        # Normalize weights
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
        loss_free_balancing: bool = True
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
            capacity_factor=capacity_factor
        )

        if self.loss_free_balancing:
            self.balancer = LossFreeBalancing(
                num_experts=num_experts,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        else:
            self.balancer = None
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(
                hidden_size=hidden_size,
                intermediate_size=expert_intermediate_size,
                dropout=dropout
            ) for _ in range(num_experts)
        ])
        
        # Shared experts (always active)
        if shared_experts > 0:
            self.shared_experts_layer = nn.ModuleList([
                Expert(
                    hidden_size=hidden_size,
                    intermediate_size=expert_intermediate_size,
                    dropout=dropout
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
            
            # Use index_add_ for better performance and memory locality
            expert_input = hidden_states_flat[token_indices]
            weights = routing_weights_flat[token_indices, k_indices].unsqueeze(-1)

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
                router_logits, compute_loads=True
            )
        else:
            routing_weights, selected_experts, router_logits = self.router(hidden_states)
        
        # Flatten for processing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        routing_weights_flat = routing_weights.view(-1, self.num_experts_per_token)
        selected_experts_flat = selected_experts.view(-1, self.num_experts_per_token)
        total_tokens = hidden_states_flat.size(0)
        
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
