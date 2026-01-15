"""Optimized Mixture of Experts (MoE) implementation with token-centric processing"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict
import torch.distributed as dist
from .core import RMSNorm


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
    
    def forward(
        self, 
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # Compute router logits
        router_logits = self.router_linear(hidden_states)
        
        # Apply softmax with temperature for better routing
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k selection with load-aware adjustment
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
    """Optimized MoE layer with token-centric processing and memory efficiency"""
    
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
        shared_experts: int = 1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.capacity_factor = capacity_factor
        self.load_balance_loss_weight = load_balance_loss_weight
        self.shared_experts = shared_experts
        
        if expert_intermediate_size is None:
            expert_intermediate_size = hidden_size * 4
        
        # Optimized router
        self.router = OptimizedExpertRouter(
            hidden_size=hidden_size,
            num_experts=num_experts + shared_experts,
            num_experts_per_token=num_experts_per_token,
            router_bias=router_bias,
            router_dtype=router_dtype,
            capacity_factor=capacity_factor
        )
        
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
        
        # Pre-allocated buffers for efficiency
        self.register_buffer('_expert_outputs_buffer', None)
        self.register_buffer('_expert_loads_buffer', None)
    
    def _token_centric_expert_forward(
        self,
        hidden_states_flat: torch.Tensor,
        routing_weights_flat: torch.Tensor,
        selected_experts_flat: torch.Tensor,
        total_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized token-centric expert processing
        Groups tokens by expert and processes efficiently
        """
        device = hidden_states_flat.device
        
        # Initialize output and loads
        expert_outputs = torch.zeros_like(hidden_states_flat)
        expert_loads = torch.zeros(self.num_experts + self.shared_experts, device=device)
        
        # Create mapping from tokens to experts
        token_to_experts = []
        for k in range(self.num_experts_per_token):
            expert_indices = selected_experts_flat[:, k]
            weights = routing_weights_flat[:, k]
            token_to_experts.append((expert_indices, weights))
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find all tokens that should go to this expert
            expert_token_mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)
            expert_token_weights = torch.zeros(total_tokens, device=device, dtype=hidden_states_flat.dtype)
            
            for expert_indices, weights in token_to_experts:
                mask = (expert_indices == expert_idx)
                expert_token_mask |= mask
                expert_token_weights[mask] = weights[mask]
            
            if not expert_token_mask.any():
                continue
            
            # Get tokens for this expert
            token_indices = expert_token_mask.nonzero(as_tuple=True)[0]
            expert_tokens = hidden_states_flat[token_indices]
            expert_weights = expert_token_weights[token_indices]
            
            # Apply expert
            expert_output = self.experts[expert_idx](expert_tokens)
            
            # Weight and accumulate
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            expert_outputs[token_indices] += weighted_output
            
            expert_loads[expert_idx] = token_indices.size(0)
        
        return expert_outputs, expert_loads
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # Route tokens to experts
        routing_weights, selected_experts, router_logits = self.router(hidden_states)
        
        # Flatten for processing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        routing_weights_flat = routing_weights.view(-1, self.num_experts_per_token)
        selected_experts_flat = selected_experts.view(-1, self.num_experts_per_token)
        total_tokens = hidden_states_flat.size(0)
        
        # Token-centric expert processing
        expert_outputs, expert_loads = self._token_centric_expert_forward(
            hidden_states_flat, routing_weights_flat, selected_experts_flat, total_tokens
        )
        
        # Process shared experts
        if self.shared_experts > 0:
            shared_output = torch.zeros_like(hidden_states_flat)
            for shared_idx in range(self.shared_experts):
                shared_expert_output = self.shared_experts_layer[shared_idx](hidden_states_flat)
                shared_output += shared_expert_output / self.shared_experts
                expert_loads[self.num_experts + shared_idx] = total_tokens
            
            expert_outputs += shared_output
        
        # Reshape back to original shape
        final_outputs = expert_outputs.view(batch_size, sequence_length, hidden_dim)
        
        # Update router load statistics
        self.router.update_load_stats(expert_loads)
        
        # Compute losses (simplified and optimized)
        aux_losses = self._compute_aux_losses(router_logits, expert_loads)
        
        return final_outputs, aux_losses['total_aux_loss'], aux_losses
    
    def _compute_aux_losses(
        self, 
        router_logits: torch.Tensor, 
        expert_loads: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Optimized auxiliary loss computation"""
        
        # Load balancing loss (simplified)
        expert_loads_normalized = expert_loads / (expert_loads.sum() + 1e-6)
        ideal_load = torch.full_like(expert_loads_normalized, 1.0 / (self.num_experts + self.shared_experts))
        load_balance_loss = F.mse_loss(expert_loads_normalized, ideal_load)
        
        # Router z-loss (regularization)
        router_z_loss = torch.mean(torch.sum(torch.pow(router_logits, 2), dim=-1))
        
        # Combine losses
        total_aux_loss = (
            self.load_balance_loss_weight * load_balance_loss + 
            0.01 * router_z_loss
        )
        
        return {
            'load_balance_loss': load_balance_loss,
            'router_z_loss': router_z_loss,
            'total_aux_loss': total_aux_loss
        }


# Export the optimized class
__all__ = ['OptimizedMoELayer', 'OptimizedExpertRouter']