"""Mixture of Experts (MoE) implementation for DeepSeek model"""

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


class ExpertRouter(nn.Module):
    """Router network for MoE layer"""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        router_bias: bool = False,
        router_dtype: torch.dtype = torch.float32,
        pre_router_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        
        if pre_router_dim:
            self.pre_router_net = nn.Sequential(
                nn.Linear(hidden_size, pre_router_dim),
                nn.ReLU(),
                nn.Linear(pre_router_dim, hidden_size)
            )
        else:
            self.pre_router_net = nn.Identity()

        # Router projection
        self.router_linear = nn.Linear(
            hidden_size,
            num_experts,
            bias=router_bias,
            dtype=router_dtype
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # Pre-router processing
        hidden_states = self.pre_router_net(hidden_states)

        # Compute router logits
        router_logits = self.router_linear(hidden_states)
        
        # Apply softmax to get routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k selection
        routing_weights, selected_experts = torch.topk(
            router_probs, 
            self.num_experts_per_token, 
            dim=-1
        )
        
        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, selected_experts, router_logits


class MoELayer(nn.Module):
    """Mixture of Experts layer with load balancing"""
    
    def _parallel_expert_forward(
        self,
        hidden_states_flat: torch.Tensor,
        routing_weights_flat: torch.Tensor,
        selected_experts_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Token-centric parallel expert processing
        Groups tokens by expert and processes all experts simultaneously
        """
        total_tokens = hidden_states_flat.size(0)
        device = hidden_states_flat.device
        
        # Pre-allocate output tensor (memory optimization)
        expert_outputs = torch.zeros_like(hidden_states_flat)
        expert_loads = torch.zeros(self.num_experts + self.shared_experts, device=device)
        
        # Create expert token assignment matrix
        expert_token_mask = torch.zeros(
            self.num_experts, total_tokens, 
            dtype=torch.bool, device=device
        )
        
        # Build mask matrix in one pass (vectorized)
        # Add bounds checking to prevent CUDA assertion errors
        max_expert_index = self.num_experts - 1  # Valid expert indices are 0 to num_experts-1
        
        for k in range(self.num_experts_per_token):
            expert_indices = selected_experts_flat[:, k]
            
            # Validate expert indices are within bounds
            if expert_indices.max() > max_expert_index:
                print(f"Warning: Expert index {expert_indices.max().item()} exceeds max {max_expert_index}, clamping")
                expert_indices = torch.clamp(expert_indices, 0, max_expert_index)
            
            mask = torch.zeros_like(expert_token_mask, dtype=torch.bool)
            mask.scatter_(0, expert_indices.unsqueeze(0), True)
            expert_token_mask |= mask
        
        # Process experts in parallel using batch operations
        for expert_idx in range(self.num_experts):
            expert_mask = expert_token_mask[expert_idx]
            
            if not expert_mask.any():
                continue
                
            # Get tokens and weights for this expert
            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            expert_tokens = hidden_states_flat[token_indices]
            
            # Aggregate weights for tokens assigned to this expert
            expert_weights = torch.zeros(
                token_indices.size(0), 
                device=device, dtype=routing_weights_flat.dtype
            )
            for k in range(self.num_experts_per_token):
                token_expert_mask = (selected_experts_flat[:, k] == expert_idx)
                expert_weights += routing_weights_flat[:, k] * token_expert_mask.float()
            expert_weights = expert_weights[token_expert_mask.sum().item():]
            
            # Apply expert (parallel computation)
            expert_output = self.experts[expert_idx](expert_tokens)
            
            # Apply weights and accumulate
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            expert_outputs[token_indices] += weighted_output
            
            # Track expert load
            expert_loads[expert_idx] = token_indices.size(0)
        
        return expert_outputs, expert_loads
    
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
        
        # Router
        self.router = ExpertRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,  # Only actual experts, not shared ones
            num_experts_per_token=num_experts_per_token,
            router_bias=router_bias,
            router_dtype=router_dtype
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
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # Route tokens to experts
        routing_weights, selected_experts, router_logits = self.router(hidden_states)

        # Flatten for expert processing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        routing_weights_flat = routing_weights.view(-1, self.num_experts_per_token)
        selected_experts_flat = selected_experts.view(-1, self.num_experts_per_token)
        
        # Token-centric parallel expert processing
        expert_outputs, expert_loads = self._parallel_expert_forward(
            hidden_states_flat, routing_weights_flat, selected_experts_flat
        )
        
        # Process shared experts
        shared_outputs = torch.zeros_like(hidden_states_flat)
        if self.shared_experts > 0:
            # Shared experts process all tokens
            for shared_idx in range(self.shared_experts):
                shared_output = self.shared_experts_layer[shared_idx](hidden_states_flat)
                # Apply uniform weight for shared experts
                shared_outputs += shared_output / self.shared_experts
                expert_loads[self.num_experts + shared_idx] = batch_size * sequence_length
        
        # Combine outputs
        final_outputs = expert_outputs + shared_outputs
        
        # Reshape back to original shape
        final_outputs = final_outputs.view(batch_size, sequence_length, hidden_dim)
        
        # Compute load balancing loss
        expert_loads_normalized = expert_loads / expert_loads.sum()
        ideal_load = 1.0 / (self.num_experts + self.shared_experts)
        load_balance_loss = F.mse_loss(expert_loads_normalized, torch.full_like(expert_loads_normalized, ideal_load))
        
        # Router z-loss (for stability)
        router_z_loss = torch.mean(torch.sum(torch.pow(router_logits, 2), dim=-1))
        
        # Combine losses
        total_loss = self.load_balance_loss_weight * load_balance_loss + 0.01 * router_z_loss
        
        aux_losses = {
            'load_balance_loss': load_balance_loss,
            'router_z_loss': router_z_loss,
            'total_aux_loss': total_loss
        }
        
        return final_outputs, total_loss, aux_losses


class DeepSeekMoEModel(nn.Module):
    """DeepSeek model with Mixture of Experts"""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        expert_capacity_factor: float = 1.25,
        load_balance_loss_weight: float = 0.01,
        shared_experts: int = 1,
        max_seq_length: int = 4096,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        use_moe_every_n_layers: int = 2  # Use MoE every N layers
    ):
        super().__init__()
        
        self.padding_idx = 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_length = max_seq_length
        self.use_moe_every_n_layers = use_moe_every_n_layers
        
        # Embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, self.padding_idx)
        
        # Transformer layers with MoE
        self.layers = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            if layer_idx % use_moe_every_n_layers == 0 and layer_idx > 0:  # Use MoE in selected layers
                # MoE layer
                moe_layer = MoELayer(
                    hidden_size=hidden_size,
                    num_experts=num_experts,
                    num_experts_per_token=num_experts_per_token,
                    expert_intermediate_size=intermediate_size,
                    dropout=dropout,
                    capacity_factor=expert_capacity_factor,
                    load_balance_loss_weight=load_balance_loss_weight,
                    shared_experts=shared_experts
                )
                self.layers.append(moe_layer)
            else:
                # Standard transformer block
                from .core import TransformerBlock
                transformer_block = TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=self.head_dim,
                    intermediate_size=intermediate_size,
                    norm_eps=norm_eps,
                    dropout=dropout
                )
                self.layers.append(transformer_block)
        
        # Final normalization
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using scaled normal distribution"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False
        return_dict = return_dict if return_dict is not None else True
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device)

        hidden_states = inputs_embeds

        # Prepare attention mask for the layers
        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                # Convert to causal mask
                causal_mask = torch.tril(torch.ones(seq_length, seq_length, device=attention_mask.device)).bool()
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) * causal_mask.unsqueeze(0)
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            else:
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_cache = () if use_cache else None
        total_aux_loss = 0
        
        for i, (layer_module, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Check if this is a MoE layer
            if hasattr(layer_module, 'experts'):  # MoE layer
                layer_outputs, aux_loss, aux_losses = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                )
                total_aux_loss += aux_loss
            else:  # Standard transformer block
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                layer_outputs = layer_outputs[0]  # Get hidden states only
            
            hidden_states = layer_outputs
            
            if use_cache and not hasattr(layer_module, 'experts'):  # Only cache for non-MoE layers
                next_cache += (layer_outputs[-1],)
            
            if output_attentions and not hasattr(layer_module, 'experts'):  # Only attention for non-MoE
                all_self_attns += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)
        
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, total_aux_loss]
                if v is not None
            )
        
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
            "aux_loss": total_aux_loss,
        }


def create_moe_config(
    model_size: str = "medium",
    num_experts: int = 8,
    num_experts_per_token: int = 2,
    expert_capacity_factor: float = 1.25,
    load_balance_loss_weight: float = 0.01,
    shared_experts: int = 1,
    use_moe_every_n_layers: int = 2
) -> Dict[str, Union[int, float]]:
    """Create MoE configuration based on model size"""
    
    if model_size == "small":
        return {
            "num_experts": min(num_experts, 8),
            "num_experts_per_token": min(num_experts_per_token, 2),
            "expert_capacity_factor": expert_capacity_factor,
            "load_balance_loss_weight": load_balance_loss_weight,
            "shared_experts": min(shared_experts, 1),
            "use_moe_every_n_layers": use_moe_every_n_layers
        }
    elif model_size == "medium":
        return {
            "num_experts": min(num_experts, 16),
            "num_experts_per_token": min(num_experts_per_token, 4),
            "expert_capacity_factor": expert_capacity_factor,
            "load_balance_loss_weight": load_balance_loss_weight,
            "shared_experts": min(shared_experts, 2),
            "use_moe_every_n_layers": max(1, use_moe_every_n_layers - 1)
        }
    else:  # large
        return {
            "num_experts": num_experts,
            "num_experts_per_token": num_experts_per_token,
            "expert_capacity_factor": expert_capacity_factor,
            "load_balance_loss_weight": load_balance_loss_weight,
            "shared_experts": shared_experts,
            "use_moe_every_n_layers": max(1, use_moe_every_n_layers - 2)
        }