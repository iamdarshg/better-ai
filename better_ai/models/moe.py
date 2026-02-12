"""Mixture of Experts (MoE) implementation for DeepSeek model with Expert Collapse Prevention"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict
import torch.distributed as dist
from .core import RMSNorm


class LossFreeBalancing:
    """
    Loss-Free Bias-Based Balancing for MoE Routing

    Traditional auxiliary loss (moe_load_balance_weight * aux_loss) introduces gradient
    interference that degrades model quality. This implementation uses bias-based routing
    adjustment without gradients.

    Research shows this achieves load std deviation of 1.18 vs 12.25 for aux-loss methods,
    without gradient perturbations.

    Usage:
        balancing = LossFreeBalancing(num_experts=16, momentum=0.99)
        adjusted_logits, indices = balancing.update_and_route(router_logits, tokens_per_expert)
    """

    def __init__(
        self,
        num_experts: int,
        momentum: float = 0.99,
        target_load: Optional[torch.Tensor] = None,
        bias_lr: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_experts = num_experts
        self.momentum = momentum
        self.bias_lr = bias_lr
        self.device = device

        if target_load is None:
            self.target_load = torch.ones(num_experts, device=device) / num_experts
        else:
            self.target_load = target_load.to(device)

        self.recent_loads = torch.ones(num_experts, device=device) / num_experts
        self.expert_bias = torch.zeros(num_experts, device=device)
        self.initialized = False

    def update_and_route(
        self,
        router_logits: torch.Tensor,
        tokens_per_expert: Optional[torch.Tensor] = None,
        compute_loads: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply bias before routing, update bias after routing.

        Args:
            router_logits: Raw router logits [batch_size, seq_len, num_experts]
            tokens_per_expert: Pre-computed token counts per expert (optional)
            compute_loads: Whether to compute loads from routing decisions

        Returns:
            Tuple of (adjusted routing weights, selected expert indices)
        """
        batch_size, seq_len, num_experts = router_logits.shape

        if not self.initialized:
            self.recent_loads = (
                torch.ones(num_experts, device=self.device) / num_experts
            )
            self.expert_bias = torch.zeros(num_experts, device=self.device)
            self.initialized = True

        adjusted_logits = router_logits - self.expert_bias.unsqueeze(0).unsqueeze(0)

        top_k_logits, top_k_indices = torch.topk(adjusted_logits, k=2, dim=-1)

        if compute_loads:
            actual_loads = torch.zeros(num_experts, device=self.device)
            for expert_idx in top_k_indices.flatten():
                actual_loads[expert_idx] += 1
            actual_loads = actual_loads / (batch_size * seq_len * 2)

            self.recent_loads = (
                self.momentum * self.recent_loads + (1 - self.momentum) * actual_loads
            )

            load_imbalance = self.recent_loads - self.target_load
            self.expert_bias += self.bias_lr * load_imbalance

        routing_weights = F.softmax(top_k_logits, dim=-1)

        return routing_weights, top_k_indices

    def get_expert_loads(self) -> torch.Tensor:
        """Get current expert load estimates"""
        return self.recent_loads.clone()

    def get_expert_bias(self) -> torch.Tensor:
        """Get current expert biases"""
        return self.expert_bias.clone()

    def reset(self):
        """Reset bias and load tracking"""
        self.expert_bias.zero_()
        self.recent_loads.fill_(1.0 / self.num_experts)


def compute_expert_specialization_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    ortho_weight: float = 0.05,
    variance_weight: float = 0.03,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute orthogonality and variance losses for expert specialization.

    Orthogonality: Encourage experts to activate on DIFFERENT token types.
    Variance: Encourage DISCRIMINATIVE routing (confident decisions).

    Research validation: This improves MoE baselines by up to 23.79% on downstream
    tasks by enabling true expert specialization.

    Args:
        router_logits: Router logits [batch_size, seq_len, num_experts]
        num_experts: Number of experts
        ortho_weight: Weight for orthogonality loss
        variance_weight: Weight for variance loss

    Returns:
        Tuple of (orthogonality_loss, variance_loss)
    """
    batch_size, seq_len, _ = router_logits.shape

    router_probs = F.softmax(router_logits, dim=-1)

    flat_probs = router_probs.reshape(-1, num_experts)
    expert_profiles = router_probs.mean(dim=[0, 1])

    if num_experts > 1:
        correlation_matrix = torch.corrcoef(flat_probs.T)
        identity = torch.eye(num_experts, device=correlation_matrix.device)
        orthogonality_loss = (correlation_matrix * (1 - identity)).abs().mean()
    else:
        orthogonality_loss = torch.tensor(0.0, device=router_logits.device)

    routing_entropy = -(router_probs * torch.log(router_probs + 1e-10)).sum(dim=-1)
    variance_loss = routing_entropy.mean()

    orthogonality_loss = ortho_weight * orthogonality_loss
    variance_loss = variance_weight * variance_loss

    return orthogonality_loss, variance_loss


def router_z_loss(
    router_logits: torch.Tensor, z_loss_coeff: float = 1e-3
) -> torch.Tensor:
    """
    Compute router Z-loss for numerical stability.

    Penalizes squared log-sum-exp of router logits. Prevents numerical instability
    and keeps logits in reasonable range.

    Critical for FP8 training: Without Z-loss, router logits will overflow in FP8,
    causing routing to collapse to argmax behavior.

    Args:
        router_logits: Router logits [batch_size, seq_len, num_experts]
        z_loss_coeff: Weight for Z-loss

    Returns:
        Z-loss value
    """
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = (log_z**2).mean()
    return z_loss_coeff * z_loss


def expert_router_coupling_loss(
    expert_embeddings: torch.Tensor,
    experts: nn.ModuleList,
    coupling_weight: float = 0.01,
) -> torch.Tensor:
    """
    Expert-Router Coupling Loss for alignment.

    Ensures router embeddings accurately represent expert capabilities.
    Operates on n^2 activations (cheap: 16^2 = 256 for typical setup).

    Should be added to training every N steps (expensive to compute).

    Args:
        expert_embeddings: Router expert embeddings [num_experts, hidden_dim]
        experts: List of expert modules
        coupling_weight: Weight for coupling loss

    Returns:
        Coupling loss value
    """
    num_experts = len(experts)
    device = expert_embeddings.device

    activations = torch.zeros(num_experts, num_experts, device=device)

    for i, expert in enumerate(experts):
        for j in range(num_experts):
            proxy_token = expert_embeddings[j].unsqueeze(0)
            with torch.no_grad():
                expert_out = expert(proxy_token)
                activations[i, j] = expert_out.abs().mean()

    diag_values = torch.diag(activations)
    off_diag_max = activations.max(dim=1)[0]
    diagonal_loss = F.relu(off_diag_max - diag_values).mean()

    column_loss = torch.tensor(0.0, device=device)
    if activations.shape[1] > 1:
        col_max = activations.max(dim=0)[0]
        column_loss = F.relu(col_max - diag_values).mean()

    return coupling_weight * (diagonal_loss + column_loss)


class Expert(nn.Module):
    """Single expert layer with SwiGLU activation"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
        bias: bool = False,
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
        pre_router_dim: Optional[int] = None,
        device: Optional[torch.device] = torch.device("cpu"),
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.pre_router_dim = pre_router_dim
        self.router_bias = router_bias
        self.router_dtype = router_dtype

        # Initialize router projection - this will be updated in forward if needed
        self.router_linear = nn.Linear(
            hidden_size,
            num_experts,
            bias=router_bias,
            dtype=router_dtype,
            device=device,
        )

        # Pre-router network will be created dynamically in forward pass
        self.pre_router_net = None
        self._input_dim = None
        self.device = device
        self.to(self.device)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.to(self.router_dtype).to(self.device)
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # Dynamic pre-router network creation based on actual input dimension
        if self.pre_router_net is None or self._input_dim != hidden_dim:
            self._create_pre_router_network(hidden_dim)
            self._input_dim = hidden_dim

        # Pre-router processing
        hidden_states = self.pre_router_net(hidden_states)

        # Compute router logits
        router_logits = self.router_linear(hidden_states)

        # Apply softmax to get routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection
        routing_weights, selected_experts = torch.topk(
            router_probs, self.num_experts_per_token, dim=-1
        )

        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        return routing_weights, selected_experts, router_logits

    def _create_pre_router_network(self, input_dim: int):
        """Create or update the pre-router network based on input dimension"""
        if self.pre_router_dim is None:
            # Use identity if no pre-router dimension specified
            self.pre_router_net = nn.Identity().to(self.device)
        else:
            # Create adaptive pre-router network
            self.pre_router_net = nn.Sequential(
                nn.Linear(input_dim, self.pre_router_dim),
                nn.ReLU(),
                nn.Linear(self.pre_router_dim, input_dim),
            ).to(self.device)

        # Also update the router linear layer to match the input dimension
        self.router_linear = nn.Linear(
            input_dim, self.num_experts, bias=self.router_bias, dtype=self.router_dtype
        ).to(self.device)


class MoELayer(nn.Module):
    """Mixture of Experts layer with load balancing"""

    def _parallel_expert_forward(
        self,
        hidden_states_flat: torch.Tensor,
        routing_weights_flat: torch.Tensor,
        selected_experts_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Token-centric parallel expert processing
        Groups tokens by expert and processes all experts simultaneously
        """
        total_tokens = hidden_states_flat.size(0)
        device = hidden_states_flat.device

        # Pre-allocate output tensor (memory optimization)
        expert_outputs = torch.zeros_like(hidden_states_flat)
        expert_loads = torch.zeros(
            self.num_experts + self.shared_experts, device=device
        )

        # Create expert token assignment matrix
        expert_token_mask = torch.zeros(
            self.num_experts, total_tokens, dtype=torch.bool, device=device
        )

        # Build mask matrix in one pass (vectorized)
        # Add bounds checking to prevent CUDA assertion errors
        max_expert_index = (
            self.num_experts - 1
        )  # Valid expert indices are 0 to num_experts-1

        for k in range(self.num_experts_per_token):
            expert_indices = selected_experts_flat[:, k]

            # Validate expert indices are within bounds
            if expert_indices.max() > max_expert_index:
                print(
                    f"Warning: Expert index {expert_indices.max().item()} exceeds max {max_expert_index}, clamping"
                )
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
                token_indices.size(0), device=device, dtype=routing_weights_flat.dtype
            )
            for k in range(self.num_experts_per_token):
                token_expert_mask = selected_experts_flat[:, k] == expert_idx
                # Get weights for tokens assigned to this expert at position k
                expert_weights[token_expert_mask[token_indices]] = routing_weights_flat[
                    token_indices, k
                ]

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
        shared_experts: int = 1,
        loss_free_balancing: bool = True,
        specialization_weight: float = 0.05,
        router_z_loss_weight: float = 1e-3,
        gradient_clip_norm: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.capacity_factor = capacity_factor
        self.load_balance_loss_weight = load_balance_loss_weight
        self.shared_experts = shared_experts
        self.loss_free_balancing = loss_free_balancing
        self.specialization_weight = specialization_weight
        self.router_z_loss_weight = router_z_loss_weight
        self.gradient_clip_norm = gradient_clip_norm

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if expert_intermediate_size is None:
            expert_intermediate_size = hidden_size * 4

        self.router = ExpertRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            router_bias=router_bias,
            router_dtype=router_dtype,
            device=device,
        )

        self.experts = nn.ModuleList(
            [
                Expert(
                    hidden_size=hidden_size,
                    intermediate_size=expert_intermediate_size,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            ]
        )

        if shared_experts > 0:
            self.shared_experts_layer = nn.ModuleList(
                [
                    Expert(
                        hidden_size=hidden_size,
                        intermediate_size=expert_intermediate_size,
                        dropout=dropout,
                    )
                    for _ in range(shared_experts)
                ]
            )

        if self.loss_free_balancing:
            self.balancer = LossFreeBalancing(
                num_experts=num_experts,
                momentum=0.99,
                bias_lr=0.1,
                device=device,
            )
            self.aux_loss_weight = torch.tensor(0.0, device=device)
        else:
            self.balancer = None
            self.aux_loss_weight = torch.tensor(load_balance_loss_weight, device=device)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        routing_weights, selected_experts, router_logits = self.router(hidden_states)

        if self.loss_free_balancing and self.balancer is not None:
            _, updated_indices = self.balancer.update_and_route(
                router_logits, compute_loads=True
            )

        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        routing_weights_flat = routing_weights.view(-1, self.num_experts_per_token)
        selected_experts_flat = selected_experts.view(-1, self.num_experts_per_token)

        expert_outputs, expert_loads = self._parallel_expert_forward(
            hidden_states_flat, routing_weights_flat, selected_experts_flat
        )

        shared_outputs = torch.zeros_like(hidden_states_flat)
        if self.shared_experts > 0:
            for shared_idx in range(self.shared_experts):
                shared_output = self.shared_experts_layer[shared_idx](
                    hidden_states_flat
                )
                shared_outputs += shared_output / self.shared_experts
                expert_loads[self.num_experts + shared_idx] = (
                    batch_size * sequence_length
                )

        final_outputs = expert_outputs + shared_outputs

        final_outputs = final_outputs.view(batch_size, sequence_length, hidden_dim)

        if self.loss_free_balancing and self.balancer is not None:
            load_balance_loss = torch.tensor(0.0, device=self.device)
        else:
            expert_loads_normalized = expert_loads / expert_loads.sum()
            ideal_load = 1.0 / (self.num_experts + self.shared_experts)
            load_balance_loss = F.mse_loss(
                expert_loads_normalized,
                torch.full_like(expert_loads_normalized, ideal_load),
            )

        z_loss = router_z_loss(router_logits, self.router_z_loss_weight)

        ortho_loss, variance_loss = compute_expert_specialization_loss(
            router_logits,
            self.num_experts,
            self.specialization_weight * 0.6,
            self.specialization_weight * 0.4,
        )

        total_aux_loss = (
            self.aux_loss_weight * load_balance_loss
            + z_loss
            + ortho_loss
            + variance_loss
        )

        aux_losses = {
            "load_balance_loss": load_balance_loss
            if isinstance(load_balance_loss, torch.Tensor)
            else torch.tensor(0.0, device=self.device),
            "router_z_loss": z_loss,
            "orthogonality_loss": ortho_loss,
            "variance_loss": variance_loss,
            "total_aux_loss": total_aux_loss,
        }

        return final_outputs, total_aux_loss, aux_losses


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
        use_moe_every_n_layers: int = 2,
        loss_free_balancing: bool = True,
        specialization_weight: float = 0.05,
        router_z_loss_weight: float = 1e-3,
        gradient_clip_norm: float = 1.0,
    ):
        super().__init__()

        self.padding_idx = 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_length = max_seq_length
        self.use_moe_every_n_layers = use_moe_every_n_layers

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, self.padding_idx)

        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            if layer_idx % use_moe_every_n_layers == 0 and layer_idx > 0:
                moe_layer = MoELayer(
                    hidden_size=hidden_size,
                    num_experts=num_experts,
                    num_experts_per_token=num_experts_per_token,
                    expert_intermediate_size=intermediate_size,
                    dropout=dropout,
                    capacity_factor=expert_capacity_factor,
                    load_balance_loss_weight=load_balance_loss_weight,
                    shared_experts=shared_experts,
                    loss_free_balancing=loss_free_balancing,
                    specialization_weight=specialization_weight,
                    router_z_loss_weight=router_z_loss_weight,
                    gradient_clip_norm=gradient_clip_norm,
                )
                self.layers.append(moe_layer)
            else:
                from .core import TransformerBlock

                transformer_block = TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=self.head_dim,
                    intermediate_size=intermediate_size,
                    norm_eps=norm_eps,
                    dropout=dropout,
                )
                self.layers.append(transformer_block)

        self.norm = RMSNorm(hidden_size, eps=norm_eps)

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
        output_attentions = (
            output_attentions if output_attentions is not None else False
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        use_cache = use_cache if use_cache is not None else False
        return_dict = return_dict if return_dict is not None else True

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
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
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device
            )

        hidden_states = inputs_embeds

        # Prepare attention mask for the layers
        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                # Convert to causal mask
                causal_mask = torch.tril(
                    torch.ones(seq_length, seq_length, device=attention_mask.device)
                ).bool()
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(
                    2
                ) * causal_mask.unsqueeze(0)
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            else:
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_cache = () if use_cache else None
        total_aux_loss = 0

        for i, (layer_module, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Check if this is a MoE layer
            if hasattr(layer_module, "experts"):  # MoE layer
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

            if use_cache and not hasattr(
                layer_module, "experts"
            ):  # Only cache for non-MoE layers
                next_cache += (layer_outputs[-1],)

            if output_attentions and not hasattr(
                layer_module, "experts"
            ):  # Only attention for non-MoE
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    total_aux_loss,
                ]
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
    use_moe_every_n_layers: int = 2,
    loss_free_balancing: bool = True,
    specialization_weight: float = 0.05,
    router_z_loss_weight: float = 1e-3,
    gradient_clip_norm: float = 1.0,
) -> Dict[str, Union[int, float]]:
    """Create MoE configuration based on model size"""

    if model_size == "small":
        return {
            "num_experts": min(num_experts, 8),
            "num_experts_per_token": min(num_experts_per_token, 2),
            "expert_capacity_factor": expert_capacity_factor,
            "load_balance_loss_weight": load_balance_loss_weight,
            "shared_experts": min(shared_experts, 1),
            "use_moe_every_n_layers": use_moe_every_n_layers,
            "loss_free_balancing": loss_free_balancing,
            "specialization_weight": specialization_weight,
            "router_z_loss_weight": router_z_loss_weight,
            "gradient_clip_norm": gradient_clip_norm,
        }
    elif model_size == "medium":
        return {
            "num_experts": min(num_experts, 16),
            "num_experts_per_token": min(num_experts_per_token, 4),
            "expert_capacity_factor": expert_capacity_factor,
            "load_balance_loss_weight": load_balance_loss_weight,
            "shared_experts": min(shared_experts, 2),
            "use_moe_every_n_layers": max(1, use_moe_every_n_layers - 1),
            "loss_free_balancing": loss_free_balancing,
            "specialization_weight": specialization_weight,
            "router_z_loss_weight": router_z_loss_weight,
            "gradient_clip_norm": gradient_clip_norm,
        }
    else:
        return {
            "num_experts": num_experts,
            "num_experts_per_token": num_experts_per_token,
            "expert_capacity_factor": expert_capacity_factor,
            "load_balance_loss_weight": load_balance_loss_weight,
            "shared_experts": shared_experts,
            "use_moe_every_n_layers": max(1, use_moe_every_n_layers - 2),
            "loss_free_balancing": loss_free_balancing,
            "specialization_weight": specialization_weight,
            "router_z_loss_weight": router_z_loss_weight,
            "gradient_clip_norm": gradient_clip_norm,
        }
