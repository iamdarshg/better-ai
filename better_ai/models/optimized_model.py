"""Optimized DeepSeek MoE Model with all improvements integrated"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union, Any
from .core import RMSNorm


class OptimizedDeepSeekMoEModel(nn.Module):
    """
    Fully optimized DeepSeek model with:
    - Token-centric expert processing
    - Multi-Head Latent Attention (MLA)
    - DeepSeek Sparse Attention (DSA)
    - Memory-efficient operations
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        expert_capacity_factor: float = 1.25,
        load_balance_loss_weight: float = 0.01,
        shared_experts: int = 1,
        max_seq_length: int = 4096,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        use_moe_every_n_layers: int = 2,
        # Optimization flags
        use_token_centric_moe: bool = True,
        use_mla: bool = True,
        use_dsa: bool = False,
        mla_compression_ratio: float = 4.0,
        dsa_config: Optional[Dict] = None
    ):
        super().__init__()
        
        self.padding_idx = 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_length = max_seq_length
        self.use_moe_every_n_layers = use_moe_every_n_layers
        self.use_mla = use_mla
        self.use_dsa = use_dsa
        
        # Default DSA config
        if dsa_config is None:
            dsa_config = {
                'max_selected_tokens': 2048,
                'num_indexer_heads': 8,
                'indexer_temperature': 1.0,
                'local_window_size': 512,
                'use_flash_attention': True
            }
        
        # Embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, self.padding_idx)
        
        # Optimized transformer layers
        self.layers = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            if layer_idx % use_moe_every_n_layers == 0 and layer_idx > 0:
                # MoE layer with optimizations
                if use_token_centric_moe and use_mla:
                    # Use optimized MoE with MLA
                    from .attention_optimized import MoEWithMLABlock
                    layer = MoEWithMLABlock(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        num_key_value_heads=num_key_value_heads,
                        head_dim=self.head_dim,
                        num_experts=num_experts,
                        num_experts_per_token=num_experts_per_token,
                        expert_intermediate_size=intermediate_size,
                        compression_ratio=mla_compression_ratio,
                        norm_eps=norm_eps,
                        dropout=dropout,
                        use_mla=use_mla
                    )
                elif use_token_centric_moe:
                    # Use optimized MoE with standard attention
                    from .moe_optimized import OptimizedMoELayer
                    from .attention import FlashMultiHeadAttention
                    from .core import SwiGLU
                    
                    class CustomMoEBlock(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.self_attn = FlashMultiHeadAttention(
                                hidden_size=hidden_size,
                                num_heads=num_heads,
                                num_key_value_heads=num_key_value_heads,
                                head_dim=self.head_dim,
                                dropout=dropout,
                                use_flash_attention=True,
                                use_gqa=True
                            )
                            self.moe = OptimizedMoELayer(
                                hidden_size=hidden_size,
                                num_experts=num_experts,
                                num_experts_per_token=num_experts_per_token,
                                expert_intermediate_size=intermediate_size,
                                dropout=dropout,
                                capacity_factor=expert_capacity_factor,
                                load_balance_loss_weight=load_balance_loss_weight,
                                shared_experts=shared_experts
                            )
                            self.input_layernorm = RMSNorm(hidden_size, eps=norm_eps)
                            self.post_attention_layernorm = RMSNorm(hidden_size, eps=norm_eps)
                            self.residual_dropout = nn.Dropout(dropout)
                        
                        def forward(self, hidden_states, attention_mask=None, past_key_value=None, 
                                   output_attentions=False, use_cache=False):
                            # Self-attention
                            residual = hidden_states
                            hidden_states = self.input_layernorm(hidden_states)
                            
                            attn_outputs = self.self_attn(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                past_key_value=past_key_value,
                                output_attentions=output_attentions,
                                use_cache=use_cache
                            )
                            
                            attn_output, attn_weights, present_key_value = attn_outputs
                            hidden_states = self.residual_dropout(attn_output)
                            hidden_states = residual + hidden_states
                            
                            # MoE
                            residual = hidden_states
                            hidden_states = self.post_attention_layernorm(hidden_states)
                            
                            moe_output, aux_loss, aux_losses = self.moe(
                                hidden_states,
                                attention_mask=attention_mask
                            )
                            
                            hidden_states = self.residual_dropout(moe_output)
                            hidden_states = residual + hidden_states
                            
                            outputs = (hidden_states, aux_loss)
                            
                            if output_attentions:
                                outputs += (attn_weights,)
                            
                            if use_cache:
                                outputs += (present_key_value,)
                            
                            return outputs
                    
                    layer = CustomMoEBlock()
                else:
                    # Fallback to original MoE implementation
                    from .moe import DeepSeekMoEModel
                    original_model = DeepSeekMoEModel(
                        vocab_size=vocab_size,
                        hidden_size=hidden_size,
                        num_layers=1,  # Single layer
                        num_heads=num_heads,
                        num_key_value_heads=num_key_value_heads,
                        intermediate_size=intermediate_size,
                        num_experts=num_experts,
                        num_experts_per_token=num_experts_per_token,
                        expert_capacity_factor=expert_capacity_factor,
                        load_balance_loss_weight=load_balance_loss_weight,
                        shared_experts=shared_experts,
                        max_seq_length=max_seq_length,
                        norm_eps=norm_eps,
                        dropout=dropout,
                        use_moe_every_n_layers=1  # Always use MoE
                    )
                    layer = original_model.layers[0]  # Get the MoE layer
                
                self.layers.append(layer)
            else:
                # Standard transformer block with optimizations
                if use_dsa:
                    from .dsa import OptimizedTransformerBlockWithDSA
                    layer = OptimizedTransformerBlockWithDSA(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        num_key_value_heads=num_key_value_heads,
                        head_dim=self.head_dim,
                        intermediate_size=intermediate_size,
                        norm_eps=norm_eps,
                        dropout=dropout,
                        use_dsa=True,
                        use_mla=False,
                        dsa_config=dsa_config
                    )
                elif use_mla:
                    from .attention_optimized import MultiHeadLatentAttention
                    from .core import SwiGLU
                    
                    class CustomTransformerBlock(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.self_attn = MultiHeadLatentAttention(
                                hidden_size=hidden_size,
                                num_heads=num_heads,
                                num_key_value_heads=num_key_value_heads,
                                head_dim=self.head_dim,
                                compression_ratio=mla_compression_ratio,
                                norm_eps=norm_eps,
                                dropout=dropout,
                                use_flash_attention=True
                            )
                            self.mlp = SwiGLU(hidden_size, intermediate_size)
                            self.input_layernorm = RMSNorm(hidden_size, eps=norm_eps)
                            self.post_attention_layernorm = RMSNorm(hidden_size, eps=norm_eps)
                            self.residual_dropout = nn.Dropout(dropout)
                        
                        def forward(self, hidden_states, attention_mask=None, past_key_value=None,
                                   output_attentions=False, use_cache=False):
                            # Self-attention
                            residual = hidden_states
                            hidden_states = self.input_layernorm(hidden_states)
                            
                            attn_outputs = self.self_attn(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                past_key_value=past_key_value,
                                use_cache=use_cache,
                                output_attentions=output_attentions
                            )
                            
                            attn_output, attn_weights, present_key_value = attn_outputs
                            hidden_states = self.residual_dropout(attn_output)
                            hidden_states = residual + hidden_states
                            
                            # Feed-forward
                            residual = hidden_states
                            hidden_states = self.post_attention_layernorm(hidden_states)
                            hidden_states = self.mlp(hidden_states)
                            hidden_states = self.residual_dropout(hidden_states)
                            hidden_states = residual + hidden_states
                            
                            outputs = (hidden_states,)
                            
                            if output_attentions:
                                outputs += (attn_weights,)
                            
                            if use_cache:
                                outputs += (present_key_value,)
                            
                            return outputs
                    
                    layer = CustomTransformerBlock()
                else:
                    # Fallback to standard optimized transformer
                    from .attention import OptimizedTransformerBlock
                    layer = OptimizedTransformerBlock(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        num_key_value_heads=num_key_value_heads,
                        head_dim=self.head_dim,
                        intermediate_size=intermediate_size,
                        norm_eps=norm_eps,
                        dropout=dropout,
                        use_flash_attention=True,
                        use_gqa=True
                    )
                
                self.layers.append(layer)
        
        # Final normalization
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        
        # Memory management buffers
        self.register_buffer('_cached_attention_mask', None)
        self.register_buffer('_cached_position_ids', None)
        
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
    
    def _create_optimized_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """
        Create optimized attention mask with caching
        """
        batch_size, seq_length = input_shape
        
        # Check if we can reuse cached mask
        if (self._cached_attention_mask is not None and 
            self._cached_attention_mask.shape == (batch_size, seq_length, seq_length) and
            self._cached_attention_mask.device == device):
            return self._cached_attention_mask.to(dtype)
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=device)
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_length, seq_length, device=device)).bool()
        
        # Expand and combine
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) * causal_mask.unsqueeze(0)
        attention_mask = attention_mask.to(dtype)
        
        # Cache for future use
        self._cached_attention_mask = attention_mask.detach().clone()
        
        return attention_mask
    
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
        
        # Create optimized attention mask
        attention_mask = self._create_optimized_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds.device, inputs_embeds.dtype
        )
        
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_cache = () if use_cache else None
        total_aux_loss = 0
        
        for i, (layer_module, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Handle different layer types
            if hasattr(layer_module, 'moe'):  # MoE layer
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                
                # Handle different return formats
                if len(layer_outputs) >= 2 and isinstance(layer_outputs[1], torch.Tensor):
                    hidden_states, aux_loss = layer_outputs[:2]
                    total_aux_loss += aux_loss
                    
                    if output_attentions and len(layer_outputs) > 2:
                        all_self_attns += (layer_outputs[2],)
                    
                    if use_cache and len(layer_outputs) > 3:
                        next_cache += (layer_outputs[3],)
                else:
                    hidden_states = layer_outputs[0]
                    
                    if output_attentions and len(layer_outputs) > 1:
                        all_self_attns += (layer_outputs[1],)
                    
                    if use_cache and len(layer_outputs) > 2:
                        next_cache += (layer_outputs[2],)
            else:  # Standard transformer block
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                
                hidden_states = layer_outputs[0]
                
                if output_attentions and len(layer_outputs) > 1:
                    all_self_attns += (layer_outputs[1],)
                
                if use_cache and len(layer_outputs) > 2:
                    next_cache += (layer_outputs[2],)
        
        hidden_states = self.norm(hidden_states)
        
        # Add last hidden state
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
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics for monitoring"""
        stats = {
            'use_mla': self.use_mla,
            'use_dsa': self.use_dsa,
            'num_layers': len(self.layers),
            'moe_layers': sum(1 for layer in self.layers if hasattr(layer, 'moe')),
            'memory_cache_active': self._cached_attention_mask is not None,
        }
        
        # Add per-layer stats
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'moe') and hasattr(layer.moe, 'router'):
                stats[f'layer_{i}_expert_loads'] = layer.moe.router.expert_loads_ema.tolist()
        
        return stats


# Export the optimized model
__all__ = ['OptimizedDeepSeekMoEModel']