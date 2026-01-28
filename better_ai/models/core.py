"""Core transformer components for DeepSeek model"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Any
from .rope import RoPECache


class RMSNorm(nn.Module):
    """RMS Normalization layer"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with support for GQA"""
    
    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int, head_dim: int, dropout: float = 0.0, use_nope: bool = False):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.use_nope = use_nope
        
        if head_dim * num_heads != hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads")
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        # Dropout
        self.attention_dropout = nn.Dropout(dropout)

        if not self.use_nope:
            self.rope_cache = RoPECache(
                dim=self.head_dim,
                max_seq_len=4096,
                base=10000,
                device=torch.device("cpu")
            )
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int, num_heads: int):
        """Reshape tensor for attention computation"""
        return tensor.view(bsz, seq_len, num_heads, self.head_dim).transpose(1, 2)
    
    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value heads to match query heads"""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention
        query_states = self._shape(query_states, q_len, bsz, self.num_heads)
        key_states = self._shape(key_states, q_len, bsz, self.num_key_value_heads)
        value_states = self._shape(value_states, q_len, bsz, self.num_key_value_heads)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if not self.use_nope:
            self.rope_cache.to(query_states.device)
            query_states, key_states = self.rope_cache(query_states, key_states)
        
        # Handle key-value caching for inference
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        if use_cache:
            present = (key_states, value_states)
        else:
            present = None
        
        # Handle GQA (Grouped Query Attention)
        if self.num_key_value_groups > 1:
            key_states = self.repeat_kv(key_states, self.num_key_value_groups)
            value_states = self.repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights += attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, present


class TransformerBlock(nn.Module):
    """Transformer block with RMSNorm and SwiGLU"""
    
    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int, head_dim: int, 
                 intermediate_size: int, norm_eps: float = 1e-6, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Attention
        self.self_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            dropout=dropout
        )
        
        # Feed-forward
        self.mlp = SwiGLU(hidden_size, intermediate_size)
        
        # Normalization
        self.input_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        
        # Dropout
        self.residual_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = self.residual_dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # Feed-forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.residual_dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
        
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs


class DeepSeekModel(nn.Module):
    """DeepSeek-inspired Transformer model"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = 0
        self.hidden_size = config.hidden_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_seq_length = config.max_seq_length
        
        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, self.hidden_size, self.padding_idx)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                intermediate_size=config.intermediate_dim,
                norm_eps=config.norm_eps,
                dropout=config.residual_dropout
            ) for _ in range(config.num_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(self.hidden_size, eps=self.config.norm_eps)
        
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

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings matrix of the model if new_num_tokens != config.vocab_size."""
        if new_num_tokens is None or new_num_tokens == self.config.vocab_size:
            return

        old_embeddings = self.get_input_embeddings()
        new_embeddings = nn.Embedding(new_num_tokens, self.config.hidden_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

        # numbers of tokens to copy
        n = min(old_embeddings.weight.shape[0], new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        self.set_input_embeddings(new_embeddings)
        self.config.vocab_size = new_num_tokens
    
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
    ) -> Any:
        
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
                # Convert to causal mask with bounds checking
                if seq_length <= 0:
                    raise ValueError(f"Invalid sequence length: {seq_length}")
                
                # Ensure device consistency
                device = inputs_embeds.device
                causal_mask = torch.tril(torch.ones(seq_length, seq_length, device=device)).bool()
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) * causal_mask.unsqueeze(0)
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            else:
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_cache = () if use_cache else None
        
        for i, (layer_module, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_cache = next_cache + (layer_outputs[-1],)
            
            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)
        
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }


class LinearAttention(nn.Module):
    """Gated Linear Attention (GLA) variant."""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.g_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = hidden_states
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        g = torch.sigmoid(self.g_proj(x)).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Simple linear attention: Q * (K^T * V)
        kv = torch.einsum("b h s d, b h s e -> b h d e", k, v)
        output = torch.einsum("b h s d, b h d e -> b h s e", q, kv)

        output = self.o_proj((output * g).transpose(1, 2).reshape(batch_size, seq_len, -1))
        return output, None, None