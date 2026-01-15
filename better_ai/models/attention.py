"""Flash Attention, Ring Attention, and GQA optimizations for DeepSeek model"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
import warnings
from .core import RMSNorm, SwiGLU
from .ring_attention import RingAttention, RotaryEmbedding


def flash_attention_forward(query, key, value, dropout_p=0.0, softmax_scale=None, causal=True):
    """
    Flash Attention forward pass using PyTorch's built-in flash attention
    Fallback to standard attention if flash attention is not available
    """
    try:
        # Try to use PyTorch's built-in flash attention
        if hasattr(F, 'scaled_dot_product_attention'):
            return F.scaled_dot_product_attention(
                query, key, value,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=softmax_scale
            )
        else:
            raise ImportError("Flash attention not available")
    except ImportError:
        # Fallback to standard attention
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * softmax_scale
        
        # Apply causal mask
        if causal:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, value)
        
        return output


class FlashMultiHeadAttention(nn.Module):
    """Multi-Head Attention with Flash Attention support and GQA"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
        use_gqa: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.use_flash_attention = use_flash_attention
        self.use_gqa = use_gqa
        
        if use_gqa:
            self.num_key_value_groups = num_heads // num_key_value_heads
        else:
            self.num_key_value_groups = 1
        
        if head_dim * num_heads != hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads")
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        # Dropout
        self.attention_dropout = nn.Dropout(dropout)
        
        # For Flash Attention optimization
        self.softmax_scale = None
        
        # KV cache
        self.kv_cache = None
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """Reshape tensor for attention computation"""
        return tensor.view(bsz, seq_len, -1, self.head_dim).transpose(1, 2)
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
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
        query_states = self._shape(query_states, q_len, bsz)  # (bsz, num_heads, q_len, head_dim)
        key_states = self._shape(key_states, q_len, bsz)      # (bsz, num_kv_heads, q_len, head_dim)
        value_states = self._shape(value_states, q_len, bsz)  # (bsz, num_kv_heads, q_len, head_dim)
        
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
        if self.use_gqa and self.num_key_value_groups > 1:
            key_states = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention
        if self.use_flash_attention:
            # Use Flash Attention
            dropout_p = self.attention_dropout.p if self.training else 0.0
            attn_output = flash_attention_forward(
                query_states, key_states, value_states,
                dropout_p=dropout_p,
                softmax_scale=self.softmax_scale,
                causal=True
            )
        else:
            # Standard attention
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
            
            if self.softmax_scale is not None:
                attn_weights = attn_weights * self.softmax_scale
            else:
                attn_weights = attn_weights / math.sqrt(self.head_dim)
            
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


class SparseAttention(nn.Module):
    """Sparse attention pattern implementation"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        local_window_size: int = 1024,
        global_stride: int = 512,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.local_window_size = local_window_size
        self.global_stride = global_stride
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        # Dropout
        self.attention_dropout = nn.Dropout(dropout)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """Reshape tensor for attention computation"""
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def _create_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create sparse attention mask"""
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        
        # Local attention window
        for i in range(seq_len):
            start = max(0, i - self.local_window_size // 2)
            end = min(seq_len, i + self.local_window_size // 2 + 1)
            mask[i, start:end] = True
        
        # Global attention tokens
        if self.global_stride > 0:
            global_tokens = torch.arange(0, seq_len, self.global_stride, device=device)
            mask[:, global_tokens] = True
        
        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        mask = mask & causal_mask
        
        return mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention
        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)
        
        # Create sparse mask
        sparse_mask = self._create_sparse_mask(q_len, hidden_states.device)
        
        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply sparse mask
        attn_weights = attn_weights.masked_fill(~sparse_mask, float('-inf'))
        
        if attention_mask is not None:
            attn_weights += attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights


class LatentAttention(nn.Module):
    """Multi-Head Latent Attention (MLA) for KV cache compression"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        latent_dim: Optional[int] = None,
        compression_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        if latent_dim is None:
            latent_dim = int(head_dim / compression_ratio)
        
        self.latent_dim = latent_dim
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_down_proj = nn.Linear(hidden_size, num_heads * latent_dim, bias=False)
        self.v_down_proj = nn.Linear(hidden_size, num_heads * latent_dim, bias=False)
        self.k_up_proj = nn.Linear(num_heads * latent_dim, num_heads * head_dim, bias=False)
        self.v_up_proj = nn.Linear(num_heads * latent_dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        # Dropout
        self.attention_dropout = nn.Dropout(dropout)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int, is_latent: bool = False):
        """Reshape tensor for attention computation"""
        if is_latent:
            return tensor.view(bsz, seq_len, self.num_heads, self.latent_dim).transpose(1, 2)
        else:
            return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # Projections
        query_states = self.q_proj(hidden_states)
        key_latent = self.k_down_proj(hidden_states)
        value_latent = self.v_down_proj(hidden_states)
        
        # Reshape
        query_states = self._shape(query_states, q_len, bsz)
        key_latent = self._shape(key_latent, q_len, bsz, is_latent=True)
        value_latent = self._shape(value_latent, q_len, bsz, is_latent=True)
        
        # Compute attention with compressed KV
        attn_weights = torch.matmul(query_states, key_latent.transpose(-2, -1)) / math.sqrt(self.latent_dim)
        
        if attention_mask is not None:
            attn_weights += attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention to compressed values
        attn_output_latent = torch.matmul(attn_weights, value_latent)
        
        # Upsample back to original dimension
        attn_output = self.k_up_proj(attn_output_latent.transpose(1, 2).contiguous().view(bsz, q_len, -1))
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights


class OptimizedTransformerBlock(nn.Module):
    """Optimized transformer block with Flash Attention and GQA"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
        use_gqa: bool = True,
        use_sparse_attention: bool = False,
        use_latent_attention: bool = False,
        local_window_size: int = 1024,
        global_stride: int = 512,
        latent_dim: Optional[int] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Attention mechanism selection
        if use_sparse_attention:
            self.self_attn = SparseAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                local_window_size=local_window_size,
                global_stride=global_stride,
                dropout=dropout
            )
        elif use_latent_attention:
            self.self_attn = LatentAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                latent_dim=latent_dim,
                dropout=dropout
            )
        else:
            self.self_attn = FlashMultiHeadAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                dropout=dropout,
                use_flash_attention=use_flash_attention,
                use_gqa=use_gqa
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
        
        if hasattr(self.self_attn, 'forward'):  # Regular attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        else:  # Sparse or latent attention (no KV cache support)
            hidden_states, self_attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            present_key_value = None
        
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


def get_optimized_attention_config(
    model_size: str = "small",
    use_flash_attention: bool = True,
    use_gqa: bool = True,
    use_sparse: bool = False,
    use_latent: bool = False
) -> dict:
    """Get optimized attention configuration based on model size and requirements"""
    
    if model_size == "small":
        return {
            "use_flash_attention": use_flash_attention,
            "use_gqa": use_gqa,
            "use_sparse_attention": use_sparse,
            "use_latent_attention": use_latent,
            "local_window_size": 512,
            "global_stride": 256,
            "latent_dim": 64 if use_latent else None
        }
    elif model_size == "medium":
        return {
            "use_flash_attention": use_flash_attention,
            "use_gqa": use_gqa,
            "use_sparse_attention": use_sparse,
            "use_latent_attention": use_latent,
            "local_window_size": 1024,
            "global_stride": 512,
            "latent_dim": 128 if use_latent else None
        }
    else:  # large
        return {
            "use_flash_attention": use_flash_attention,
            "use_gqa": use_gqa,
            "use_sparse_attention": use_sparse,
            "use_latent_attention": use_latent,
            "local_window_size": 2048,
            "global_stride": 1024,
            "latent_dim": 256 if use_latent else None
        }