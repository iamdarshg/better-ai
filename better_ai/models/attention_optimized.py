"""Multi-Head Latent Attention (MLA) implementation for KV cache compression"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) as implemented in DeepSeek
    Compresses KV cache to reduce memory requirements for long sequences
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        compression_ratio: float = 4.0,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        use_flash_attention: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.compression_ratio = compression_ratio
        self.use_flash_attention = use_flash_attention
        
        # Latent dimensions (compressed space)
        self.latent_dim = int(head_dim / compression_ratio)
        
        # Check compatibility
        if head_dim * num_heads != hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads")
        
        # Query projections (uncompressed for better quality)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        
        # Key/Value compression projections
        self.k_down_proj = nn.Linear(hidden_size, num_key_value_heads * self.latent_dim, bias=False)
        self.v_down_proj = nn.Linear(hidden_size, num_key_value_heads * self.latent_dim, bias=False)
        
        # Decompression projections (for compressed KV)
        self.k_up_proj = nn.Linear(num_key_value_heads * self.latent_dim, num_key_value_heads * head_dim, bias=False)
        self.v_up_proj = nn.Linear(num_key_value_heads * self.latent_dim, num_key_value_heads * head_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        # Normalization for stability (applied after reshaping)
        # self.k_norm = nn.LayerNorm(self.latent_dim, eps=norm_eps)
        # self.v_norm = nn.LayerNorm(self.latent_dim, eps=norm_eps)
        
        # Dropout
        self.attention_dropout = nn.Dropout(dropout)
        
        # Cache for compressed KV
        self.register_buffer('compressed_kv_cache', None)
        self.cache_initialized = False
    
    def _shape_queries(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        """Reshape query tensor for attention computation"""
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def _shape_latent_kv(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        """Reshape compressed KV tensor for attention computation"""
        return tensor.view(bsz, seq_len, self.num_key_value_heads, self.latent_dim).transpose(1, 2)
    
    def _upsample_kv(self, latent_kv: torch.Tensor, bsz: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Upsample compressed KV back to original dimension"""
        # Reshape latent KV
        latent_kv = latent_kv.view(bsz, seq_len, -1)  # (bsz, seq_len, num_kv_heads * latent_dim)
        
        # Split into key and value components
        kv_split = latent_kv.chunk(2, dim=-1)
        k_compressed = kv_split[0]
        v_compressed = kv_split[1]
        
        # Decompress
        k_compressed = k_compressed.view(bsz, seq_len, self.num_key_value_heads, self.latent_dim)
        v_compressed = v_compressed.view(bsz, seq_len, self.num_key_value_heads, self.latent_dim)
        
        # Upsample to full dimension
        k_up = self.k_up_proj(k_compressed.view(bsz, seq_len, -1))
        v_up = self.v_up_proj(v_compressed.view(bsz, seq_len, -1))
        
        # Reshape for attention
        k_up = k_up.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v_up = v_up.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        return k_up, v_up
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value heads to match query heads (for GQA)"""
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
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # Compute queries (uncompressed)
        query_states = self.q_proj(hidden_states)
        query_states = self._shape_queries(query_states, q_len, bsz)
        
        # Compute and compress keys/values
        key_compressed = self.k_down_proj(hidden_states)
        value_compressed = self.v_down_proj(hidden_states)
        
        # Normalization will be applied later if needed
        # key_compressed = self.k_norm(key_compressed)
        # value_compressed = self.v_norm(value_compressed)
        
        # Shape compressed KV
        key_compressed = self._shape_latent_kv(key_compressed, q_len, bsz)
        value_compressed = self._shape_latent_kv(value_compressed, q_len, bsz)
        
        # Handle caching with compressed KV
        if use_cache:
            if past_key_value is not None:
                # Concatenate with cached compressed KV
                past_key_compressed, past_value_compressed = past_key_value
                key_compressed = torch.cat([past_key_compressed, key_compressed], dim=2)
                value_compressed = torch.cat([past_value_compressed, value_compressed], dim=2)
            
            # Cache compressed KV
            present = (key_compressed, value_compressed)
        else:
            present = None
        
        # Upsample KV for attention computation
        key_upsampled, value_upsampled = self._upsample_kv(
            torch.cat([key_compressed, value_compressed], dim=-1), bsz, key_compressed.size(2)
        )
        
        # Handle GQA (Grouped Query Attention)
        num_key_value_groups = self.num_heads // self.num_key_value_heads
        if num_key_value_groups > 1:
            key_upsampled = self._repeat_kv(key_upsampled, num_key_value_groups)
            value_upsampled = self._repeat_kv(value_upsampled, num_key_value_groups)
        
        # Compute attention
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use Flash Attention with upsampled KV
            attn_output = F.scaled_dot_product_attention(
                query_states, key_upsampled, value_upsampled,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
                is_causal=True
            )
            attn_weights = None
        else:
            # Standard attention computation
            attn_weights = torch.matmul(query_states, key_upsampled.transpose(-2, -1))
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights += attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attention_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, value_upsampled)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights, present


class MoEWithMLABlock(nn.Module):
    """MoE block that combines token-centric MoE with MLA attention"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        expert_intermediate_size: Optional[int] = None,
        compression_ratio: float = 4.0,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        use_mla: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_mla = use_mla
        
        # Attention with MLA
        if use_mla:
            self.self_attn = MultiHeadLatentAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                compression_ratio=compression_ratio,
                norm_eps=norm_eps,
                dropout=dropout,
                use_flash_attention=True
            )
        else:
            # Fallback to standard attention
            from .attention import FlashMultiHeadAttention
            self.self_attn = FlashMultiHeadAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                dropout=dropout,
                use_flash_attention=True,
                use_gqa=True
            )
        
        # Optimized MoE layer
        from .moe_optimized import OptimizedMoELayer
        self.moe = OptimizedMoELayer(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            expert_intermediate_size=expert_intermediate_size,
            dropout=dropout,
            capacity_factor=1.25,
            load_balance_loss_weight=0.01,
            shared_experts=1
        )
        
        # Normalization
        from .core import RMSNorm
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        
        # Self-attention with MLA
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
        
        # MoE layer
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


# Export classes
__all__ = ['MultiHeadLatentAttention', 'MoEWithMLABlock']