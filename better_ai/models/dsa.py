"""DeepSeek Sparse Attention (DSA) implementation with Lightning Indexer"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class LightningIndexer(nn.Module):
    """
    Lightning Indexer for DSA - computes relevance scores between tokens
    Based on DeepSeek's implementation for efficient token selection
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_indexer_heads: int = 8,
        max_selected_tokens: int = 2048,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_indexer_heads = num_indexer_heads
        self.max_selected_tokens = max_selected_tokens
        self.temperature = temperature
        
        # Indexer projections (small heads for efficiency)
        self.indexer_dim = hidden_size // num_indexer_heads
        self.q_indexer = nn.Linear(hidden_size, num_indexer_heads * self.indexer_dim, bias=False)
        self.k_indexer = nn.Linear(hidden_size, num_indexer_heads * self.indexer_dim, bias=False)
        
        # Head weights for combining scores
        self.head_weights = nn.Parameter(torch.ones(num_indexer_heads))
        
        # Learnable temperature
        self.temperature_param = nn.Parameter(torch.tensor(temperature))
    
    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute relevance scores for token selection
        
        Args:
            query_states: (batch_size, seq_len, hidden_size)
            key_states: (batch_size, seq_len, hidden_size)
            attention_mask: optional mask
            
        Returns:
            selected_mask: (batch_size, seq_len) boolean mask
            relevance_scores: (batch_size, seq_len) relevance scores
        """
        batch_size, seq_len, _ = query_states.shape
        device = query_states.device
        
        # Project to indexer space
        q_proj = self.q_indexer(query_states)  # (batch, seq_len, num_heads * indexer_dim)
        k_proj = self.k_indexer(key_states)
        
        # Reshape for multi-head computation
        q_proj = q_proj.view(batch_size, seq_len, self.num_indexer_heads, self.indexer_dim)
        k_proj = k_proj.view(batch_size, seq_len, self.num_indexer_heads, self.indexer_dim)
        
        # Compute relevance scores per head
        # Formula: I_{t,s} = sum_{j=1}^{H^I} w_{t,j} * ReLU(q_{t,j} · k_{s})
        relevance_scores = torch.zeros(batch_size, seq_len, device=device)
        
        for head_idx in range(self.num_indexer_heads):
            q_head = q_proj[:, :, head_idx, :]  # (batch, seq_len, indexer_dim)
            k_head = k_proj[:, :, head_idx, :]  # (batch, seq_len, indexer_dim)
            
            # Compute dot products: q · k for all token pairs
            # For efficiency, we compute similarity of each token with all previous tokens
            dot_products = torch.bmm(q_head, k_head.transpose(-2, -1))  # (batch, seq_len, seq_len)
            
            # Apply ReLU to introduce sparsity
            dot_products = F.relu(dot_products)
            
            # Weight by head importance
            weighted_scores = dot_products * self.head_weights[head_idx] * torch.exp(-self.temperature_param)
            
            # Aggregate across heads (sum over sequence dimension for current token)
            # Each token attends to previous tokens, so we take the row corresponding to each token
            token_scores = torch.sum(weighted_scores, dim=-1)  # (batch, seq_len)
            relevance_scores += token_scores
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                relevance_scores = relevance_scores.masked_fill(~attention_mask.bool(), -float('inf'))
        
        # Select top-k tokens
        selected_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        for batch_idx in range(batch_size):
            batch_scores = relevance_scores[batch_idx]
            
            # Get top-k indices (excluding padding tokens)
            if attention_mask is not None and attention_mask.dim() == 2:
                valid_mask = attention_mask[batch_idx].bool()
                if valid_mask.any():
                    valid_scores = batch_scores.masked_fill(~valid_mask, -float('inf'))
                    top_k = min(self.max_selected_tokens, valid_mask.sum().item())
                    _, top_indices = torch.topk(valid_scores, top_k, largest=True)
                else:
                    top_indices = torch.tensor([], device=device, dtype=torch.long)
            else:
                top_k = min(self.max_selected_tokens, seq_len)
                _, top_indices = torch.topk(batch_scores, top_k, largest=True)
            
            selected_mask[batch_idx, top_indices] = True
        
        return selected_mask, relevance_scores


class DeepSeekSparseAttention(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) with Lightning Indexer
    Reduces O(L²) complexity to O(Lk) where k << L
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_selected_tokens: int = 2048,
        num_indexer_heads: int = 8,
        indexer_temperature: float = 1.0,
        local_window_size: int = 512,
        use_flash_attention: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_selected_tokens = max_selected_tokens
        self.local_window_size = local_window_size
        self.use_flash_attention = use_flash_attention
        
        # Lightning indexer for token selection
        self.lightning_indexer = LightningIndexer(
            hidden_size=hidden_size,
            num_indexer_heads=num_indexer_heads,
            max_selected_tokens=max_selected_tokens,
            temperature=indexer_temperature
        )
        
        # Standard attention projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        # Dropout
        self.attention_dropout = nn.Dropout(dropout)
        
        # For handling GQA
        self.num_key_value_groups = num_heads // num_key_value_heads
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int, num_heads: int) -> torch.Tensor:
        """Reshape tensor for attention computation"""
        return tensor.view(bsz, seq_len, num_heads, self.head_dim).transpose(1, 2)
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value heads to match query heads"""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    def _create_combined_mask(
        self,
        selected_mask: torch.Tensor,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create attention mask that combines local window and selected tokens
        """
        batch_size = selected_mask.size(0)
        
        # Start with causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        
        # Add local window (ensure recent tokens are always included)
        local_mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - self.local_window_size)
            end = min(seq_len, i + 1)  # Current token + previous local window
            local_mask[i, start:end] = True
        
        # Combine local and selected masks
        combined_mask = causal_mask & (local_mask | selected_mask.unsqueeze(1))
        
        return combined_mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device
        
        # Compute projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Use lightning indexer to select important tokens
        selected_mask, relevance_scores = self.lightning_indexer(
            query_states, key_states, attention_mask
        )
        
        # Create sparse attention mask
        sparse_mask = self._create_combined_mask(selected_mask, q_len, device)
        
        # Reshape for attention computation
        query_states = self._shape(query_states, q_len, bsz, self.num_heads)
        key_states = self._shape(key_states, q_len, bsz, self.num_key_value_heads)
        value_states = self._shape(value_states, q_len, bsz, self.num_key_value_heads)
        
        # Handle GQA
        if self.num_key_value_groups > 1:
            key_states = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention with sparse mask
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Convert sparse mask to attention mask format
            attention_mask_full = (~sparse_mask).float() * -1e9
            
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask_full,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
                is_causal=False  # Causal is handled in our mask
            )
            attn_weights = None
        else:
            # Standard attention computation with sparse mask
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply sparse mask
            attn_weights = attn_weights.masked_fill(~sparse_mask, -float('inf'))
            
            # Apply additional attention mask if provided
            if attention_mask is not None and attention_mask.dim() == 3:
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


class OptimizedTransformerBlockWithDSA(nn.Module):
    """Optimized transformer block with DSA and MLA"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        use_dsa: bool = True,
        use_mla: bool = True,
        dsa_config: Optional[Dict] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Default DSA config
        if dsa_config is None:
            dsa_config = {
                'max_selected_tokens': 2048,
                'num_indexer_heads': 8,
                'indexer_temperature': 1.0,
                'local_window_size': 512,
                'use_flash_attention': True
            }
        
        # Attention mechanism
        if use_dsa:
            self.self_attn = DeepSeekSparseAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                **dsa_config,
                dropout=dropout
            )
        elif use_mla:
            from .attention_optimized import MultiHeadLatentAttention
            self.self_attn = MultiHeadLatentAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                compression_ratio=4.0,
                norm_eps=norm_eps,
                dropout=dropout,
                use_flash_attention=True
            )
        else:
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
        
        # Feed-forward
        from .core import SwiGLU
        self.mlp = SwiGLU(hidden_size, intermediate_size)
        
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # DSA doesn't support KV cache yet
        if hasattr(self.self_attn, 'forward'):
            hidden_states, self_attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions
            )
            present_key_value = None
        else:
            # Fallback to standard attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
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
        
        if use_cache and present_key_value is not None:
            outputs += (present_key_value,)
        
        return outputs


# Export classes
__all__ = ['LightningIndexer', 'DeepSeekSparseAttention', 'OptimizedTransformerBlockWithDSA']