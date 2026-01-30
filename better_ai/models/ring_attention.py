"""
Ring Attention implementation for near-infinite context processing
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import torch.distributed as dist


class RingAttention(nn.Module):
    """
    Ring Attention mechanism for distributed context processing
    Splits attention computation across devices in a ring topology
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        block_size: int = 1024,
        dropout: float = 0.0,
        use_flash: bool = True,
        rope_theta: float = 10000.0,
        max_seq_len: int = 8192,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.head_dim = head_dim or (hidden_dim // num_heads)
        self.block_size = block_size
        self.dropout = dropout
        self.use_flash = use_flash
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ensure dimensions are compatible
        assert hidden_dim == num_heads * self.head_dim, "hidden_dim must be num_heads * head_dim"
        assert num_heads % num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads"
        
        self.num_groups = num_heads // num_key_value_heads
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)
        
        # Dropout
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len
        )
        
        # Ring communication setup
        self.rank = 0
        self.world_size = 1
        self.setup_ring_communication()
    
    def repeat_kv(self, x: torch.Tensor, num_rep: int) -> torch.Tensor:
        """Repeat KV heads to match query heads (grouped-query attention)"""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        if num_rep == 1:
            return x
        return x[:, :, None, :, :].expand(batch, num_kv_heads, num_rep, seq_len, head_dim).reshape(batch, num_kv_heads * num_rep, seq_len, head_dim)
        
    def setup_ring_communication(self):
        """Setup distributed ring communication"""
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings, handle offset if past_key_value is present
        offset = past_key_value[0].size(2) if past_key_value is not None else 0
        query_states, key_states = self.rotary_emb(query_states, key_states, offset=offset)
        
        # Expand KV heads for grouped-query attention if needed
        if self.num_key_value_heads != self.num_heads:
            key_states = self.repeat_kv(key_states, self.num_heads // self.num_key_value_heads)
            value_states = self.repeat_kv(value_states, self.num_heads // self.num_key_value_heads)
        
        # Handle past key values
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        # Ring attention computation
        if self.world_size > 1:
            attn_output, attn_weights = self.ring_attention_forward(
                query_states, key_states, value_states, attention_mask, output_attentions
            )
        else:
            # Standard attention for small sequences or single device
            attn_output, attn_weights = self.standard_attention_forward(
                query_states, key_states, value_states, attention_mask
            )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        attn_output = self.o_proj(attn_output)
        attn_output = self.output_dropout(attn_output)
        
        # Cache for future use
        past_key_value = (key_states, value_states) if use_cache else None
        
        return attn_output, past_key_value, attn_weights
    
    def ring_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        
        # Local Q, K, V
        local_q = query_states
        curr_k = key_states
        curr_v = value_states
        
        # Online softmax accumulation buffers
        # out: [B, H, S, D], lse: [B, H, S, 1]
        out = torch.zeros_like(local_q)
        lse = torch.full((batch_size, num_heads, seq_len, 1), -float('inf'), device=local_q.device, dtype=torch.float32)
        
        for step in range(self.world_size):
            # Compute attention scores: [B, H, S_q, S_kv]
            attn_scores = torch.matmul(local_q, curr_k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                # Proper slicing of attention mask for distributed shards
                # attention_mask: [B, 1, S_q, S_total]
                kv_offset = step * seq_len
                mask_slice = attention_mask[..., kv_offset : kv_offset + seq_len]
                attn_scores = attn_scores + mask_slice

            # Online softmax update logic
            # mi: max of current block
            mi = torch.max(attn_scores, dim=-1, keepdim=True).values
            # Pi: unnormalized attention weights for current block
            Pi = torch.exp(attn_scores - mi)
            # Li: sum of weights for current block
            Li = torch.sum(Pi, dim=-1, keepdim=True)

            # Update global LSE and Output
            new_lse = torch.maximum(lse, mi) + torch.log(
                torch.exp(lse - torch.maximum(lse, mi)) +
                torch.exp(mi - torch.maximum(lse, mi)) * Li
            )
            
            # Rescale previous output and add new block contribution
            out = out * torch.exp(lse - new_lse) + torch.matmul(Pi, curr_v) * torch.exp(mi - new_lse)
            lse = new_lse
            
            # Circulate K, V around the ring
            if self.world_size > 1:
                curr_k, curr_v = self.ring_communicate(curr_k, curr_v)
            else:
                break

        return out, None
    
    def standard_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # Use flash attention if available and enabled
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )
            attn_weights = None
        else:
            # Standard attention computation
            attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attention_dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, value_states)
        
        return attn_output, attn_weights
    
    def ring_communicate(
        self,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ring communication: send K,V to next device, receive from previous"""
        if self.world_size <= 1:
            return k, v

        next_rank = (self.rank + 1) % self.world_size
        prev_rank = (self.rank - 1) % self.world_size
        
        new_k = torch.empty_like(k)
        new_v = torch.empty_like(v)
        
        # Use send/recv for ring communication
        # In production, use dist.batch_isend_irecv for better performance
        ops = [
            dist.P2POp(dist.isend, k, next_rank),
            dist.P2POp(dist.isend, v, next_rank),
            dist.P2POp(dist.irecv, new_k, prev_rank),
            dist.P2POp(dist.irecv, new_v, prev_rank),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        return new_k, new_v


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    
    def __init__(self, head_dim: int, rope_theta: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        
        # Create rotary embeddings
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for max sequence length
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        self.register_buffer('cos_cached', emb[:, :head_dim // 2])
        self.register_buffer('sin_cached', emb[:, head_dim // 2:])
    
    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        seq_len: Optional[int] = None,
        offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if seq_len is None:
            seq_len = query_states.size(-2)
        
        # Use cached cos/sin if available, handle offset for KV cache
        # Ensure we don't go out of bounds
        cos = self.cos_cached[offset:offset+seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset:offset+seq_len].unsqueeze(0).unsqueeze(0)
        
        # Apply rotary embeddings
        query_states = self.apply_rotary_pos_emb(query_states, cos, sin)
        key_states = self.apply_rotary_pos_emb(key_states, cos, sin)
        
        return query_states, key_states
    
    def apply_rotary_pos_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        
        # Split into real and imaginary parts
        x_real = x[..., :self.head_dim // 2]
        x_imag = x[..., self.head_dim // 2:]
        
        # Apply rotation
        x_rot_real = x_real * cos - x_imag * sin
        x_rot_imag = x_real * sin + x_imag * cos
        
        # Concatenate back
        x_rot = torch.cat([x_rot_real, x_rot_imag], dim=-1)
        
        return x_rot


class StripedAttention(RingAttention):
    """
    Striped Attention: A faster variant of Ring Attention for causal models.
    Distributes tokens uniformly throughout the sequence (interleaved) to balance
    the triangular causal computation workload across devices.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:

        batch_size, seq_len, _ = hidden_states.shape

        # If single device, Striped Attention is equivalent to standard attention
        if self.world_size <= 1:
            return super().forward(
                hidden_states, attention_mask, past_key_value, use_cache, output_attentions
            )

        # In truly distributed Striped Attention, each rank i only holds tokens i, i+world_size...
        # Here we assume hidden_states passed to this rank is already sharded OR we shard it.
        # For simplicity in this implementation, we assume we receive the full hidden_states
        # and we pick our stripe.

        indices = torch.arange(seq_len, device=hidden_states.device)
        my_indices = indices[self.rank::self.world_size]

        # Local Q, K, V
        striped_hidden = hidden_states[:, my_indices, :]

        query_states = self.q_proj(striped_hidden)
        key_states = self.k_proj(striped_hidden)
        value_states = self.v_proj(striped_hidden)

        # Reshape
        query_states = query_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE with CORRECT global offsets
        # Each token in my_indices has its own offset
        # We need a variant of rotary_emb that takes a tensor of offsets
        query_states, key_states = self.rotary_emb_striped(query_states, key_states, my_indices)

        # Expand KV heads
        if self.num_key_value_heads != self.num_heads:
            key_states = self.repeat_kv(key_states, self.num_heads // self.num_key_value_heads)
            value_states = self.repeat_kv(value_states, self.num_heads // self.num_key_value_heads)

        # Distributed Ring forward pass
        attn_output, _ = self.ring_attention_forward(
            query_states, key_states, value_states, attention_mask, output_attentions
        )

        # Project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.hidden_dim)
        attn_output = self.o_proj(attn_output)

        # Now we have local outputs for our stripe.
        # We need to gather them back to match the input shape if needed,
        # or keep them sharded depending on the training strategy.
        # For now, let's gather to return full output
        full_output = torch.zeros(batch_size, seq_len, self.hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)

        # Use all_gather to collect all striped outputs
        all_striped_outputs = [torch.zeros_like(attn_output) for _ in range(self.world_size)]
        dist.all_gather(all_striped_outputs, attn_output)

        # Place them in the right positions
        for r, out in enumerate(all_striped_outputs):
            full_output[:, r::self.world_size, :] = out

        full_output = self.output_dropout(full_output)

        past_key_value = (key_states, value_states) if use_cache else None

        return full_output, past_key_value, None

    def rotary_emb_striped(self, q, k, indices):
        """Apply RoPE with per-token indices"""
        # q, k: [B, H, S_local, D]
        # indices: [S_local]

        cos = self.rotary_emb.cos_cached[indices].unsqueeze(0).unsqueeze(0) # [1, 1, S_local, D/2]
        sin = self.rotary_emb.sin_cached[indices].unsqueeze(0).unsqueeze(0)

        q_rot = self.rotary_emb.apply_rotary_pos_emb(q, cos, sin)
        k_rot = self.rotary_emb.apply_rotary_pos_emb(k, cos, sin)

        return q_rot, k_rot
