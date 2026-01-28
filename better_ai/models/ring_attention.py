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
        
        # Apply rotary embeddings
        query_states, key_states = self.rotary_emb(query_states, key_states)
        
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
        if self.world_size > 1 and seq_len > self.block_size:
            attn_output, attn_weights = self.ring_attention_forward(
                query_states, key_states, value_states, attention_mask
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
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        
        # Split sequence into blocks
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        blocks_per_device = (num_blocks + self.world_size - 1) // self.world_size
        
        # Initialize output
        output = torch.zeros_like(query_states)
        attn_weights = None
        
        # Process blocks in ring fashion
        for block_idx in range(num_blocks):
            # Determine which device processes this block
            device_rank = block_idx % self.world_size
            
            if device_rank == self.rank:
                # This device processes the current block
                start_idx = block_idx * self.block_size
                end_idx = min(start_idx + self.block_size, seq_len)
                
                # Extract current block
                q_block = query_states[:, :, start_idx:end_idx, :]
                
                # Compute attention with all key/value blocks
                block_output, block_attn = self.compute_block_attention(
                    q_block, key_states, value_states, start_idx, end_idx, attention_mask
                )
                
                output[:, :, start_idx:end_idx, :] = block_output
                
                if output_attentions and block_attn is not None:
                    if attn_weights is None:
                        attn_weights = torch.zeros(
                            batch_size, num_heads, seq_len, seq_len,
                            device=query_states.device, dtype=query_states.dtype
                        )
                    attn_weights[:, :, start_idx:end_idx, :] = block_attn
            
            # Ring communication: send processed block to next device
            if self.world_size > 1:
                self.ring_communicate(block_idx, output, attn_weights)
        
        return output, attn_weights
    
    def compute_block_attention(
        self,
        query_block: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        start_idx: int,
        end_idx: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, num_heads, block_len, head_dim = query_block.shape
        _, _, kv_seq_len, _ = key_states.shape
        
        # Compute attention scores
        attn_scores = torch.matmul(
            query_block.transpose(1, 2),  # [B, block_len, num_heads, head_dim]
            key_states.permute(0, 2, 3, 1)  # [B, num_heads, head_dim, kv_seq_len]
        ) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_scores = attn_scores + attention_mask[:, :, start_idx:end_idx, :]
        
        # Apply softmax and attention dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states.permute(0, 2, 1, 3))
        attn_output = attn_output.transpose(1, 2)  # [B, block_len, num_heads, head_dim]
        
        return attn_output, attn_weights
    
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
        block_idx: int,
        output: torch.Tensor,
        attn_weights: Optional[torch.Tensor]
    ):
        """Ring communication for block processing"""
        if not dist.is_initialized():
            return
        
        # Send to next device, receive from previous device
        send_rank = (self.rank + 1) % self.world_size
        recv_rank = (self.rank - 1) % self.world_size
        
        # Non-blocking communication
        send_req = dist.isend(output, send_rank)
        recv_req = dist.irecv(output, recv_rank)
        
        # Wait for completion
        send_req.wait()
        recv_req.wait()


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
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if seq_len is None:
            seq_len = query_states.size(-2)
        
        # Use cached cos/sin if available
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        
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
        
        # Get the dimensions
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        # Ensure cos and sin have the right shape with bounds checking
        cos_seq_len = min(seq_len, cos.size(2))
        cos_head_dim = min(head_dim // 2, cos.size(3))
        sin_seq_len = min(seq_len, sin.size(2))
        sin_head_dim = min(head_dim // 2, sin.size(3))
        
        cos = cos[:, :, :cos_seq_len, :cos_head_dim]
        sin = sin[:, :, :sin_seq_len, :sin_head_dim]
        
        # Split into real and imaginary parts
        x_real = x[..., :head_dim // 2]
        x_imag = x[..., head_dim // 2:]
        
        # Apply rotation with proper broadcasting
        x_rot_real = x_real * cos - x_imag * sin
        x_rot_imag = x_real * sin + x_imag * cos
        
        # Concatenate back
        x_rot = torch.cat([x_rot_real, x_rot_imag], dim=-1)
        
        return x_rot
