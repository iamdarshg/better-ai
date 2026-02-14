import unittest
import torch
import torch.nn.functional as F
import math
from better_ai.models.striped_attention import StripedAttention

class TestStripedAttentionRing(unittest.TestCase):
    def test_ring_matches_local(self):
        hidden_dim = 64
        num_heads = 4
        num_kv_heads = 2
        seq_len = 16
        batch_size = 1

        # Initialize StripedAttention
        model = StripedAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            max_seq_len=seq_len
        )
        model.eval()

        # Test data
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

        # 1. Compute locally (world_size = 1)
        model.world_size = 1
        model.rank = 0
        with torch.no_grad():
            out_local, _, _ = model(hidden_states)

        # 2. Simulate distributed (world_size = 2)
        model.world_size = 2

        # We need to simulate the "striped" view of the world.
        # Rank 0 should see tokens 0, 2, 4...
        # Rank 1 should see tokens 1, 3, 5...

        # Rank 0
        model.rank = 0
        # For the mock simulation to work correctly, we need to provide ALL K and V
        # because _ring_attention_optimized gathers them from full_k/full_v list.
        # But in a real striped forward pass, 'q', 'k', 'v' passed to it are LOCAL.

        # Let's extract local Q, K, V for each rank to simulate _ring_attention_forward input
        q_proj = model.q_proj(hidden_states).view(batch_size, seq_len, num_heads, -1).transpose(1, 2)
        k_proj = model.k_proj(hidden_states).view(batch_size, seq_len, num_kv_heads, -1).transpose(1, 2)
        v_proj = model.v_proj(hidden_states).view(batch_size, seq_len, num_kv_heads, -1).transpose(1, 2)

        # Apply RoPE
        indices = torch.arange(seq_len)
        cos, sin = model.rotary_emb.get_cos_sin(indices, device=hidden_states.device)
        q_rope = model.rotary_emb._apply_rotary_emb(q_proj, cos, sin)
        k_rope = model.rotary_emb._apply_rotary_emb(k_proj, cos, sin)

        q_rank0 = q_rope[:, :, 0::2, :]
        k_rank0 = k_rope[:, :, 0::2, :]
        v_rank0 = v_proj[:, :, 0::2, :]

        q_rank1 = q_rope[:, :, 1::2, :]
        k_rank1 = k_rope[:, :, 1::2, :]
        v_rank1 = v_proj[:, :, 1::2, :]

        full_k = [k_rank0, k_rank1]
        full_v = [v_rank0, v_rank1]

        model._test_full_k = full_k
        model._test_full_v = full_v

        model.rank = 0
        out_rank0, _, _ = model._ring_attention_optimized(q_rank0, full_k, full_v, hidden_states.device, hidden_states.dtype, None)

        model.rank = 1
        out_rank1, _, _ = model._ring_attention_optimized(q_rank1, full_k, full_v, hidden_states.device, hidden_states.dtype, None)

        # Combine
        out_combined = out_rank0 + out_rank1

        # Check if they match local output
        torch.testing.assert_close(out_combined, out_local, rtol=1e-4, atol=1e-4)

    def test_causal_mask(self):
        model = StripedAttention(hidden_dim=32, num_heads=4)
        q_len = 4
        k_len = 4
        world_size = 2

        inf = float("inf")
        # Rank 0: indices 0, 2, 4, 6
        # Rank 1: indices 1, 3, 5, 7

        # Query Rank 0 attending to Key Rank 0
        mask00 = model._get_striped_causal_mask(q_len, k_len, 0, 0, world_size, torch.device("cpu"), torch.float32)
        # Expected: standard causal mask
        # 0 <= 0 (T), 0 <= 2 (F) -> diag 0
        expected00 = torch.tensor([
            [0., -inf, -inf, -inf],
            [0., 0., -inf, -inf],
            [0., 0., 0., -inf],
            [0., 0., 0., 0.]
        ])

        # Query Rank 0 attending to Key Rank 1
        mask01 = model._get_striped_causal_mask(q_len, k_len, 0, 1, world_size, torch.device("cpu"), torch.float32)
        # k_local * 2 + 1 <= q_local * 2 + 0  => k_local <= q_local - 0.5 => k_local <= q_local - 1

        # Query Rank 1 attending to Key Rank 0
        mask10 = model._get_striped_causal_mask(q_len, k_len, 1, 0, world_size, torch.device("cpu"), torch.float32)
        # k_local * 2 + 0 <= q_local * 2 + 1  => k_local <= q_local + 0.5 => k_local <= q_local

        # Check some values
        self.assertEqual(mask01[0, 0], -inf) # q_idx 0 (global 0), k_idx 0 (global 1) -> F
        self.assertEqual(mask10[0, 0], 0.0)  # q_idx 0 (global 1), k_idx 0 (global 0) -> T

if __name__ == "__main__":
    unittest.main()
