import torch
import unittest
from better_ai.models.reward_model import BranchRewardModel
from better_ai.models.moe import ExpertRouter
from better_ai.models.core import MultiHeadAttention, LinearAttention
from better_ai.models.enhanced_model import EnhancedDeepSeekModel
from better_ai.config import ModelConfig
from transformers import AutoTokenizer

class TestModelEnhancements(unittest.TestCase):
    """Unit tests for various model enhancement components."""

    def test_multi_turn_reward_model(self):
        """Test the forward pass of the BranchRewardModel."""
        config = ModelConfig()
        model = BranchRewardModel(config, hidden_dim=512)
        hidden_states = torch.randn(1, 10, config.hidden_dim)
        attention_mask = torch.ones(1, 10)
        reward = model(hidden_states, attention_mask)
        self.assertEqual(reward.shape, (1,))

    def test_pre_moe_router(self):
        """Test the ExpertRouter's output shapes."""
        router = ExpertRouter(hidden_size=512, num_experts=8, pre_router_dim=128)
        hidden_states = torch.randn(1, 10, 512)
        routing_weights, selected_experts, router_logits = router(hidden_states)
        self.assertEqual(routing_weights.shape, (1, 10, 2))
        self.assertEqual(selected_experts.shape, (1, 10, 2))

    def test_hybrid_attention(self):
        """Test both RoPE and NoPE attention mechanisms."""
        # Test with RoPE
        attention_rope = MultiHeadAttention(hidden_size=512, num_heads=8, num_key_value_heads=4, head_dim=64, use_nope=False)
        hidden_states = torch.randn(1, 10, 512)
        output_rope, _, _ = attention_rope(hidden_states)
        self.assertEqual(output_rope.shape, (1, 10, 512))

        # Test with NoPE
        attention_nope = MultiHeadAttention(hidden_size=512, num_heads=8, num_key_value_heads=4, head_dim=64, use_nope=True)
        output_nope, _, _ = attention_nope(hidden_states)
        self.assertEqual(output_nope.shape, (1, 10, 512))

    def test_qk_normalization(self):
        """Test the forward pass of MultiHeadAttention with QK normalization."""
        attention = MultiHeadAttention(hidden_size=512, num_heads=8, num_key_value_heads=4, head_dim=64)
        hidden_states = torch.randn(1, 10, 512)
        output, _, _ = attention(hidden_states)
        self.assertEqual(output.shape, (1, 10, 512))

    def test_self_correction(self):
        """Test the self-correction mechanism of the EnhancedDeepSeekModel."""
        config = ModelConfig()
        model = EnhancedDeepSeekModel(config)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Create a dummy input that will trigger a correction
        input_text = "This is an error."
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids

        # Run self-correction
        final_response, corrected = model.self_correct(input_ids, tokenizer, verification_keyword="error")

        # Assert that correction was performed and the response is not empty
        self.assertTrue(corrected)
        self.assertIsInstance(final_response, str)
        self.assertGreater(len(final_response), 0)

    def test_linear_attention(self):
        """Test the integration and forward pass of the LinearAttention module."""
        config = ModelConfig(use_linear_attention=True, use_ring_attention=False)
        model = EnhancedDeepSeekModel(config)

        # Check if the attention layer is replaced
        self.assertIsInstance(model.model.layers[0].self_attn, LinearAttention)

        # Test forward pass
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        outputs = model(input_ids)
        self.assertEqual(outputs["logits"].shape, (1, 10, config.vocab_size))

if __name__ == '__main__':
    unittest.main()
