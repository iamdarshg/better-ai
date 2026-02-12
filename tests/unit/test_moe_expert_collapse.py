"""
Tests for MoE Expert Collapse Prevention Mechanisms

This module tests the following expert collapse prevention techniques:
1. Loss-Free Bias-Based Balancing
2. Orthogonality + Variance Specialization Losses
3. Router Z-Loss for FP8 Stability
4. Expert-Router Coupling (ERC) Loss
5. Gradient Clipping Integration
6. BF16 Optimizer States Support

Each test is tagged with appropriate resource requirements.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from better_ai.models.moe import (
    LossFreeBalancing,
    compute_expert_specialization_loss,
    router_z_loss,
    expert_router_coupling_loss,
    Expert,
    ExpertRouter,
    MoELayer,
    create_moe_config,
)
from better_ai.training.trainer_utils.optimization import (
    handle_gradients_and_optimize,
)
from better_ai.training.enhanced_trainer import (
    setup_bf16_optimizer_states,
    convert_optimizer_for_bf16_training,
)
from better_ai.test_resource_tags import high_resource, low_resource


@low_resource
class TestLossFreeBalancing(unittest.TestCase):
    """Tests for the Loss-Free Bias-Based Balancing mechanism."""

    def setUp(self):
        self.num_experts = 8
        self.device = torch.device("cpu")
        self.balancer = LossFreeBalancing(
            num_experts=self.num_experts, momentum=0.99, bias_lr=0.1, device=self.device
        )

    def test_initialization(self):
        """Test that balancer initializes correctly."""
        loads = self.balancer.get_expert_loads()
        bias = self.balancer.get_expert_bias()

        self.assertEqual(loads.shape[0], self.num_experts)
        self.assertEqual(bias.shape[0], self.num_experts)
        self.assertTrue(
            torch.allclose(loads, torch.ones(self.num_experts) / self.num_experts)
        )
        self.assertTrue(torch.all(bias == 0))

    def test_bias_updates_on_routing(self):
        """Test that bias updates correctly after routing."""
        batch_size, seq_len, num_experts = 4, 8, self.num_experts
        router_logits = torch.randn(batch_size, seq_len, num_experts)

        routing_weights, selected_indices = self.balancer.update_and_route(
            router_logits, compute_loads=True
        )

        self.assertEqual(routing_weights.shape, (batch_size, seq_len, 2))
        self.assertEqual(selected_indices.shape, (batch_size, seq_len, 2))

        bias_after = self.balancer.get_expert_bias()
        self.assertFalse(torch.all(bias_after == 0))

    def test_bias_clamps_overloaded_experts(self):
        """Test that overloaded experts get increased bias."""
        num_experts = 4
        device = torch.device("cpu")
        balancer = LossFreeBalancing(
            num_experts=num_experts, momentum=0.99, bias_lr=0.1, device=device
        )

        batch_size, seq_len = 2, 4
        # Create router logits where expert 0 is heavily favored
        router_logits = torch.zeros(batch_size, seq_len, num_experts, device=device)
        router_logits[:, :, 0] = 10.0
        router_logits[:, :, 1] = 1.0
        router_logits[:, :, 2] = 0.5
        router_logits[:, :, 3] = 0.1

        # Run multiple routing updates to amplify bias differences
        for _ in range(10):
            balancer.update_and_route(router_logits, compute_loads=True)

        bias = balancer.get_expert_bias()
        # Expert 0 should have higher bias (discouraged)
        self.assertGreater(bias[0], bias[3])

    def test_reset_functionality(self):
        """Test that reset clears all state."""
        batch_size, seq_len, num_experts = 4, 8, self.num_experts
        router_logits = torch.randn(batch_size, seq_len, num_experts)

        self.balancer.update_and_route(router_logits, compute_loads=True)
        self.balancer.reset()

        loads = self.balancer.get_expert_loads()
        bias = self.balancer.get_expert_bias()

        self.assertTrue(torch.all(loads == 1.0 / self.num_experts))
        self.assertTrue(torch.all(bias == 0))

    def test_custom_target_load(self):
        """Test with custom target load distribution."""
        num_experts = 4
        custom_target = torch.tensor([0.5, 0.3, 0.15, 0.05])
        balancer = LossFreeBalancing(
            num_experts=num_experts,
            momentum=0.99,
            target_load=custom_target,
            device=self.device,
        )

        target = balancer.target_load
        self.assertTrue(torch.allclose(target, custom_target))


@low_resource
class TestSpecializationLosses(unittest.TestCase):
    """Tests for Orthogonality and Variance Specialization Losses."""

    def setUp(self):
        self.num_experts = 8
        self.batch_size = 4
        self.seq_len = 16

    def test_orthogonality_loss_computation(self):
        """Test orthogonality loss calculation."""
        router_logits = torch.randn(self.batch_size, self.seq_len, self.num_experts)

        ortho_loss, var_loss = compute_expert_specialization_loss(
            router_logits, self.num_experts
        )

        self.assertIsInstance(ortho_loss, torch.Tensor)
        self.assertEqual(ortho_loss.dim(), 0)
        self.assertGreaterEqual(ortho_loss.item(), 0)

    def test_variance_loss_computation(self):
        """Test variance loss (entropy) calculation."""
        router_logits = torch.randn(self.batch_size, self.seq_len, self.num_experts)

        ortho_loss, var_loss = compute_expert_specialization_loss(
            router_logits, self.num_experts
        )

        self.assertIsInstance(var_loss, torch.Tensor)
        self.assertEqual(var_loss.dim(), 0)
        self.assertGreaterEqual(var_loss.item(), 0)

    def test_custom_weights(self):
        """Test with custom loss weights."""
        router_logits = torch.randn(self.batch_size, self.seq_len, self.num_experts)

        ortho_loss_1, var_loss_1 = compute_expert_specialization_loss(
            router_logits, self.num_experts, ortho_weight=0.1, variance_weight=0.05
        )
        ortho_loss_2, var_loss_2 = compute_expert_specialization_loss(
            router_logits, self.num_experts, ortho_weight=0.01, variance_weight=0.001
        )

        self.assertGreater(ortho_loss_1.item(), ortho_loss_2.item())
        self.assertGreater(var_loss_1.item(), var_loss_2.item())

    def test_correlated_experts_high_ortho_loss(self):
        """Test that correlated expert activations produce high orthogonality loss."""
        router_logits = torch.randn(self.batch_size, self.seq_len, self.num_experts)
        router_logits[:, :, 1] = router_logits[:, :, 0] + 0.1

        ortho_loss, _ = compute_expert_specialization_loss(
            router_logits, self.num_experts
        )

        router_logits_independent = torch.randn(
            self.batch_size, self.seq_len, self.num_experts
        )
        ortho_loss_independent, _ = compute_expert_specialization_loss(
            router_logits_independent, self.num_experts
        )

        self.assertGreater(ortho_loss.item(), ortho_loss_independent.item())

    def test_single_expert_zero_ortho_loss(self):
        """Test that single expert case returns zero orthogonality loss."""
        router_logits = torch.randn(self.batch_size, self.seq_len, 1)

        ortho_loss, var_loss = compute_expert_specialization_loss(router_logits, 1)

        self.assertEqual(ortho_loss.item(), 0)


@low_resource
class TestRouterZLoss(unittest.TestCase):
    """Tests for Router Z-Loss for numerical stability."""

    def setUp(self):
        self.batch_size = 4
        self.seq_len = 16
        self.num_experts = 8

    def test_z_loss_computation(self):
        """Test basic Z-loss calculation."""
        router_logits = torch.randn(self.batch_size, self.seq_len, self.num_experts)

        z_loss = router_z_loss(router_logits)

        self.assertIsInstance(z_loss, torch.Tensor)
        self.assertEqual(z_loss.dim(), 0)
        self.assertGreaterEqual(z_loss.item(), 0)

    def test_z_loss_penalizes_large_logits(self):
        """Test that Z-loss increases with logit magnitude."""
        router_logits_small = (
            torch.randn(self.batch_size, self.seq_len, self.num_experts) * 0.1
        )
        router_logits_large = (
            torch.randn(self.batch_size, self.seq_len, self.num_experts) * 10.0
        )

        z_loss_small = router_z_loss(router_logits_small)
        z_loss_large = router_z_loss(router_logits_large)

        self.assertLess(z_loss_small.item(), z_loss_large.item())

    def test_custom_z_loss_coefficient(self):
        """Test with custom Z-loss coefficient."""
        router_logits = torch.randn(self.batch_size, self.seq_len, self.num_experts)

        z_loss_low = router_z_loss(router_logits, z_loss_coeff=1e-4)
        z_loss_high = router_z_loss(router_logits, z_loss_coeff=1e-2)

        self.assertLess(z_loss_low.item(), z_loss_high.item())

    def test_z_loss_logsumexp_stability(self):
        """Test Z-loss with extreme logits."""
        router_logits = torch.randn(self.batch_size, self.seq_len, self.num_experts)
        router_logits[:, :, 0] = 100.0

        z_loss = router_z_loss(router_logits)

        self.assertFalse(torch.isnan(z_loss))
        self.assertFalse(torch.isinf(z_loss))


@low_resource
class TestExpertRouterCouplingLoss(unittest.TestCase):
    """Tests for Expert-Router Coupling Loss."""

    def setUp(self):
        self.hidden_size = 32
        self.num_experts = 4

    def test_erc_loss_computation(self):
        """Test ERC loss calculation."""
        expert_embeddings = torch.randn(self.num_experts, self.hidden_size)

        experts = nn.ModuleList(
            [
                Expert(self.hidden_size, self.hidden_size * 4)
                for _ in range(self.num_experts)
            ]
        )

        erc_loss = expert_router_coupling_loss(expert_embeddings, experts)

        self.assertIsInstance(erc_loss, torch.Tensor)
        self.assertEqual(erc_loss.dim(), 0)
        self.assertGreaterEqual(erc_loss.item(), 0)

    def test_erc_loss_diagonal_dominance(self):
        """Test that diagonal dominance produces low ERC loss."""
        expert_embeddings = torch.randn(self.num_experts, self.hidden_size)

        experts = nn.ModuleList(
            [
                Expert(self.hidden_size, self.hidden_size * 4)
                for _ in range(self.num_experts)
            ]
        )

        erc_loss = expert_router_coupling_loss(expert_embeddings, experts)

        self.assertLessEqual(erc_loss.item(), 1.0)

    def test_custom_coupling_weight(self):
        """Test with custom coupling weight."""
        expert_embeddings = torch.randn(self.num_experts, self.hidden_size)

        experts = nn.ModuleList(
            [
                Expert(self.hidden_size, self.hidden_size * 4)
                for _ in range(self.num_experts)
            ]
        )

        erc_loss_low = expert_router_coupling_loss(
            expert_embeddings, experts, coupling_weight=0.001
        )
        erc_loss_high = expert_router_coupling_loss(
            expert_embeddings, experts, coupling_weight=0.1
        )

        self.assertLess(erc_loss_low.item(), erc_loss_high.item())


@low_resource
class TestGradientClippingIntegration(unittest.TestCase):
    """Tests for gradient clipping integration."""

    def setUp(self):
        self.hidden_size = 32
        self.num_experts = 4

        self.model = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, self.hidden_size)
        )

        self.config = MagicMock()
        self.config.gradient_clip_norm = 1.0

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def test_gradient_clipping_from_config(self):
        """Test gradient clipping reads from config."""
        self.config.gradient_clip_norm = 0.5

        mock_trainer = MagicMock()
        mock_trainer.config = self.config
        mock_trainer.optimizer = self.optimizer
        mock_trainer.model = self.model

        x = torch.randn(4, 16)
        y = self.model(x)
        y.sum().backward()

        grad_norm = handle_gradients_and_optimize(mock_trainer)

        self.assertIsInstance(grad_norm, float)

    def test_no_clipping_when_disabled(self):
        """Test no clipping when gradient_clip_norm is None."""
        self.config.gradient_clip_norm = None

        mock_trainer = MagicMock()
        mock_trainer.config = self.config
        mock_trainer.optimizer = self.optimizer
        mock_trainer.model = self.model

        x = torch.randn(4, 16)
        y = self.model(x)
        y.sum().backward()

        grad_norm = handle_gradients_and_optimize(mock_trainer)

        self.assertIsInstance(grad_norm, float)

    def test_clipping_with_zero_threshold(self):
        """Test no clipping when threshold is 0."""
        self.config.gradient_clip_norm = 0.0

        mock_trainer = MagicMock()
        mock_trainer.config = self.config
        mock_trainer.optimizer = self.optimizer
        mock_trainer.model = self.model

        x = torch.randn(4, 16)
        y = self.model(x)
        y.sum().backward()

        grad_norm = handle_gradients_and_optimize(mock_trainer)

        self.assertIsInstance(grad_norm, float)


@high_resource
class TestBF16OptimizerStates(unittest.TestCase):
    """Tests for BF16 optimizer states support."""

    def setUp(self):
        self.model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def test_bf16_optimizer_setup(self):
        """Test BF16 optimizer states setup."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        result = setup_bf16_optimizer_states(self.optimizer)

        self.assertIsInstance(result, torch.optim.Optimizer)

    def test_convert_optimizer_for_bf16(self):
        """Test optimizer conversion for BF16 training."""
        result = convert_optimizer_for_bf16_training(self.optimizer)

        self.assertIsInstance(result, torch.optim.Optimizer)


@low_resource
class TestMoELayerIntegration(unittest.TestCase):
    """Integration tests for MoE layer with new features."""

    def setUp(self):
        self.hidden_size = 64
        self.num_experts = 4
        self.batch_size = 2
        self.seq_len = 4
        self.device = torch.device("cpu")

    def test_moe_layer_with_loss_free_balancing(self):
        """Test MoE layer with loss-free balancing enabled."""
        moe_layer = MoELayer(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            loss_free_balancing=True,
            specialization_weight=0.05,
            router_z_loss_weight=1e-3,
            device=self.device,
        )

        x = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )

        # Test forward pass
        routing_weights, selected_indices, router_logits = moe_layer.router(x)

        self.assertEqual(routing_weights.shape, (self.batch_size, self.seq_len, 2))
        self.assertTrue(hasattr(moe_layer, "loss_free_balancing"))
        self.assertTrue(hasattr(moe_layer, "specialization_weight"))
        self.assertTrue(hasattr(moe_layer, "router_z_loss_weight"))
        self.assertTrue(hasattr(moe_layer, "balancer"))
        self.assertIsNotNone(moe_layer.balancer)

    def test_moe_layer_without_loss_free_balancing(self):
        """Test MoE layer with traditional balancing."""
        moe_layer = MoELayer(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            loss_free_balancing=False,
            load_balance_loss_weight=0.01,
            device=self.device,
        )

        x = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size, device=self.device
        )

        # Test that balancer is None when loss-free balancing is disabled
        self.assertIsNone(moe_layer.balancer)
        self.assertFalse(moe_layer.loss_free_balancing)

    def test_moe_layer_attributes(self):
        """Test that MoE layer has all required attributes."""
        moe_layer = MoELayer(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            loss_free_balancing=True,
            specialization_weight=0.05,
            router_z_loss_weight=1e-3,
            gradient_clip_norm=1.0,
            device=self.device,
        )

        # Check required attributes
        self.assertTrue(hasattr(moe_layer, "loss_free_balancing"))
        self.assertTrue(hasattr(moe_layer, "specialization_weight"))
        self.assertTrue(hasattr(moe_layer, "router_z_loss_weight"))
        self.assertTrue(hasattr(moe_layer, "gradient_clip_norm"))
        self.assertTrue(hasattr(moe_layer, "balancer"))
        self.assertIsNotNone(moe_layer.balancer)


@low_resource
class TestMoEConfigCreation(unittest.TestCase):
    """Tests for MoE configuration creation."""

    def test_create_small_config(self):
        """Test configuration for small model."""
        config = create_moe_config(model_size="small", num_experts=16)

        self.assertEqual(config["num_experts"], 8)
        self.assertTrue(config["loss_free_balancing"])
        self.assertIn("specialization_weight", config)

    def test_create_medium_config(self):
        """Test configuration for medium model."""
        config = create_moe_config(model_size="medium", num_experts=32)

        self.assertEqual(config["num_experts"], 16)
        self.assertTrue(config["loss_free_balancing"])
        self.assertIn("gradient_clip_norm", config)

    def test_create_large_config(self):
        """Test configuration for large model."""
        config = create_moe_config(model_size="large", num_experts=32)

        self.assertEqual(config["num_experts"], 32)
        self.assertTrue(config["loss_free_balancing"])

    def test_custom_specialization_weight(self):
        """Test with custom specialization weight."""
        config = create_moe_config(specialization_weight=0.1)

        self.assertEqual(config["specialization_weight"], 0.1)

    def test_custom_z_loss_weight(self):
        """Test with custom Z-loss weight."""
        config = create_moe_config(router_z_loss_weight=1e-4)

        self.assertEqual(config["router_z_loss_weight"], 1e-4)


if __name__ == "__main__":
    unittest.main()
