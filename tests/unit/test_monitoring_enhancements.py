"""
Tests for Monitoring Enhancements and Dashboard

Tests for:
1. HTML Dashboard metrics and rendering
2. Gradient Noise Scale (GNS) estimation
3. Weight Entropy calculation
4. Updated Entropic Steering
"""

import unittest
import torch
import torch.nn as nn
import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from better_ai.monitoring.dashboard import HTMLDashboard, LogLevel
from better_ai.training.enhanced_trainer import EnhancedMoETrainer
from better_ai.models.core import DeepSeekModel
from better_ai.models.features.entropic_steering import EntropicSteering
from better_ai.config import ModelConfig, TrainingConfig
from better_ai.test_resource_tags import low_resource

@low_resource
class TestMonitoringEnhancements(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig.get_small_model_config()
        self.device = torch.device("cpu")
        self.model = DeepSeekModel(self.config, device=self.device)
        self.dashboard = HTMLDashboard()

    def test_dashboard_metrics_update(self):
        """Test that dashboard correctly stores and retrieves new metrics."""
        step = 10
        weight_entropy = 3.5
        power_draw = 200.0
        utilization = 0.8
        gns = 0.05

        self.dashboard.update_system_metrics(weight_entropy, power_draw, step)
        self.dashboard.update_moe_metrics(utilization, gns, step)

        data = self.dashboard.get_dashboard_data()

        self.assertEqual(len(data["weight_entropy_history"]), 1)
        self.assertEqual(data["weight_entropy_history"][0]["entropy"], weight_entropy)
        self.assertEqual(data["power_draw_history"][0]["power"], power_draw)
        self.assertEqual(data["expert_utilization_history"][0]["utilization"], utilization)
        self.assertEqual(data["gns_history"][0]["gns"], gns)

    def test_dashboard_html_rendering(self):
        """Test that HTML report contains the new metric sections."""
        self.dashboard.update_system_metrics(3.5, 200.0, 1)
        self.dashboard.update_moe_metrics(0.8, 0.05, 1)

        html = self.dashboard.get_html_report()

        self.assertIn("Grokking & Noise", html)
        self.assertIn("MoE & System", html)
        self.assertIn("Expert Utilization:", html)
        self.assertIn("Power Draw:", html)

    def test_weight_entropy_calculation(self):
        """Test that model can calculate its own weight entropy."""
        entropy = self.model.calculate_weight_entropy()
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0.0)

    def test_gns_estimation_logic(self):
        """Test GNS estimation logic in trainer."""
        # Mock trainer enough to test GNS
        class MockTrainer:
            def __init__(self, model):
                self.model = model
                self._grad_buffer = deque(maxlen=20)

            _collect_grad_sample = EnhancedMoETrainer._collect_grad_sample
            _estimate_gradient_noise_scale = EnhancedMoETrainer._estimate_gradient_noise_scale

        trainer = MockTrainer(self.model)

        # Manually create some gradients
        for p in self.model.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p) * 0.1

        # Collect samples
        for _ in range(10):
            trainer._collect_grad_sample()
            # Perturb grads slightly for next sample
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * 0.01

        gns = trainer._estimate_gradient_noise_scale()
        self.assertGreater(gns, 0.0)

    def test_updated_entropic_steering(self):
        """Test that entropic steering uses weight entropy."""
        steering = EntropicSteering(hidden_dim=self.config.hidden_dim, entropy_threshold=2.5)
        hidden_states = torch.randn(1, 4, self.config.hidden_dim)
        logits = torch.randn(1, 4, self.config.vocab_size)

        # Test with low weight entropy (should be more sensitive)
        out_low = steering(hidden_states, logits, weight_entropy=1.0)

        # Test with high weight entropy (should be less sensitive)
        out_high = steering(hidden_states, logits, weight_entropy=5.0)

        self.assertIn("weight_entropy_used", out_low)
        self.assertIn("effective_threshold", out_low)
        self.assertLess(out_low["effective_threshold"], out_high["effective_threshold"])

if __name__ == "__main__":
    unittest.main()
