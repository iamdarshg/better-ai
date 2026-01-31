"""
Unit tests for configuration validation and edge cases
Tests configuration parsing, validation, and error handling
"""

import unittest
import json
import yaml
from better_ai.config import ModelConfig, TrainingConfig, InferenceConfig
from better_ai.utils.exceptions import ConfigError
from better_ai.test_config_utils import get_small_model_config


class TestModelConfigValidation(unittest.TestCase):
    """Test ModelConfig validation and edge cases."""

    def test_default_config_creation(self):
        """Test that default config is created successfully."""
        config = ModelConfig()
        self.assertGreater(config.vocab_size, 0)
        self.assertGreater(config.hidden_dim, 0)
        self.assertGreater(config.num_layers, 0)

    def test_config_validation_invalid_values(self):
        """Test that invalid config values raise appropriate errors."""
        # Test negative values
        with self.assertRaises(ConfigError):
            ModelConfig(vocab_size=-1)

        with self.assertRaises(ConfigError):
            ModelConfig(hidden_dim=0)

        # Test invalid combinations
        with self.assertRaises(ConfigError):
            ModelConfig(
                num_key_value_heads=64, num_attention_heads=32
            )  # kv_heads > att_heads

    def test_moe_config_validation(self):
        """Test MoE-specific configuration validation."""
        # Valid MoE config
        config = ModelConfig(
            num_experts=8, num_experts_per_token=2, expert_capacity_factor=1.25
        )
        self.assertEqual(config.num_experts, 8)
        self.assertEqual(config.num_experts_per_token, 2)

        # Invalid MoE config
        with self.assertRaises(ConfigError):
            ModelConfig(num_experts=4, num_experts_per_token=8)  # per_token > total

    def test_attention_config_validation(self):
        """Test attention-specific configuration validation."""
        # Test ring attention config
        config = ModelConfig(
            use_ring_attention=True, ring_block_size=1024, ring_num_devices=2
        )
        self.assertTrue(config.use_ring_attention)
        self.assertEqual(config.ring_block_size, 1024)

        # Invalid ring attention
        with self.assertRaises(ConfigError):
            ModelConfig(use_ring_attention=True, ring_block_size=0)

    def test_config_serialization(self):
        """Test config serialization to dict and back."""
        original = ModelConfig(vocab_size=4096, hidden_dim=512)
        config_dict = original.to_dict()

        # Test dict creation
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["vocab_size"], 4096)

        # Test reconstruction
        reconstructed = ModelConfig.from_dict(config_dict)
        self.assertEqual(reconstructed.vocab_size, original.vocab_size)
        self.assertEqual(reconstructed.hidden_dim, original.hidden_dim)


class TestTrainingConfigValidation(unittest.TestCase):
    """Test TrainingConfig validation and edge cases."""

    def test_default_training_config(self):
        """Test that default training config is created successfully."""
        config = TrainingConfig()
        self.assertGreater(config.batch_size, 0)
        self.assertGreater(config.learning_rate, 0)
        self.assertGreater(config.max_steps, 0)

    def test_optimizer_config_validation(self):
        """Test optimizer configuration validation."""
        # Valid optimizer configs
        config = TrainingConfig(
            optimizer="adamw", beta1=0.9, beta2=0.95, weight_decay=0.1
        )
        self.assertEqual(config.optimizer, "adamw")
        self.assertTrue(0 <= config.beta1 < 1)
        self.assertTrue(0 <= config.beta2 < 1)

        # Invalid optimizer
        with self.assertRaises(ConfigError):
            TrainingConfig(optimizer="invalid_optimizer")

        # Invalid beta values
        with self.assertRaises(ConfigError):
            TrainingConfig(beta1=1.5)

    def test_learning_rate_schedule_validation(self):
        """Test learning rate schedule validation."""
        # Valid schedules
        for schedule in ["cosine", "linear", "constant"]:
            config = TrainingConfig(lr_schedule=schedule)
            self.assertEqual(config.lr_schedule, schedule)

        # Invalid schedule
        with self.assertRaises(ConfigError):
            TrainingConfig(lr_schedule="invalid_schedule")

    def test_gradient_accumulation_validation(self):
        """Test gradient accumulation configuration."""
        # Valid accumulation
        config = TrainingConfig(batch_size=8, gradient_accumulation_steps=4)
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.gradient_accumulation_steps, 4)

        # Invalid values
        with self.assertRaises(ConfigError):
            TrainingConfig(gradient_accumulation_steps=0)

    def test_checkpoint_config_validation(self):
        """Test checkpointing configuration validation."""
        config = TrainingConfig(save_steps=100, eval_steps=50, save_total_limit=3)
        self.assertEqual(config.save_steps, 100)
        self.assertEqual(config.eval_steps, 50)
        self.assertEqual(config.save_total_limit, 3)

        # Invalid checkpointing
        with self.assertRaises(ConfigError):
            TrainingConfig(save_steps=0)


class TestInferenceConfigValidation(unittest.TestCase):
    """Test InferenceConfig validation and edge cases."""

    def test_default_inference_config(self):
        """Test that default inference config is created successfully."""
        config = InferenceConfig()
        self.assertGreater(config.max_new_tokens, 0)
        self.assertGreater(config.batch_size, 0)

    def test_generation_config_validation(self):
        """Test generation parameter validation."""
        # Valid generation config
        config = InferenceConfig(
            max_new_tokens=256, temperature=0.8, top_k=50, top_p=0.9
        )
        self.assertTrue(0 <= config.temperature <= 2.0)
        self.assertTrue(0 <= config.top_p <= 1.0)
        self.assertGreater(config.top_k, 0)

        # Invalid values
        with self.assertRaises(ConfigError):
            InferenceConfig(temperature=-1.0)

        with self.assertRaises(ConfigError):
            InferenceConfig(top_p=1.5)

    def test_quantization_config_validation(self):
        """Test quantization configuration validation."""
        config = InferenceConfig(
            quantize_weights=True, weight_bits=8, activation_bits=8
        )
        self.assertTrue(config.quantize_weights)
        self.assertEqual(config.weight_bits, 8)

        # Invalid bit depths
        with self.assertRaises(ConfigError):
            InferenceConfig(quantize_weights=True, weight_bits=7)


class TestConfigIntegration(unittest.TestCase):
    """Test configuration integration and compatibility."""

    def test_model_training_compatibility(self):
        """Test that model and training configs are compatible."""
        model_config = ModelConfig(hidden_dim=512, vocab_size=4096, use_fp8=True)

        training_config = TrainingConfig(batch_size=4, fp8_loss_scale=1.0, bf16=True)

        # Should not raise any errors
        self.assertEqual(model_config.hidden_dim, 512)
        self.assertEqual(training_config.batch_size, 4)

    def test_config_inheritance_and_overrides(self):
        """Test config inheritance and parameter overrides."""
        base_config = ModelConfig(hidden_dim=512, vocab_size=4096)

        # Override specific parameters
        overridden_config = ModelConfig(
            hidden_dim=1024,  # Override
            vocab_size=4096,  # Keep same
        )

        self.assertEqual(overridden_config.hidden_dim, 1024)
        self.assertEqual(overridden_config.vocab_size, 4096)

    def test_config_edge_cases(self):
        """Test edge cases in configuration."""
        # Minimal viable config
        minimal = ModelConfig(
            vocab_size=1000, hidden_dim=128, num_layers=1, num_attention_heads=4
        )
        self.assertEqual(minimal.vocab_size, 1000)

        # Large config (within reasonable bounds)
        large = ModelConfig(
            vocab_size=100000, hidden_dim=8192, num_layers=64, num_attention_heads=128
        )
        self.assertEqual(large.vocab_size, 100000)


class TestConfigFileHandling(unittest.TestCase):
    """Test configuration file loading and saving."""

    def test_json_config_loading(self):
        """Test loading config from JSON file."""
        import tempfile
        import os
        config_data = {
            "vocab_size": 4096,
            "hidden_dim": 512,
            "num_layers": 12,
            "num_attention_heads": 16,
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            config = ModelConfig.from_file(config_file)
            self.assertEqual(config.vocab_size, 4096)
            self.assertEqual(config.hidden_dim, 512)
        finally:
            if os.path.exists(config_file):
                os.remove(config_file)

    def test_yaml_config_loading(self):
        """Test loading config from YAML file."""
        import tempfile
        import os
        config_data = """
        vocab_size: 4096
        hidden_dim: 512
        num_layers: 12
        num_attention_heads: 16
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_data)
            config_file = f.name

        try:
            config = ModelConfig.from_file(config_file)
            self.assertEqual(config.vocab_size, 4096)
            self.assertEqual(config.hidden_dim, 512)
        finally:
            if os.path.exists(config_file):
                os.remove(config_file)

    def test_config_saving(self):
        """Test saving config to file."""
        import tempfile
        import os
        config = ModelConfig(vocab_size=4096, hidden_dim=512)

        # Save as JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name

        try:
            config.to_file(json_file)

            # Load and verify
            loaded = ModelConfig.from_file(json_file)
            self.assertEqual(loaded.vocab_size, config.vocab_size)
            self.assertEqual(loaded.hidden_dim, config.hidden_dim)
        finally:
            if os.path.exists(json_file):
                os.remove(json_file)

if __name__ == "__main__":
    unittest.main()
