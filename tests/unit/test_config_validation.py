"""
Unit tests for configuration validation and edge cases
Tests configuration parsing, validation, and error handling
"""

import pytest
import json
import yaml
from better_ai.config import ModelConfig, TrainingConfig, InferenceConfig
from better_ai.utils.exceptions import ConfigError


class TestModelConfigValidation:
    """Test ModelConfig validation and edge cases."""

    def test_default_config_creation(self):
        """Test that default config is created successfully."""
        config = ModelConfig()
        assert config.vocab_size > 0
        assert config.hidden_dim > 0
        assert config.num_layers > 0

    def test_config_validation_invalid_values(self):
        """Test that invalid config values raise appropriate errors."""
        # Test negative values
        with pytest.raises(ConfigError):
            ModelConfig(vocab_size=-1)

        with pytest.raises(ConfigError):
            ModelConfig(hidden_dim=0)

        # Test invalid combinations
        with pytest.raises(ConfigError):
            ModelConfig(
                num_key_value_heads=64, num_attention_heads=32
            )  # kv_heads > att_heads

    def test_moe_config_validation(self):
        """Test MoE-specific configuration validation."""
        # Valid MoE config
        config = ModelConfig(
            num_experts=8, num_experts_per_token=2, expert_capacity_factor=1.25
        )
        assert config.num_experts == 8
        assert config.num_experts_per_token == 2

        # Invalid MoE config
        with pytest.raises(ConfigError):
            ModelConfig(num_experts=4, num_experts_per_token=8)  # per_token > total

    def test_attention_config_validation(self):
        """Test attention-specific configuration validation."""
        # Test ring attention config
        config = ModelConfig(
            use_ring_attention=True, ring_block_size=1024, ring_num_devices=2
        )
        assert config.use_ring_attention
        assert config.ring_block_size == 1024

        # Invalid ring attention
        with pytest.raises(ConfigError):
            ModelConfig(use_ring_attention=True, ring_block_size=0)

    def test_config_serialization(self):
        """Test config serialization to dict and back."""
        original = ModelConfig(vocab_size=4096, hidden_dim=512)
        config_dict = original.to_dict()

        # Test dict creation
        assert isinstance(config_dict, dict)
        assert config_dict["vocab_size"] == 4096

        # Test reconstruction
        reconstructed = ModelConfig.from_dict(config_dict)
        assert reconstructed.vocab_size == original.vocab_size
        assert reconstructed.hidden_dim == original.hidden_dim


class TestTrainingConfigValidation:
    """Test TrainingConfig validation and edge cases."""

    def test_default_training_config(self):
        """Test that default training config is created successfully."""
        config = TrainingConfig()
        assert config.batch_size > 0
        assert config.learning_rate > 0
        assert config.max_steps > 0

    def test_optimizer_config_validation(self):
        """Test optimizer configuration validation."""
        # Valid optimizer configs
        config = TrainingConfig(
            optimizer="adamw", beta1=0.9, beta2=0.95, weight_decay=0.1
        )
        assert config.optimizer == "adamw"
        assert 0 <= config.beta1 < 1
        assert 0 <= config.beta2 < 1

        # Invalid optimizer
        with pytest.raises(ConfigError):
            TrainingConfig(optimizer="invalid_optimizer")

        # Invalid beta values
        with pytest.raises(ConfigError):
            TrainingConfig(beta1=1.5)

    def test_learning_rate_schedule_validation(self):
        """Test learning rate schedule validation."""
        # Valid schedules
        for schedule in ["cosine", "linear", "constant"]:
            config = TrainingConfig(lr_schedule=schedule)
            assert config.lr_schedule == schedule

        # Invalid schedule
        with pytest.raises(ConfigError):
            TrainingConfig(lr_schedule="invalid_schedule")

    def test_gradient_accumulation_validation(self):
        """Test gradient accumulation configuration."""
        # Valid accumulation
        config = TrainingConfig(batch_size=8, gradient_accumulation_steps=4)
        assert config.batch_size == 8
        assert config.gradient_accumulation_steps == 4

        # Invalid values
        with pytest.raises(ConfigError):
            TrainingConfig(gradient_accumulation_steps=0)

    def test_checkpoint_config_validation(self):
        """Test checkpointing configuration validation."""
        config = TrainingConfig(save_steps=100, eval_steps=50, save_total_limit=3)
        assert config.save_steps == 100
        assert config.eval_steps == 50
        assert config.save_total_limit == 3

        # Invalid checkpointing
        with pytest.raises(ConfigError):
            TrainingConfig(save_steps=0)


class TestInferenceConfigValidation:
    """Test InferenceConfig validation and edge cases."""

    def test_default_inference_config(self):
        """Test that default inference config is created successfully."""
        config = InferenceConfig()
        assert config.max_new_tokens > 0
        assert config.batch_size > 0

    def test_generation_config_validation(self):
        """Test generation parameter validation."""
        # Valid generation config
        config = InferenceConfig(
            max_new_tokens=256, temperature=0.8, top_k=50, top_p=0.9
        )
        assert 0 <= config.temperature <= 2.0
        assert 0 <= config.top_p <= 1.0
        assert config.top_k > 0

        # Invalid values
        with pytest.raises(ConfigError):
            InferenceConfig(temperature=-1.0)

        with pytest.raises(ConfigError):
            InferenceConfig(top_p=1.5)

    def test_quantization_config_validation(self):
        """Test quantization configuration validation."""
        config = InferenceConfig(
            quantize_weights=True, weight_bits=8, activation_bits=8
        )
        assert config.quantize_weights
        assert config.weight_bits == 8

        # Invalid bit depths
        with pytest.raises(ConfigError):
            InferenceConfig(quantize_weights=True, weight_bits=7)


class TestConfigIntegration:
    """Test configuration integration and compatibility."""

    def test_model_training_compatibility(self):
        """Test that model and training configs are compatible."""
        model_config = ModelConfig(hidden_dim=512, vocab_size=4096, use_fp8=True)

        training_config = TrainingConfig(batch_size=4, fp8_loss_scale=1.0, bf16=True)

        # Should not raise any errors
        assert model_config.hidden_dim == 512
        assert training_config.batch_size == 4

    def test_config_inheritance_and_overrides(self):
        """Test config inheritance and parameter overrides."""
        base_config = ModelConfig(hidden_dim=512, vocab_size=4096)

        # Override specific parameters
        overridden_config = ModelConfig(
            hidden_dim=1024,  # Override
            vocab_size=4096,  # Keep same
        )

        assert overridden_config.hidden_dim == 1024
        assert overridden_config.vocab_size == 4096

    def test_config_edge_cases(self):
        """Test edge cases in configuration."""
        # Minimal viable config
        minimal = ModelConfig(
            vocab_size=1000, hidden_dim=128, num_layers=1, num_attention_heads=4
        )
        assert minimal.vocab_size == 1000

        # Large config (within reasonable bounds)
        large = ModelConfig(
            vocab_size=100000, hidden_dim=8192, num_layers=64, num_attention_heads=128
        )
        assert large.vocab_size == 100000


class TestConfigFileHandling:
    """Test configuration file loading and saving."""

    def test_json_config_loading(self, tmp_path):
        """Test loading config from JSON file."""
        config_data = {
            "vocab_size": 4096,
            "hidden_dim": 512,
            "num_layers": 12,
            "num_attention_heads": 16,
        }

        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = ModelConfig.from_file(str(config_file))
        assert config.vocab_size == 4096
        assert config.hidden_dim == 512

    def test_yaml_config_loading(self, tmp_path):
        """Test loading config from YAML file."""
        config_data = """
        vocab_size: 4096
        hidden_dim: 512
        num_layers: 12
        num_attention_heads: 16
        """

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            f.write(config_data)

        config = ModelConfig.from_file(str(config_file))
        assert config.vocab_size == 4096
        assert config.hidden_dim == 512

    def test_config_saving(self, tmp_path):
        """Test saving config to file."""
        config = ModelConfig(vocab_size=4096, hidden_dim=512)

        # Save as JSON
        json_file = tmp_path / "config.json"
        config.to_file(str(json_file))

        # Load and verify
        loaded = ModelConfig.from_file(str(json_file))
        assert loaded.vocab_size == config.vocab_size
        assert loaded.hidden_dim == config.hidden_dim
