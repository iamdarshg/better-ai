"""
Tests for model export functionality (GGUF and vLLM)
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
import torch
import numpy as np

from better_ai.config import ModelConfig
from better_ai.models.core import DeepSeekModel
from better_ai.models.moe import DeepSeekMoEModel


class TestGGUFExport(unittest.TestCase):
    """Test GGUF export functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ModelConfig.get_small_model_config()
        self.model = DeepSeekModel(self.config)

        # Save a test checkpoint
        self.checkpoint_path = os.path.join(self.temp_dir, "test_model.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config.to_dict(),
            },
            self.checkpoint_path,
        )

        self.config_path = os.path.join(self.temp_dir, "config.json")
        with open(self.config_path, "w") as f:
            json.dump(self.config.to_dict(), f)

    def test_tensor_name_mapping(self):
        """Test PyTorch to GGUF tensor name mapping."""
        from scripts.convert_to_gguf import get_gguf_tensor_name

        # Test embedding mapping
        self.assertEqual(
            get_gguf_tensor_name("embed_tokens.weight"), "token_embd.weight"
        )

        # Test attention mapping
        self.assertEqual(
            get_gguf_tensor_name("layers.0.self_attn.q_proj.weight"),
            "blk.0.attn_q.weight",
        )

        # Test MoE expert mapping
        result = get_gguf_tensor_name("layers.0.mlp.experts.0.gate_up_proj.weight")
        self.assertIn("expert_0_gate_up.weight", result)
        self.assertTrue(result.startswith("blk.0."))

    def test_advanced_feature_tensor_mapping(self):
        """Test advanced feature tensor name mapping."""
        from scripts.convert_to_gguf import get_gguf_tensor_name

        # Test scratchpad mapping
        self.assertEqual(
            get_gguf_tensor_name("scratchpad.scratchpad_transform.weight"),
            "scratchpad_transform.weight",
        )

        # Test TiDAR mapping
        self.assertEqual(
            get_gguf_tensor_name("tidar.refinement_head.weight"),
            "tidar_refinement.weight",
        )

        # Test reward model mapping
        self.assertEqual(
            get_gguf_tensor_name("reward_model.classifier.weight"),
            "reward_classifier.weight",
        )

    def test_quantization_f16(self):
        """Test F16 quantization."""
        from scripts.convert_to_gguf import quantize_tensor

        test_tensor = np.random.randn(10, 10).astype(np.float32)
        quantized, dtype = quantize_tensor(test_tensor, "f16")

        self.assertEqual(quantized.dtype, np.float16)
        self.assertIsNotNone(dtype)

    def test_quantization_q8_0(self):
        """Test Q8_0 quantization."""
        from scripts.convert_to_gguf import quantize_tensor

        test_tensor = np.random.randn(10, 10).astype(np.float32)
        quantized, dtype = quantize_tensor(test_tensor, "q8_0")

        self.assertEqual(quantized.dtype, np.int8)
        self.assertIsNotNone(dtype)

    def test_advanced_features_metadata(self):
        """Test advanced features metadata generation."""
        try:
            from scripts.convert_to_gguf import add_advanced_features_metadata

            # Create mock writer
            class MockWriter:
                def __init__(self):
                    self.metadata = {}

                def add_bool(self, key, value):
                    self.metadata[key] = value

                def add_uint32(self, key, value):
                    self.metadata[key] = value

                def add_float32(self, key, value):
                    self.metadata[key] = value

                def add_string(self, key, value):
                    self.metadata[key] = value

            writer = MockWriter()
            config = ModelConfig(
                use_recursive_scratchpad=True,
                use_tidar=True,
                use_cot_specialization=True,
                scratchpad_max_iterations=8,
                tidar_num_steps=3,
            )

            add_advanced_features_metadata(writer, config)

            # Check that metadata was added
            self.assertTrue(writer.metadata.get("better_ai.use_recursive_scratchpad"))
            self.assertTrue(writer.metadata.get("better_ai.use_tidar"))
            self.assertEqual(
                writer.metadata.get("better_ai.scratchpad_max_iterations"), 8
            )
            self.assertEqual(writer.metadata.get("better_ai.tidar_num_steps"), 3)
        except ImportError:
            self.skipTest("GGUF library not installed")

    def test_export_with_advanced_features(self):
        """Test full GGUF export with advanced features."""
        try:
            from scripts.convert_to_gguf import convert_to_gguf

            output_path = os.path.join(self.temp_dir, "output.gguf")

            # Export with advanced features enabled
            convert_to_gguf(
                model_path=self.checkpoint_path,
                output_path=output_path,
                quantization="f16",
                config_path=self.config_path,
            )

            # Check output file exists
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

        except ImportError:
            self.skipTest("GGUF library not installed")


class TestVLLMExport(unittest.TestCase):
    """Test vLLM export functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ModelConfig.get_small_model_config()
        self.model = DeepSeekModel(self.config)

        # Save a test checkpoint
        self.checkpoint_path = os.path.join(self.temp_dir, "test_model.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config.to_dict(),
            },
            self.checkpoint_path,
        )

        self.config_path = os.path.join(self.temp_dir, "config.json")
        with open(self.config_path, "w") as f:
            json.dump(self.config.to_dict(), f)

    def test_get_vllm_config(self):
        """Test vLLM config generation."""
        from better_ai.inference.vllm_compat import get_vllm_config

        config = ModelConfig(
            num_experts=8,
            num_experts_per_token=2,
            use_recursive_scratchpad=True,
            use_tidar=True,
            use_cot_specialization=True,
            use_star=True,
            use_tool_heads=True,
        )

        vllm_config = get_vllm_config(config)

        # Check core architecture
        self.assertEqual(vllm_config["architecture"], "DeepSeekForCausalLM")
        self.assertEqual(vllm_config["hidden_size"], config.hidden_dim)
        self.assertEqual(vllm_config["num_hidden_layers"], config.num_layers)

        # Check MoE params
        self.assertEqual(vllm_config["moe_num_experts"], 8)
        self.assertEqual(vllm_config["moe_top_k"], 2)

        # Check advanced features
        self.assertTrue(vllm_config["better_ai_config"]["use_recursive_scratchpad"])
        self.assertTrue(vllm_config["better_ai_config"]["use_tidar"])
        self.assertTrue(vllm_config["better_ai_config"]["use_cot_specialization"])

    def test_vllm_model_wrapper_init(self):
        """Test VLLMDeepSeekModel initialization."""
        from better_ai.inference.vllm_compat import VLLMDeepSeekModel

        wrapper = VLLMDeepSeekModel(self.config)

        self.assertIsNotNone(wrapper.model)
        self.assertEqual(wrapper.config, self.config)

    def test_vllm_model_forward(self):
        """Test VLLMDeepSeekModel forward pass."""
        from better_ai.inference.vllm_compat import VLLMDeepSeekModel

        wrapper = VLLMDeepSeekModel(self.config)
        wrapper.eval()

        batch_size = 2
        seq_len = 5

        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        kv_caches = []

        with torch.no_grad():
            outputs = wrapper(input_ids, positions, kv_caches)

        self.assertEqual(outputs.shape[0], batch_size)
        self.assertEqual(outputs.shape[1], seq_len)
        self.assertEqual(outputs.shape[2], self.config.vocab_size)

    def test_export_for_vllm(self):
        """Test vLLM export function."""
        from better_ai.inference.vllm_compat import export_for_vllm

        output_dir = os.path.join(self.temp_dir, "vllm_export")

        export_for_vllm(
            model_path=self.checkpoint_path,
            output_dir=output_dir,
            config_path=self.config_path,
            tensor_parallel_size=1,
        )

        # Check config was saved
        config_path = os.path.join(output_dir, "config.json")
        self.assertTrue(os.path.exists(config_path))

        with open(config_path, "r") as f:
            saved_config = json.load(f)

        self.assertEqual(saved_config["architecture"], "DeepSeekForCausalLM")
        self.assertIn("better_ai_config", saved_config)

        # Check weights were saved
        weights_path = os.path.join(output_dir, "model.pt")
        self.assertTrue(os.path.exists(weights_path))

    def test_sharded_weight_export(self):
        """Test tensor-parallel sharded weight export."""
        from better_ai.inference.vllm_compat import export_for_vllm

        output_dir = os.path.join(self.temp_dir, "vllm_sharded")

        export_for_vllm(
            model_path=self.checkpoint_path,
            output_dir=output_dir,
            config_path=self.config_path,
            tensor_parallel_size=2,
        )

        # Check shards were created
        shard_0_path = os.path.join(output_dir, "model_shard_0.pt")
        shard_1_path = os.path.join(output_dir, "model_shard_1.pt")

        self.assertTrue(os.path.exists(shard_0_path))
        self.assertTrue(os.path.exists(shard_1_path))

        # Load and verify shards
        shard_0 = torch.load(shard_0_path, map_location="cpu")
        shard_1 = torch.load(shard_1_path, map_location="cpu")

        self.assertIn("model_state_dict", shard_0)
        self.assertIn("model_state_dict", shard_1)


class TestExportAdvancedFeatures(unittest.TestCase):
    """Test export with all advanced features enabled."""

    def setUp(self):
        """Set up test fixtures with all advanced features."""
        self.temp_dir = tempfile.mkdtemp()

        self.config = ModelConfig.get_small_model_config()
        # Enable all advanced features
        self.config.use_recursive_scratchpad = True
        self.config.use_tidar = True
        self.config.use_cot_specialization = True
        self.config.use_inner_monologue = True
        self.config.use_star = True
        self.config.use_tool_heads = True
        self.config.use_json_db_ops_head = True
        self.config.use_math_reasoning_head = True
        self.config.use_algorithm_head = True
        self.config.use_grammar_constraints = True
        self.config.use_entropic_steering = True
        self.config.use_reward_models = True
        self.config.use_reasoning_rewards = True
        self.config.use_value_head = True

        self.model = DeepSeekModel(self.config)

        self.checkpoint_path = os.path.join(self.temp_dir, "test_model_advanced.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config.to_dict(),
            },
            self.checkpoint_path,
        )

    def test_advanced_features_in_state_dict(self):
        """Test that advanced feature tensors are in state dict."""
        state_dict = self.model.state_dict()

        # Check for advanced feature tensors
        advanced_prefixes = [
            "scratchpad",
            "tidar",
            "cot_heads",
            "inner_monologue",
            "star",
            "tool_heads",
            "json_db_ops_head",
            "math_reasoning_head",
            "algorithm_head",
            "reward_model",
            "entropic_steering",
            "value_head",
        ]

        found_features = set()
        for key in state_dict.keys():
            for prefix in advanced_prefixes:
                if key.startswith(prefix):
                    found_features.add(prefix)
                    break

        # At least some advanced features should be present
        self.assertGreater(len(found_features), 0, "No advanced feature tensors found")

    def test_vllm_wrapper_with_advanced_features(self):
        """Test vLLM wrapper with advanced features."""
        from better_ai.inference.vllm_compat import VLLMDeepSeekModel

        wrapper = VLLMDeepSeekModel(self.config)
        wrapper.eval()

        input_ids = torch.randint(0, self.config.vocab_size, (1, 5))
        positions = torch.arange(5).unsqueeze(0)

        with torch.no_grad():
            outputs = wrapper(input_ids, positions, [], use_advanced_features=True)

        self.assertEqual(outputs.shape[-1], self.config.vocab_size)

        # Check that advanced features were cached
        features = wrapper.get_advanced_features()
        self.assertIsInstance(features, dict)


class TestModelCardGeneration(unittest.TestCase):
    """Test model card generation for HuggingFace Hub."""

    def test_generate_model_card_gguf(self):
        """Test model card generation for GGUF format."""
        from scripts.model_card_utils import generate_model_card

        config = ModelConfig(
            hidden_dim=4096,
            num_layers=32,
            num_attention_heads=32,
            vocab_size=64000,
            use_recursive_scratchpad=True,
            use_tidar=True,
        )

        model_card = generate_model_card(
            model_name="test-model",
            export_format="GGUF",
            config=config,
            repository_url="https://github.com/anomalyco/better-ai",
            tags=["test"],
        )

        # Check model card content
        self.assertIn("test-model", model_card)
        self.assertIn("GGUF", model_card)
        self.assertIn("https://github.com/anomalyco/better-ai", model_card)
        self.assertIn("DeepSeek-based Transformer", model_card)
        self.assertIn("4096", model_card)  # Hidden size
        self.assertIn("32", model_card)  # Layers
        self.assertIn("test", model_card)  # Custom tag

    def test_generate_model_card_vllm(self):
        """Test model card generation for vLLM format."""
        from scripts.model_card_utils import generate_model_card

        config = ModelConfig(
            hidden_dim=2048,
            num_layers=16,
            use_star=True,
            use_tool_heads=True,
        )

        model_card = generate_model_card(
            model_name="test-vllm-model",
            export_format="vLLM",
            config=config,
            repository_url="https://github.com/anomalyco/better-ai",
        )

        # Check model card content
        self.assertIn("test-vllm-model", model_card)
        self.assertIn("vLLM", model_card)
        self.assertIn("vLLM", model_card)
        self.assertIn("Advanced Features", model_card)

    def test_model_card_advanced_features(self):
        """Test that advanced features are listed in model card."""
        from scripts.model_card_utils import generate_model_card

        config = ModelConfig(
            use_recursive_scratchpad=True,
            use_tidar=True,
            use_cot_specialization=True,
            use_star=True,
            use_tool_heads=True,
            use_entropic_steering=True,
            use_reward_models=True,
        )

        model_card = generate_model_card(
            model_name="advanced-model",
            export_format="GGUF",
            config=config,
        )

        # Check that advanced features are mentioned
        self.assertIn("Recursive Scratchpad", model_card)
        self.assertIn("TiDAR", model_card)
        self.assertIn("Chain-of-Thought Specialization", model_card)
        self.assertIn("STaR", model_card)
        self.assertIn("Tool Use", model_card)

    def test_model_card_moe_tags(self):
        """Test that MoE models get appropriate tags."""
        from scripts.model_card_utils import generate_model_card

        config = ModelConfig(num_experts=8, num_experts_per_token=2)

        model_card = generate_model_card(
            model_name="moe-model",
            export_format="GGUF",
            config=config,
        )

        # Check MoE-related content
        self.assertIn("mixture-of-experts", model_card)
        self.assertIn("moe", model_card)

    def test_save_model_card(self):
        """Test saving model card to file."""
        from scripts.model_card_utils import generate_model_card, save_model_card

        model_card = generate_model_card(
            model_name="test-save",
            export_format="GGUF",
        )

        temp_file = tempfile.mktemp(suffix=".md")
        save_model_card(model_card, temp_file)

        self.assertTrue(os.path.exists(temp_file))

        with open(temp_file, "r") as f:
            content = f.read()

        self.assertEqual(content, model_card)

        os.unlink(temp_file)


class TestHuggingFacePush(unittest.TestCase):
    """Test HuggingFace Hub push functionality (mocked)."""

    def test_push_to_huggingface_params(self):
        """Test that push_to_huggingface accepts correct parameters."""
        from scripts.convert_to_gguf import push_to_huggingface

        # Test function signature accepts all expected params
        import inspect

        sig = inspect.signature(push_to_huggingface)
        params = list(sig.parameters.keys())

        expected_params = [
            "gguf_path",
            "repo_id",
            "commit_message",
            "private",
            "token",
            "config",
            "repository_url",
        ]

        for param in expected_params:
            self.assertIn(param, params)


if __name__ == "__main__":
    unittest.main()
