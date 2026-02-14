import unittest
import torch
import torch.nn as nn
from better_ai.inference.quantization import Quantizer, apply_weight_only_quantization
from better_ai.models.core import DeepSeekModel
from better_ai.config import ModelConfig
from better_ai.test_resource_tags import low_resource

@low_resource
class TestQuantization(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig.get_small_model_config()
        self.model = DeepSeekModel(self.config)

    def test_weight_only_quantization(self):
        """Test that weight-only quantization logic doesn't crash"""
        # Save a sample weight to compare later if needed
        original_weight = self.model.layers[0].mlp.gate_proj.weight.clone()

        # Apply quantization
        apply_weight_only_quantization(self.model, bits=8)

        # In a real dynamic quant, weights might be replaced or just scaled
        # Current implementation in quantization.py is a placeholder that prints
        # but let's see if we can make it do something testable
        pass

    def test_int8_quantization_stub(self):
        """Test that INT8 quantization stub doesn't crash"""
        try:
            quantized_model = Quantizer.quantize_to_int8(self.model)
            # This uses torch.quantization.quantize_dynamic which should work on CPU
            self.assertIsNotNone(quantized_model)
        except Exception as e:
            # Depending on torch version/platform, this might fail, but let's catch it
            self.fail(f"Quantization failed with error: {e}")

    def test_int4_quantization_stub(self):
        """Test that INT4 quantization stub doesn't crash"""
        quantized_model = Quantizer.quantize_to_int4_stub(self.model)
        self.assertIsNotNone(quantized_model)

if __name__ == "__main__":
    unittest.main()
