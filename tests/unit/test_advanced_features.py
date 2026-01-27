
import unittest
import torch
from better_ai.models.enhanced_model import EnhancedDeepSeekModel
from better_ai.config import ModelConfig
from tests.test_config_utils import get_small_model_config

class TestAdvancedFeatures(unittest.TestCase):

    def test_specialized_head_creation(self):
        config = get_small_model_config()
        config.use_json_db_ops_head = True
        config.json_db_ops_ratio = 0.2
        model = EnhancedDeepSeekModel(config)
        self.assertTrue(hasattr(model, 'json_db_ops_head'))
        self.assertEqual(model.json_db_ops_head.ratio, 0.2)

        config = get_small_model_config()
        config.use_math_reasoning_head = True
        config.math_reasoning_ratio = 0.3
        model = EnhancedDeepSeekModel(config)
        self.assertTrue(hasattr(model, 'math_reasoning_head'))
        self.assertEqual(model.math_reasoning_head.ratio, 0.3)

        config = get_small_model_config()
        config.use_algorithm_head = True
        config.algorithm_ratio = 0.4
        model = EnhancedDeepSeekModel(config)
        self.assertTrue(hasattr(model, 'algorithm_head'))
        self.assertEqual(model.algorithm_head.ratio, 0.4)

    def test_head_forward_pass(self):
        config = get_small_model_config()
        config.use_json_db_ops_head = True
        config.use_math_reasoning_head = True
        config.use_algorithm_head = True
        model = EnhancedDeepSeekModel(config)
        input_ids = torch.randint(0, 100, (1, 10))
        outputs = model.forward(input_ids, return_advanced_features=True)
        self.assertIn('json_db_ops_head', outputs['advanced_features'])
        self.assertIn('math_reasoning_head', outputs['advanced_features'])
        self.assertIn('algorithm_head', outputs['advanced_features'])

    def test_ring_attention_creation(self):
        config = get_small_model_config()
        config.use_ring_attention = True
        model = EnhancedDeepSeekModel(config)
        from better_ai.models.ring_attention import RingAttention
        for layer in model.model.layers:
            self.assertTrue(isinstance(layer.self_attn, RingAttention))

        config = get_small_model_config()
        config.use_ring_attention = False
        model = EnhancedDeepSeekModel(config)
        from better_ai.models.core import MultiHeadAttention
        for layer in model.model.layers:
            self.assertTrue(isinstance(layer.self_attn, MultiHeadAttention))

if __name__ == '__main__':
    unittest.main()
