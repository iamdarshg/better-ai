"""
Integration tests for all RLHF components
Tests core functionality of BR-RM, GRPO, and advanced features
"""

import os
import torch
import unittest
import sys
import subprocess
sys.path.append(".")
from better_ai.config import ModelConfig
from better_ai.models.enhanced_model import EnhancedDeepSeekModel
from better_ai.models.reward_model import BranchRewardModel, MultiAttributeRewardModel
from better_ai.training.grpo import GRPOTrainer, GRPOLoss
from better_ai.models.advanced_features import (
    RecursiveScratchpad,
    CoTSpecializationHeads,
    InnerMonologue,
    STaRModule,
    ToolUseHeads,
    GBNFConstraint,
    JSONEnforcer,
    EntropicSteering,
)

class TestBranchRewardModel(unittest.TestCase):
    """Integration tests for the BranchRewardModel."""
    
    def setUp(self):
        """Set up the test environment."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.model = BranchRewardModel(self.config).to(self.device)
    
    def test_forward_pass(self):
        """Test the forward pass of the reward model with both sequence and pooled inputs."""
        batch_size = 1
        seq_len = 64
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_dim).to(self.device)
        
        # Test with sequence
        scores = self.model(hidden_states)
        self.assertEqual(scores.shape, (batch_size,))
        
        # Test with pool
        pooled = hidden_states[:, -1, :]
        scores = self.model(pooled)
        self.assertEqual(scores.shape, (batch_size,))
    
    def test_branch_scores(self):
        """Test that the reward model returns scores for all branches."""
        batch_size = 2
        hidden_states = torch.randn(batch_size, self.config.hidden_dim).to(self.device)
        
        scores, branches = self.model(hidden_states, return_branch_scores=True)
        
        self.assertEqual(scores.shape, (batch_size,))
        self.assertIn("correctness", branches)
        self.assertIn("efficiency", branches)
        self.assertIn("readability", branches)
        self.assertIn("robustness", branches)
        self.assertIn("branch_weights", branches)
    
    def test_pair_scoring(self):
        """Test the scoring of preference pairs."""
        batch_size = 2
        hidden_states = torch.randn(batch_size, self.config.hidden_dim).to(self.device)
        
        chosen_scores, rejected_scores = self.model.score_pair(
            torch.randn(batch_size, self.config.hidden_dim),
            torch.randn(batch_size, self.config.hidden_dim),
        )
        
        self.assertEqual(chosen_scores.shape, (batch_size,))
        self.assertEqual(rejected_scores.shape, (batch_size,))


class TestMultiAttributeRewardModel(unittest.TestCase):
    """Integration tests for the MultiAttributeRewardModel."""
    
    def setUp(self):
        """Set up the test environment."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.model = MultiAttributeRewardModel(self.config, num_attributes=5).to(self.device)
    
    def test_forward_pass(self):
        """Test the forward pass of the multi-attribute reward model."""
        batch_size = 2
        hidden_states = torch.randn(batch_size, self.config.hidden_dim).to(self.device)
        
        results = self.model(hidden_states)
        
        self.assertIn("correctness", results)
        self.assertIn("efficiency", results)
        self.assertIn("readability", results)
        self.assertIn("robustness", results)
        self.assertIn("creativity", results)
        self.assertIn("point_estimates", results)


class TestGRPOTrainer(unittest.TestCase):
    """Integration tests for the GRPOTrainer."""
    
    def setUp(self):
        """Set up the test environment."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.model = EnhancedDeepSeekModel(self.config).to(self.device)
        self.reward_model = BranchRewardModel(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        self.grpo_config = {
            "beta": 0.01,
            "group_size": 4,
            "device": self.device,
            "hidden_dim": self.config.hidden_dim,
        }
    
    def test_advantage_computation(self):
        """Test the computation of group advantages."""
        trainer = GRPOTrainer(self.model, self.reward_model, self.optimizer, self.grpo_config)
        
        batch_size = 2
        group_size = 2
        
        rewards = torch.randn(batch_size, group_size)
        logprobs = torch.randn(batch_size, group_size)
        values = torch.randn(batch_size, group_size)
        
        advantages, returns, norm_advantages = trainer.compute_group_advantages(
            rewards, logprobs, values
        )
        
        self.assertEqual(advantages.shape, (batch_size, group_size))
        self.assertEqual(returns.shape, (batch_size, group_size))
        self.assertEqual(norm_advantages.shape, (batch_size, group_size))


class TestGRPOLoss(unittest.TestCase):
    """Unit tests for the GRPOLoss function."""
    
    def test_loss_computation(self):
        """Test the computation of the GRPO loss."""
        loss_fn = GRPOLoss(beta=0.01, eps_clip=0.2)
        
        batch_size = 2
        old_logprobs = torch.randn(batch_size, requires_grad=False)
        new_logprobs = torch.randn(batch_size, requires_grad=True)
        advantages = torch.randn(batch_size)
        
        loss = loss_fn(old_logprobs, new_logprobs, advantages)
        
        self.assertNotEqual(loss.item(), 0)
        loss.backward()
        self.assertIsNotNone(new_logprobs.grad)


class TestRecursiveScratchpad(unittest.TestCase):
    """Integration tests for the RecursiveScratchpad module."""
    
    def setUp(self):
        """Set up the test environment."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.module = RecursiveScratchpad(
            self.config.hidden_dim,
            max_iterations=2,
            scratchpad_dim=64,
        ).to(self.device)
    
    def test_forward_pass(self):
        """Test the forward pass of the RecursiveScratchpad."""
        batch_size = 2
        seq_len = 32
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_dim).to(self.device)
        
        outputs = self.module(hidden_states)
        
        self.assertEqual(outputs["scratchpad_output"].shape, (batch_size, seq_len, self.config.hidden_dim))
        self.assertGreater(outputs["iteration_count"], 0)


class TestCoTSpecializationHeads(unittest.TestCase):
    """Integration tests for the CoTSpecializationHeads module."""
    
    def setUp(self):
        """Set up the test environment."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.module = CoTSpecializationHeads(
            self.config.hidden_dim,
            num_cot_heads=2,
        ).to(self.device)
    
    def test_forward_pass(self):
        """Test the forward pass of the CoTSpecializationHeads."""
        batch_size = 2
        seq_len = 32
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_dim).to(self.device)
        
        outputs = self.module(hidden_states, is_reasoning_phase=True)
        
        self.assertEqual(outputs["cot_output"].shape, (batch_size, seq_len, self.config.hidden_dim))
        self.assertEqual(outputs["final_output"].shape, (batch_size, seq_len, self.config.hidden_dim))


class TestToolUseHeads(unittest.TestCase):
    """Integration tests for the ToolUseHeads module."""
    
    def setUp(self):
        """Set up the test environment."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.module = ToolUseHeads(
            self.config.hidden_dim,
            tool_vocab_size=self.config.tool_vocab_size
        ).to(self.device)
    
    def test_forward_pass(self):
        """Test the forward pass of the ToolUseHeads."""
        batch_size = 2
        hidden_states = torch.randn(batch_size, self.config.hidden_dim).to(self.device)
        try:
            outputs = self.module.forward(hidden_states)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"ToolUseHeads forward pass failed with exception: {e}")
        self.assertEqual(outputs["tool_logits"].shape, (batch_size, self.config.tool_vocab_size))
        self.assertEqual(outputs["mode_score"].shape, (batch_size, 1))
        self.assertEqual(outputs["confidence"].shape, (batch_size, 1))


class TestEnhancedModel(unittest.TestCase):
    """Integration tests for the EnhancedDeepSeekModel."""
    
    def setUp(self):
        """Set up the test environment."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.model = EnhancedDeepSeekModel(self.config).to(self.device)
    
    def test_forward_pass(self):
        """Test the basic forward pass of the model."""
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        outputs = self.model(input_ids, return_advanced_features=False)
        
        self.assertEqual(outputs["logits"].shape, (batch_size, seq_len, self.config.vocab_size))
    
    def test_advanced_features(self):
        """Test the forward pass with all advanced features enabled."""
        batch_size = 1
        seq_len = 16
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        try:
            outputs = self.model.forward(input_ids, return_advanced_features=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"EnhancedDeepSeekModel forward pass with advanced features failed with exception: {e}")
        self.assertIn("advanced_features", outputs)
        advanced = outputs["advanced_features"]
        
        if self.config.use_recursive_scratchpad:
            self.assertIn("scratchpad", advanced)
        
        if self.config.use_tool_heads:
            self.assertIn("tool_use", advanced)
        
        self.assertIn("reward", advanced)
    
    def test_loss_computation(self):
        """Test the loss computation of the model."""
        batch_size = 1
        seq_len = 16
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        labels = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        try:
            losses = self.model.compute_loss(input_ids, labels)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"EnhancedDeepSeekModel loss computation failed with exception: {e}")    
        self.assertIn("lm_loss", losses)
        self.assertIn("total_loss", losses)
        self.assertGreater(losses["lm_loss"].item(), 0)


class TestEntropyMonitoring(unittest.TestCase):
    """Integration tests for the EntropicSteering module."""
    
    def setUp(self):
        """Set up the test environment."""
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.module = EntropicSteering(self.config.hidden_dim).to(self.device)
    
    def test_entropy_computation(self):
        """Test the entropy computation and spike detection."""
        batch_size = 1
        seq_len = 16
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_dim).to(self.device)

        # Create a high-entropy distribution to trigger a spike
        logits = torch.ones(batch_size, seq_len, self.config.vocab_size).to(self.device)
        
        outputs = self.module.forward(hidden_states, logits)
        
        self.assertEqual(outputs["entropy_scores"].shape, (batch_size, seq_len))
        self.assertEqual(outputs["spike_detected"].shape, (batch_size, seq_len))
        self.assertTrue(outputs["spike_detected"].any(), "No entropy spike was detected")


class TestWorkflow(unittest.TestCase):
    """End-to-end tests for the main training workflow."""
    
    def setUp(self):
        """Set up the test environment."""
        pass
    
    def test_workflow_whole(self):
        """Test the entire training workflow from start to finish."""
        n = subprocess.run(
            ["python", "train_enhanced.py", "--stage", "full", "--test", "--batch-size", "1", "--max-steps", "1"],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
            capture_output=True, text=True
        )
        if n.returncode != 0:
            self.fail(f"Workflow test failed because of stderr- {n.stderr} \n and stdout- {n.stdout}")
        elif "Error" in str(n.stdout):
            self.fail(f"Workflow test failed because of stdout- {n.stdout} \n and stderr- {n.stderr}")


def run_tests():
    """Run all tests"""
    import re
    
    with open("better_ai/config.py", "r") as f:
        config_code = f.read()
    
    # Define scaling ratios for test mode (much smaller for CI)
    scaling_rules = {
        r'hidden_dim:\s*int\s*=\s*(\d+)': lambda m: f'hidden_dim: int = {max(32, int(m.group(1)) // 64)}',
        r'num_layers:\s*int\s*=\s*(\d+)': lambda m: f'num_layers: int = {max(1, int(m.group(1)) // 60)}',
        r'num_attention_heads:\s*int\s*=\s*(\d+)': lambda m: f'num_attention_heads: int = {max(1, int(m.group(1)) // 8)}',
        r'num_key_value_heads:\s*Optional\[int\]\s*=\s*(\d+)': lambda m: f'num_key_value_heads: Optional[int] = {max(1, int(m.group(1)) // 8)}',
        r'intermediate_dim:\s*int\s*=\s*(\d+)': lambda m: f'intermediate_dim: int = {max(32, int(m.group(1)) // 48)}',
        r'vocab_size:\s*int\s*=\s*(\d+)': lambda m: f'vocab_size: int = {max(256, int(m.group(1)) // 100)}',
        r'max_seq_length:\s*int\s*=\s*(\d+)': lambda m: f'max_seq_length: int = {max(32, int(m.group(1)) // 256)}',
        r'cot_num_heads:\s*int\s*=\s*(\d+)': lambda m: f'cot_num_heads: int = {max(1, int(m.group(1)) // 12)}',
        r'tool_vocab_size:\s*int\s*=\s*(\d+)': lambda m: f'tool_vocab_size: int = {max(32, int(m.group(1)) // 50)}',
        r'tool_hidden_dim:\s*int\s*=\s*(\d+)': lambda m: f'tool_hidden_dim: int = {max(32, int(m.group(1)) // 24)}',
        r'scratchpad_hidden_dim:\s*int\s*=\s*(\d+)': lambda m: f'scratchpad_hidden_dim: int = {max(32, int(m.group(1)) // 64)}',
        r'warmup_steps:\s*int\s*=\s*(\d+)': lambda m: f'warmup_steps: int = {max(1, int(m.group(1)) // 10000)}',
        r'max_steps:\s*int\s*=\s*(\d+)': lambda m: f'max_steps: int = {max(1, int(m.group(1)) // 5000)}',
        r'save_steps:\s*int\s*=\s*(\d+)': lambda m: f'save_steps: int = {max(1, int(m.group(1)) // 1000)}',
        r'eval_steps:\s*int\s*=\s*(\d+)': lambda m: f'eval_steps: int = {max(1, int(m.group(1)) // 1000)}',
    }
    
    # Apply all scaling rules dynamically
    scaled_config = config_code
    for pattern, replacement_fn in scaling_rules.items():
        scaled_config = re.sub(pattern, replacement_fn, scaled_config)
    
    with open("better_ai/config.py", "w") as f:
        f.write(scaled_config)
    
    try:
        unittest.main(argv=['--locals'], exit=False, verbosity=2)
        with open("better_ai/config.py", "w") as f:
            f.write(config_code)
    except Exception as e:
        with open("better_ai/config.py", "w") as f:
            f.write(config_code)
        raise e
    except SystemExit as e:
        with open("better_ai/config.py", "w") as f:
            f.write(config_code)
        if e.code != 0:
            raise e
    except KeyboardInterrupt:
        with open("better_ai/config.py", "w") as f:
            f.write(config_code)
        raise KeyboardInterrupt


if __name__ == "__main__":
    run_tests()
