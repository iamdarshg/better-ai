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
    """Test BR-RM functionality"""
    
    def setUp(self):
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.model = BranchRewardModel(self.config).to(self.device)
    
    def test_forward_pass(self):
        """Test basic forward pass"""
        batch_size = 4
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_dim).to(self.device)
        
        # Test with sequence
        scores = self.model(hidden_states)
        self.assertEqual(scores.shape, (batch_size,))
        
        # Test with pool
        pooled = hidden_states[:, -1, :]
        scores = self.model(pooled)
        self.assertEqual(scores.shape, (batch_size,))
    
    def test_branch_scores(self):
        """Test branch scoring"""
        batch_size = 4
        hidden_states = torch.randn(batch_size, self.config.hidden_dim).to(self.device)
        
        scores, branches = self.model(hidden_states, return_branch_scores=True)
        
        self.assertEqual(scores.shape, (batch_size,))
        self.assertIn("correctness", branches)
        self.assertIn("efficiency", branches)
        self.assertIn("readability", branches)
        self.assertIn("robustness", branches)
        self.assertIn("branch_weights", branches)
    
    def test_pair_scoring(self):
        """Test preference pair scoring"""
        batch_size = 4
        hidden_states = torch.randn(batch_size, self.config.hidden_dim).to(self.device)
        
        chosen_scores, rejected_scores = self.model.score_pair(
            torch.randn(batch_size, self.config.hidden_dim),
            torch.randn(batch_size, self.config.hidden_dim),
        )
        
        self.assertEqual(chosen_scores.shape, (batch_size,))
        self.assertEqual(rejected_scores.shape, (batch_size,))


class TestMultiAttributeRewardModel(unittest.TestCase):
    """Test multi-attribute reward model"""
    
    def setUp(self):
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.model = MultiAttributeRewardModel(self.config, num_attributes=5).to(self.device)
    
    def test_forward_pass(self):
        """Test multi-attribute forward pass"""
        batch_size = 4
        hidden_states = torch.randn(batch_size, self.config.hidden_dim).to(self.device)
        
        results = self.model(hidden_states)
        
        self.assertIn("correctness", results)
        self.assertIn("efficiency", results)
        self.assertIn("readability", results)
        self.assertIn("robustness", results)
        self.assertIn("creativity", results)
        self.assertIn("point_estimates", results)


class TestGRPOTrainer(unittest.TestCase):
    """Test GRPO training"""
    
    def setUp(self):
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
        """Test group advantage estimation"""
        trainer = GRPOTrainer(self.model, self.reward_model, self.optimizer, self.grpo_config)
        
        batch_size = 4
        group_size = 4
        
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
    """Test GRPO loss computation"""
    
    def test_loss_computation(self):
        """Test GRPO loss"""
        loss_fn = GRPOLoss(beta=0.01, eps_clip=0.2)
        
        batch_size = 4
        old_logprobs = torch.randn(batch_size, requires_grad=False)
        new_logprobs = torch.randn(batch_size, requires_grad=True)
        advantages = torch.randn(batch_size)
        
        loss = loss_fn(old_logprobs, new_logprobs, advantages)
        
        self.assertNotEqual(loss.item(), 0)
        loss.backward()
        self.assertIsNotNone(new_logprobs.grad)


class TestRecursiveScratchpad(unittest.TestCase):
    """Test recursive scratchpad"""
    
    def setUp(self):
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.module = RecursiveScratchpad(
            self.config.hidden_dim,
            max_iterations=5,
            scratchpad_dim=256,
        ).to(self.device)
    
    def test_forward_pass(self):
        """Test scratchpad processing"""
        batch_size = 4
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_dim).to(self.device)
        
        outputs = self.module(hidden_states)
        
        self.assertEqual(outputs["scratchpad_output"].shape, (batch_size, seq_len, self.config.hidden_dim))
        self.assertGreater(outputs["iteration_count"], 0)


class TestCoTSpecializationHeads(unittest.TestCase):
    """Test CoT specialization"""
    
    def setUp(self):
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.module = CoTSpecializationHeads(
            self.config.hidden_dim,
            num_cot_heads=4,
        ).to(self.device)
    
    def test_forward_pass(self):
        """Test CoT heads"""
        batch_size = 4
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_dim).to(self.device)
        
        outputs = self.module(hidden_states, is_reasoning_phase=True)
        
        self.assertEqual(outputs["cot_output"].shape, (batch_size, seq_len, self.config.hidden_dim))
        self.assertEqual(outputs["final_output"].shape, (batch_size, seq_len, self.config.hidden_dim))


class TestToolUseHeads(unittest.TestCase):
    """Test tool-use heads"""
    
    def setUp(self):
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.module = ToolUseHeads(
            self.config.hidden_dim,
            tool_vocab_size=self.config.tool_vocab_size
        ).to(self.device)
    
    def test_forward_pass(self):
        """Test tool-use prediction"""
        batch_size = 4
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
    """Test integrated enhanced model"""
    
    def setUp(self):
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.model = EnhancedDeepSeekModel(self.config).to(self.device)
    
    def test_forward_pass(self):
        """Test basic forward pass"""
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        outputs = self.model(input_ids, return_advanced_features=False)
        
        self.assertEqual(outputs["logits"].shape, (batch_size, seq_len, self.config.vocab_size))
    
    def test_advanced_features(self):
        """Test with all advanced features"""
        batch_size = 2
        seq_len = 64
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
        """Test loss computation"""
        batch_size = 2
        seq_len = 64
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
    """Test entropic steering"""
    
    def setUp(self):
        self.device = torch.device("cpu")
        self.config = ModelConfig()
        self.module = EntropicSteering(self.config.hidden_dim).to(self.device)
    
    def test_entropy_computation(self):
        """Test entropy monitoring"""
        batch_size = 4
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_dim).to(self.device)

        # Create a high-entropy distribution to trigger a spike
        logits = torch.ones(batch_size, seq_len, self.config.vocab_size).to(self.device)
        
        outputs = self.module.forward(hidden_states, logits)
        
        self.assertEqual(outputs["entropy_scores"].shape, (batch_size, seq_len))
        self.assertEqual(outputs["spike_detected"].shape, (batch_size, seq_len))
        self.assertTrue(outputs["spike_detected"].any(), "No entropy spike was detected")


class TestWorkflow(unittest.TestCase):
    """Test main workflow file"""
    
    def setUp(self):
        pass
    
    def test_workflow_whole(self):
        """Test entropy monitoring"""
        n = subprocess.run(
            ["python", "train_enhanced.py", "--stage", "full", "--test", "--batch-size", "1", "--max-steps", "1"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True, text=True
        )
        if n.returncode != 0:
            self.fail(f"Workflow test failed because of stderr- {n.stderr} \n and stdout- {n.stdout}")
        elif "Error" in str(n.stdout):
            self.fail(f"Workflow test failed because of stdout- {n.stdout} \n and stderr- {n.stderr}")


def run_tests():
    """Run all tests"""
    with open("better_ai/config.py", "r") as f:
        config_code = f.read()
    with open("better_ai/config.py", "w") as f:
        f.write(config_code.replace(
            "hidden_dim: int = 2048", "hidden_dim: int = 128"
        ).replace(
            "num_layers: int = 30", "num_layers: int = 1"
        ).replace(
            "num_attention_heads: int = 32", "num_attention_heads: int = 8"
        ).replace(
            "num_key_value_heads: Optional[int] = 16", "num_key_value_heads: Optional[int] = 4"
        ).replace(
            "intermediate_dim: int = 6144", "intermediate_dim: int = 512"
        ).replace(
            "vocab_size: int = 64000", "vocab_size: int = 6400"
        ).replace(
            "max_seq_length: int = 16384", "max_seq_length: int = 256"
        ).replace(
            "cot_num_heads: int = 12", "cot_num_heads: int = 2"
        ).replace(
            "tool_vocab_size: int = 2000", "tool_vocab_size: int = 100"
        ).replace(
            "tool_hidden_dim: int = 768", "tool_hidden_dim: int = 64"
        ).replace(
            "scratchpad_hidden_dim: int = 2048", "scratchpad_hidden_dim: int = 128"
        ).replace(
            "warmup_steps: int = 1000", "warmup_steps: int = 1"
        ).replace(
            "max_steps: int = 100000", "max_steps: int = 100"
        ).replace(
            "save_steps: int = 1000", "save_steps: int = 10"
        ).replace(
            "eval_steps: int = 1000", "eval_steps: int = 10"
        ).replace(
            "max_seq_length: int = 16384", "max_seq_length: int = 256"
        )
        )
    try:
        unittest.main(argv=['--locals'], exit=False, verbosity=2)
        with open("better_ai/config.py", "w") as f:
            f.write(config_code)
    except Exception as e:
        with open("better_ai/config.py", "w") as f:
            f.write(config_code)
        raise e


if __name__ == "__main__":
    run_tests()
