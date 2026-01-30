"""
Machine Feedback (MF-RLHF) Pipeline for Software Repair
Uses deterministic tools (linter, grammar checker, compiler) to provide rewards.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import re

class MachineFeedbackReward:
    """
    Computes rewards based on deterministic code analysis tools.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.grammar_weight = self.config.get("grammar_weight", 0.3)
        self.linter_weight = self.config.get("linter_weight", 0.3)
        self.compiler_weight = self.config.get("compiler_weight", 0.4)

    def check_grammar(self, code: str) -> float:
        """
        Check for basic syntax/grammar errors (e.g., balanced brackets, quotes).
        """
        score = 1.0
        # Simple balanced brackets check
        for open_b, close_b in [('(', ')'), ('[', ']'), ('{', '}')]:
            if code.count(open_b) != code.count(close_b):
                score -= 0.2

        # Check for unclosed quotes
        if (code.count("'") % 2 != 0) or (code.count('"') % 2 != 0):
            score -= 0.2

        return max(0.0, score)

    def run_linter(self, code: str) -> float:
        """
        Check for common code style and potential bug patterns.
        """
        score = 1.0
        # Check for common "bad" patterns in Python/C-like languages
        if "pass" in code and len(code.split('\n')) > 10:
            score -= 0.1

        # Check for undefined variables (very simplified)
        # In a real prod environment, we'd use pylint or flake8

        # Check for indentation consistency (simplified)
        lines = code.split('\n')
        indentations = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
        if len(set(i % 4 for i in indentations if i > 0)) > 1:
            score -= 0.2

        return max(0.0, score)

    def run_compilation(self, code: str) -> float:
        """
        Check if the code compiles/parses correctly.
        """
        try:
            compile(code, '<string>', 'exec')
            return 1.0
        except SyntaxError:
            return 0.0
        except Exception:
            return 0.5

    def compute_reward(self, code: str) -> float:
        """
        Compute aggregate machine feedback reward.
        """
        # Extract code block if it's in markdown
        code_match = re.search(r'```(?:python|py)?\n(.*?)\n```', code, re.DOTALL)
        if code_match:
            code = code_match.group(1)

        grammar_score = self.check_grammar(code)
        linter_score = self.run_linter(code)
        compiler_score = self.run_compilation(code)

        total_reward = (
            self.grammar_weight * grammar_score +
            self.linter_weight * linter_score +
            self.compiler_weight * compiler_score
        )
        return total_reward

class MachineFeedbackTrainer:
    """
    Trainer that uses MachineFeedbackReward for RLHF.
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.reward_engine = MachineFeedbackReward(config)
        self.tokenizer = getattr(model, "tokenizer", None)
        self.config = config

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform a training step using machine feedback as reward.
        """
        # 1. Generate responses (using KV-cache reuse if available)
        input_ids = batch['input_ids']
        if hasattr(self.model, "generate_group"):
            responses = self.model.generate_group(
                input_ids,
                group_size=self.config.get("group_size", 4),
                max_new_tokens=self.config.get("max_new_tokens", 128)
            )
        else:
            responses = self.model.generate(input_ids, max_new_tokens=128)

        # 2. Decode and get rewards
        rewards = []
        for i in range(responses.size(0)):
            text = self.tokenizer.decode(responses[i], skip_special_tokens=True) if self.tokenizer else str(responses[i])
            reward = self.reward_engine.compute_reward(text)
            rewards.append(reward)

        # 3. Use rewards for GRPO update (simplified)
        reward_tensor = torch.tensor(rewards, device=input_ids.device)

        # Here we would normally perform the GRPO backward pass
        # For this pipeline, we just return the metrics

        return {
            "mf_reward_mean": reward_tensor.mean().item(),
            "mf_reward_std": reward_tensor.std().item() if len(rewards) > 1 else 0.0
        }
