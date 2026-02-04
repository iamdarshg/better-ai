"""
Machine Feedback (MF-RLHF) Pipeline for Software Repair
Uses deterministic tools (linter, grammar checker, compiler) to provide rewards.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import re
import ast

class MachineFeedbackReward:
    """
    Computes rewards based on deterministic code analysis tools.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.grammar_weight = self.config.get("grammar_weight", 0.3)
        self.linter_weight = self.config.get("linter_weight", 0.3)
        self.compiler_weight = self.config.get("compiler_weight", 0.4)

    def check_grammar(self, code: str, is_python: bool = True) -> float:
        """
        Check for basic syntax/grammar errors. Uses AST for Python.
        """
        score = 1.0

        if is_python:
            try:
                ast.parse(code)
            except SyntaxError as e:
                # Heavily penalize syntax errors
                lines = code.split('\n')
                error_line = getattr(e, 'lineno', 1)
                # Max score is 0.2 if there is a syntax error, lower if it happens early
                score = 0.2 * (error_line / max(len(lines), 1))

        # Check for balanced brackets (language independent)
        for open_b, close_b in [('(', ')'), ('[', ']'), ('{', '}')]:
            if code.count(open_b) != code.count(close_b):
                score = min(score, 0.1) # Even heavier penalty for unbalanced brackets

        return max(0.0, score)

    def run_linter(self, code: str, is_python: bool = True) -> float:
        """
        Check for common code style and potential bug patterns. Uses AST for Python.
        """
        if not is_python:
            return 1.0 # Skip linter for non-python for now

        score = 1.0
        try:
            tree = ast.parse(code)

            # 1. Check for undefined variables (very simplified using AST)
            defined_names = set()
            used_names = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    if isinstance(node.ctx, ast.Store):
                        defined_names.add(node.id)
                    elif isinstance(node.ctx, ast.Load):
                        used_names.add(node.id)
                elif isinstance(node, ast.FunctionDef):
                    defined_names.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    defined_names.add(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        defined_names.add(alias.asname or alias.name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        defined_names.add(alias.asname or alias.name)

            # Built-ins
            import builtins
            defined_names.update(dir(builtins))

            undefined = used_names - defined_names
            if undefined:
                score -= 0.1 * len(undefined)

            # 2. Check for empty except blocks
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler):
                    if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                        score -= 0.1

            # 3. Check for too many 'pass' statements (often indicates incomplete code)
            pass_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Pass))
            if pass_count > 5:
                score -= 0.05 * min(pass_count, 10)

        except:
            # If it doesn't parse, we can't run the AST linter
            score = 0.5

        # Check for indentation consistency
        lines = [line for line in code.split('\n') if line.strip()]
        if lines:
            indentations = [len(line) - len(line.lstrip()) for line in lines]
            # Heuristic: indentation should generally be multiples of 4 or 2
            inconsistent = [i for i in indentations if i % 2 != 0]
            if len(inconsistent) > len(lines) * 0.2:
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
        is_python = False
        # Extract code block if it's in markdown
        code_match = re.search(r'```(python|py|rust|c|cpp|java|js|javascript|go)?\n(.*?)\n```', code, re.DOTALL)
        if code_match:
            lang = code_match.group(1)
            is_python = lang in ["python", "py", None]
            code = code_match.group(2)
        else:
            # If no markdown, assume based on config or content
            is_python = self.config.get("grammar_type") == "python"

        grammar_score = self.check_grammar(code, is_python=is_python)
        linter_score = self.run_linter(code, is_python=is_python)

        if is_python:
            compiler_score = self.run_compilation(code)
        else:
            compiler_score = 1.0 # Assume valid if we don't have a compiler for it


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
    def __init__(self, model: nn.Module, config: Dict[str, Any], optimizer: Optional[torch.optim.Optimizer] = None):
        self.model = model
        self.reward_engine = MachineFeedbackReward(config)
        self.tokenizer = getattr(model, "tokenizer", None)
        self.optimizer = optimizer
        self.config = config

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform a training step using machine feedback as reward.
        """
        # 1. Generate responses
        input_ids = batch['input_ids']
        batch_size = input_ids.size(0)
        group_size = self.config.get("group_size", 4)

        # Ensure we are in eval mode for generation
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, "generate_group"):
                responses = self.model.generate_group(
                    input_ids,
                    group_size=group_size,
                    max_new_tokens=self.config.get("max_new_tokens", 128)
                )
            else:
                # Fallback to multiple generations
                all_responses = []
                max_len = 0
                for _ in range(group_size):
                    resp = self.model.generate(input_ids, max_new_tokens=128)
                    all_responses.append(resp)
                    max_len = max(max_len, resp.size(1))

                # Pad to same length
                padded_responses = []
                for resp in all_responses:
                    if resp.size(1) < max_len:
                        padding = torch.zeros((resp.size(0), max_len - resp.size(1)), device=resp.device, dtype=resp.dtype)
                        resp = torch.cat([resp, padding], dim=1)
                    padded_responses.append(resp)
                responses = torch.cat(padded_responses, dim=0)

        # 2. Decode and get rewards
        rewards = []
        for i in range(responses.size(0)):
            text = self.tokenizer.decode(responses[i], skip_special_tokens=True) if self.tokenizer else str(responses[i].tolist())
            reward = self.reward_engine.compute_reward(text)
            rewards.append(reward)

        reward_tensor = torch.tensor(rewards, device=input_ids.device)

        # 3. Perform Policy Optimization Step
        self.model.train()

        # Compute logprobs for generated responses
        # responses shape: [batch_size * group_size, seq_len]
        outputs = self.model(responses)
        logits = outputs["logits"]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Get logprobs of actually generated tokens
        # Shift responses to align with logits
        target_ids = responses[:, 1:].contiguous()
        log_probs = log_probs[:, :-1, :].contiguous()

        per_token_logprobs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        # Sum over sequence, then average over group
        # (Assuming we want to optimize the whole response)
        response_logprobs = per_token_logprobs.sum(dim=-1)

        # Advantage estimation (Group Relative)
        mean_reward = reward_tensor.mean()
        std_reward = reward_tensor.std() if len(rewards) > 1 else torch.tensor(1.0, device=input_ids.device)
        advantages = (reward_tensor - mean_reward) / (std_reward + 1e-8)

        # Policy Loss: -Advantage * log_prob
        loss = -(advantages.detach() * response_logprobs).mean()

        if self.optimizer:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return {
            "mf_loss": loss.item(),
            "mf_reward_mean": mean_reward.item(),
            "mf_reward_std": std_reward.item(),
            "mf_max_advantage": advantages.max().item()
        }
