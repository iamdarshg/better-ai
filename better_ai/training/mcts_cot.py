"""
Monte Carlo Tree Search for Chain-of-Thought Reasoning
Explores both reasoning traces and answer tokens simultaneously
"""

import math
import random
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, NamedTuple
from dataclasses import dataclass, field
import logging
import time
from collections import defaultdict, deque

from ..config import ModelConfig, InferenceConfig


@dataclass
class MCTSNode:
    """Node in MCTS search tree for CoT reasoning"""

    # Node state
    question: str
    reasoning_trace: List[str] = field(default_factory=list)
    partial_answer: str = ""
    depth: int = 0

    # MCTS statistics
    visit_count: int = 0
    total_value: float = 0.0
    prior_prob: float = 0.0

    # Tree structure
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)

    # Node type
    is_reasoning_node: bool = True  # True for reasoning, False for answer token
    is_terminal: bool = False
    is_expanded: bool = False

    # Additional metadata
    action_history: List[str] = field(default_factory=list)
    state_hash: Optional[str] = None

    def __post_init__(self):
        if self.state_hash is None:
            self.state_hash = self._compute_state_hash()

    def _compute_state_hash(self) -> str:
        """Compute hash for node state"""
        state_str = f"{self.question}|{'|'.join(self.reasoning_trace)}|{self.partial_answer}|{self.depth}"
        return str(hash(state_str))

    @property
    def is_leaf(self) -> bool:
        """Check if node is a leaf"""
        return len(self.children) == 0

    @property
    def mean_value(self) -> float:
        """Get mean value of node"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def uct_score(self) -> float:
        """Calculate UCT score for node selection"""
        if self.visit_count == 0:
            return float("inf")

        exploitation = self.mean_value
        exploration = (
            math.sqrt(2.0 * math.log(self.parent.visit_count) / self.visit_count)
            if self.parent
            else 0.0
        )

        return exploitation + exploration

    def add_child(self, child: "MCTSNode"):
        """Add child node"""
        child.parent = self
        self.children.append(child)

    def get_path_to_root(self) -> List["MCTSNode"]:
        """Get path from this node to root"""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))


@dataclass
class MCTSConfig:
    """Configuration for MCTS search"""

    # Search parameters
    max_iterations: int = 100
    max_depth: int = 10
    max_time_seconds: float = 30.0
    max_nodes: int = 1000

    # Expansion parameters
    max_children_per_node: int = 5
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9

    # Exploration vs exploitation
    exploration_constant: float = 1.414  # sqrt(2)
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # Reasoning vs answer balance
    reasoning_weight: float = 0.7
    answer_weight: float = 0.3
    min_reasoning_steps: int = 2
    max_reasoning_steps: int = 8

    # Evaluation
    rollout_depth: int = 5
    rollout_temperature: float = 0.8
    value_scale: float = 1.0

    # Pruning and optimization
    prune_low_value_nodes: bool = True
    value_threshold: float = 0.1
    cache_states: bool = True

    # Logging
    log_search_progress: bool = True
    log_frequency: int = 10


class MCTSCoTSearcher:
    """
    Monte Carlo Tree Search for Chain-of-Thought reasoning
    Explores both reasoning traces and answer tokens
    """

    def __init__(self, model, tokenizer, config: MCTSConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Search state
        self.root_node: Optional[MCTSNode] = None
        self.current_iteration = 0
        self.search_start_time = 0.0

        # Caching
        self.state_cache = {} if config.cache_states else None
        self.value_cache = {}

        # Statistics
        self.search_stats = {
            "total_nodes": 0,
            "reasoning_nodes": 0,
            "answer_nodes": 0,
            "rollouts_performed": 0,
            "cache_hits": 0,
            "pruned_nodes": 0,
        }

        logging.info("Initialized MCTS CoT Searcher")

    def search(self, question: str) -> Dict[str, Any]:
        """
        Perform MCTS search for best reasoning trace and answer
        """
        self.search_start_time = time.time()
        self.current_iteration = 0

        # Initialize root node
        self.root_node = MCTSNode(
            question=question,
            reasoning_trace=[],
            partial_answer="",
            depth=0,
            is_reasoning_node=True,
        )

        logging.info(f"Starting MCTS search for question: {question[:100]}...")

        # Main search loop
        while not self._should_stop_search():
            self.current_iteration += 1

            # 1. Selection
            selected_node = self._select_node(self.root_node)

            # 2. Expansion
            expanded_node = self._expand_node(selected_node)

            # 3. Simulation (Rollout)
            rollout_value = self._simulate_rollout(expanded_node)

            # 4. Backpropagation
            self._backpropagate_value(expanded_node, rollout_value)

            # Logging
            if (
                self.config.log_search_progress
                and self.current_iteration % self.config.log_frequency == 0
            ):
                self._log_search_progress()

        # Get best result
        best_result = self._get_best_result()

        # Final statistics
        search_time = time.time() - self.search_start_time
        final_stats = {
            "search_time": search_time,
            "iterations": self.current_iteration,
            "total_nodes": self.search_stats["total_nodes"],
            "tree_depth": self._get_max_tree_depth(),
            "best_value": best_result["value"],
            "cache_hit_rate": self.search_stats["cache_hits"]
            / max(1, self.search_stats["total_nodes"]),
        }

        logging.info(f"MCTS search completed: {final_stats}")

        return {
            "best_reasoning_trace": best_result["reasoning_trace"],
            "best_answer": best_result["answer"],
            "best_value": best_result["value"],
            "search_stats": final_stats,
            "tree_info": self._get_tree_info(),
        }

    def _should_stop_search(self) -> bool:
        """Check if search should stop"""
        # Time limit
        if time.time() - self.search_start_time > self.config.max_time_seconds:
            return True

        # Iteration limit
        if self.current_iteration >= self.config.max_iterations:
            return True

        # Node limit
        if self.search_stats["total_nodes"] >= self.config.max_nodes:
            return True

        return False

    def _select_node(self, node: MCTSNode) -> MCTSNode:
        """
        Select node for expansion using UCT policy
        """
        current = node

        while not current.is_terminal and current.is_expanded:
            # Select child with highest UCT score
            best_child = max(current.children, key=lambda child: child.uct_score)
            current = best_child

        return current

    def _expand_node(self, node: MCTSNode) -> MCTSNode:
        """
        Expand node by generating children
        """
        if node.is_terminal or node.is_expanded:
            return node

        # Determine if we should generate reasoning or answer
        should_generate_reasoning = self._should_generate_reasoning(node)

        if should_generate_reasoning:
            children = self._generate_reasoning_children(node)
        else:
            children = self._generate_answer_children(node)

        # Add children to node
        for child in children:
            node.add_child(child)
            self.search_stats["total_nodes"] += 1

            if child.is_reasoning_node:
                self.search_stats["reasoning_nodes"] += 1
            else:
                self.search_stats["answer_nodes"] += 1

        # Mark node as expanded
        node.is_expanded = True

        # Return random child for simulation
        return random.choice(children) if children else node

    def _should_generate_reasoning(self, node: MCTSNode) -> bool:
        """Determine if node should generate reasoning steps"""
        # Check minimum reasoning steps
        if len(node.reasoning_trace) < self.config.min_reasoning_steps:
            return True

        # Check maximum reasoning steps
        if len(node.reasoning_trace) >= self.config.max_reasoning_steps:
            return False

        # Probabilistic decision based on depth and weights
        reasoning_prob = self.config.reasoning_weight * (
            1.0 - node.depth / self.config.max_depth
        )
        return random.random() < reasoning_prob

    def _generate_reasoning_children(self, node: MCTSNode) -> List[MCTSNode]:
        """Generate children with reasoning steps"""
        # Build prompt for reasoning generation
        prompt = self._build_reasoning_prompt(node)

        # Generate reasoning steps
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            )
            if hasattr(inputs, "input_ids"):
                input_ids = inputs.input_ids
            else:
                input_ids = inputs["input_ids"]

            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                num_return_sequences=self.config.max_children_per_node,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                num_return_sequences=self.config.max_children_per_node,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Parse reasoning steps
        children = []
        for i, output in enumerate(outputs):
            if hasattr(inputs, "input_ids"):
                seq_len = inputs.input_ids.shape[1]
            else:
                seq_len = inputs["input_ids"].shape[1]

            generated_text = self.tokenizer.decode(
                output[seq_len:], skip_special_tokens=True
            )
            reasoning_step = self._extract_reasoning_step(generated_text)

            if reasoning_step:
                child = MCTSNode(
                    question=node.question,
                    reasoning_trace=node.reasoning_trace + [reasoning_step],
                    partial_answer=node.partial_answer,
                    depth=node.depth + 1,
                    is_reasoning_node=True,
                    prior_prob=1.0 / self.config.max_children_per_node,
                )
                children.append(child)

        return children

    def _generate_answer_children(self, node: MCTSNode) -> List[MCTSNode]:
        """Generate children with answer tokens"""
        # Build prompt for answer generation
        prompt = self._build_answer_prompt(node)

        # Generate answer tokens
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            )
            if hasattr(inputs, "input_ids"):
                input_ids = inputs.input_ids
            else:
                input_ids = inputs["input_ids"]

            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=50,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                num_return_sequences=self.config.max_children_per_node,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                num_return_sequences=self.config.max_children_per_node,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Parse answer tokens
        children = []
        for i, output in enumerate(outputs):
            generated_text = self.tokenizer.decode(
                output[inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            answer_part = self._extract_answer_part(generated_text)

            if answer_part:
                child = MCTSNode(
                    question=node.question,
                    reasoning_trace=node.reasoning_trace,
                    partial_answer=node.partial_answer + answer_part,
                    depth=node.depth + 1,
                    is_reasoning_node=False,
                    prior_prob=1.0 / self.config.max_children_per_node,
                )
                children.append(child)

        return children

    def _simulate_rollout(self, node: MCTSNode) -> float:
        """
        Simulate rollout from node to estimate value
        """
        self.search_stats["rollouts_performed"] += 1

        # Check cache
        state_key = node.state_hash
        if state_key in self.value_cache:
            self.search_stats["cache_hits"] += 1
            return self.value_cache[state_key]

        # Perform rollout
        rollout_trace = self._perform_rollout(node)

        # Evaluate rollout
        value = self._evaluate_rollout(rollout_trace)

        # Cache result
        if self.config.cache_states:
            self.value_cache[state_key] = value

        return value

    def _perform_rollout(self, node: MCTSNode) -> Dict[str, Any]:
        """Perform rollout simulation"""
        if node is None:
            return {"reasoning_trace": [], "answer": "", "depth": 0}

        current_trace = node.reasoning_trace.copy()
        current_answer = node.partial_answer
        current_depth = node.depth

        # Rollout for specified depth
        for step in range(self.config.rollout_depth):
            if current_depth >= self.config.max_depth:
                break

            # Decide whether to add reasoning or answer
            if (
                len(current_trace) < self.config.max_reasoning_steps
                and random.random() < 0.7
            ):
                # Add reasoning step
                reasoning_step = self._generate_rollout_reasoning(
                    current_trace, current_answer
                )
                if reasoning_step:
                    current_trace.append(reasoning_step)
            else:
                # Add answer part
                answer_part = self._generate_rollout_answer(
                    current_trace, current_answer
                )
                if answer_part:
                    current_answer += answer_part

            current_depth += 1

        return {
            "reasoning_trace": current_trace,
            "answer": current_answer,
            "depth": current_depth,
        }

    def _generate_rollout_reasoning(self, trace: List[str], answer: str) -> str:
        """Generate reasoning step for rollout"""
        if self.root_node is None:
            return "Default reasoning step"

        prompt = f"Question: {self.root_node.question}\n"
        prompt += "Reasoning so far:\n" + "\n".join(trace) + "\n"
        if answer:
            prompt += f"Partial answer: {answer}\n"
        prompt += "Next reasoning step: "

        with torch.no_grad():
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            )
            if hasattr(inputs, "input_ids"):
                input_ids = inputs.input_ids
            else:
                input_ids = inputs["input_ids"]

            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=30,
                temperature=self.config.rollout_temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=self.config.rollout_temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return self._extract_reasoning_step(generated)

    def _generate_rollout_answer(self, trace: List[str], answer: str) -> str:
        """Generate answer part for rollout"""
        if self.root_node is None:
            return "Default answer"

        prompt = f"Question: {self.root_node.question}\n"
        prompt += "Reasoning:\n" + "\n".join(trace) + "\n"
        prompt += "Answer: " + answer

        with torch.no_grad():
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            )
            if hasattr(inputs, "input_ids"):
                input_ids = inputs.input_ids
            else:
                input_ids = inputs["input_ids"]

            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=20,
                temperature=self.config.rollout_temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=self.config.rollout_temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return self._extract_answer_part(generated)

    def _evaluate_rollout(self, rollout_trace: Dict[str, Any]) -> float:
        """Evaluate rollout trace and return value"""
        reasoning = rollout_trace["reasoning_trace"]
        answer = rollout_trace["answer"]

        # Base value from answer quality
        answer_value = self._evaluate_answer_quality(answer)

        # Reasoning coherence bonus
        reasoning_value = self._evaluate_reasoning_coherence(reasoning)

        # Length penalty/bonus
        length_value = self._evaluate_length_appropriateness(reasoning, answer)

        # Combine values
        total_value = answer_value * 0.5 + reasoning_value * 0.3 + length_value * 0.2

        return total_value * self.config.value_scale

    def _evaluate_answer_quality(self, answer: str) -> float:
        """Evaluate answer quality"""
        if not answer:
            return 0.0

        # Simple heuristics (can be replaced with more sophisticated evaluation)
        score = 0.0

        # Length appropriateness
        if 10 <= len(answer) <= 200:
            score += 0.3

        # Contains numbers if question is mathematical
        if (
            any(c.isdigit() for c in answer)
            and self.root_node
            and "math" in self.root_node.question.lower()
        ):
            score += 0.2

        # Ends with proper punctuation
        if answer.strip().endswith((".", "!", "?")):
            score += 0.1

        # No obvious repetition
        words = answer.lower().split()
        if len(set(words)) / max(1, len(words)) > 0.7:
            score += 0.2

        # Confidence based on generation probability (simplified)
        score += 0.2  # Placeholder for actual probability calculation

        return min(1.0, score)

    def _evaluate_reasoning_coherence(self, reasoning: List[str]) -> float:
        """Evaluate reasoning coherence"""
        if not reasoning:
            return 0.0

        score = 0.0

        # Reasoning length appropriateness
        if 2 <= len(reasoning) <= 6:
            score += 0.3

        # Step progression (each step should build on previous)
        for i in range(1, len(reasoning)):
            if len(reasoning[i]) > len(reasoning[i - 1]) * 0.5:  # Reasonable length
                score += 0.1

        # Logical connectors presence
        connectors = ["because", "therefore", "since", "thus", "so", "then", "next"]
        for step in reasoning:
            if any(connector in step.lower() for connector in connectors):
                score += 0.1
                break

        # No obvious repetition
        all_text = " ".join(reasoning).lower()
        words = all_text.split()
        if len(set(words)) / max(1, len(words)) > 0.8:
            score += 0.2

        return min(1.0, score)

    def _evaluate_length_appropriateness(
        self, reasoning: List[str], answer: str
    ) -> float:
        """Evaluate if lengths are appropriate"""
        total_length = sum(len(step) for step in reasoning) + len(answer)

        # Target total length based on question complexity
        question_len = len(self.root_node.question) if self.root_node else 50
        target_length = 100 + question_len * 2

        if total_length < target_length * 0.5:
            return 0.3  # Too short
        elif total_length > target_length * 2.0:
            return 0.5  # Too long
        else:
            return 1.0  # Appropriate

    def _backpropagate_value(self, node: MCTSNode, value: float):
        """Backpropagate value through tree"""
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            current = current.parent

    def _get_best_result(self) -> Dict[str, Any]:
        """Get best result from search tree"""
        # Find leaf with highest visit count and value
        best_leaf = None
        best_score = -float("inf")

        if self.root_node is None:
            return {
                "reasoning_trace": [],
                "answer": "",
                "value": 0.0,
                "visit_count": 0,
                "depth": 0,
            }

        for leaf in self._get_all_leaves(self.root_node):
            # Combine visit count and mean value
            score = leaf.visit_count * 0.5 + leaf.mean_value * 0.5
            if score > best_score:
                best_score = score
                best_leaf = leaf

        if best_leaf is None:
            # Fallback to root
            best_leaf = self.root_node

        if best_leaf is None:
            # Final fallback
            return {
                "reasoning_trace": [],
                "answer": "",
                "value": 0.0,
                "visit_count": 0,
                "depth": 0,
            }

        return {
            "reasoning_trace": best_leaf.reasoning_trace,
            "answer": best_leaf.partial_answer,
            "value": best_leaf.mean_value,
            "visit_count": best_leaf.visit_count,
            "depth": best_leaf.depth,
        }

    def _get_all_leaves(self, node: Optional[MCTSNode]) -> List[MCTSNode]:
        """Get all leaf nodes"""
        if node is None or not node.children:
            return [node] if node is not None else []

        leaves = []
        for child in node.children:
            leaves.extend(self._get_all_leaves(child))
        return leaves

    def _get_max_tree_depth(self) -> int:
        """Get maximum depth of tree"""

        def get_depth(node):
            if not node.children:
                return node.depth
            return max(get_depth(child) for child in node.children)

        return get_depth(self.root_node) if self.root_node else 0

    def _get_tree_info(self) -> Dict[str, Any]:
        """Get tree structure information"""
        if not self.root_node:
            return {}

        return {
            "total_nodes": self.search_stats["total_nodes"],
            "reasoning_nodes": self.search_stats["reasoning_nodes"],
            "answer_nodes": self.search_stats["answer_nodes"],
            "max_depth": self._get_max_tree_depth(),
            "branching_factor": self.search_stats["total_nodes"]
            / max(1, self.current_iteration),
            "leaves_count": len(self._get_all_leaves(self.root_node)),
        }

    def _build_reasoning_prompt(self, node: MCTSNode) -> str:
        """Build prompt for reasoning generation"""
        prompt = f"Question: {node.question}\n"

        if node.reasoning_trace:
            prompt += "Reasoning so far:\n"
            for i, step in enumerate(node.reasoning_trace, 1):
                prompt += f"{i}. {step}\n"

        prompt += "Next reasoning step: "
        return prompt

    def _build_answer_prompt(self, node: MCTSNode) -> str:
        """Build prompt for answer generation"""
        prompt = f"Question: {node.question}\n"

        if node.reasoning_trace:
            prompt += "Reasoning:\n"
            for step in node.reasoning_trace:
                prompt += f"- {step}\n"

        prompt += "Answer: "
        if node.partial_answer:
            prompt += node.partial_answer

        return prompt

    def _extract_reasoning_step(self, generated_text: str) -> str:
        """Extract reasoning step from generated text"""
        # Clean up and extract first sentence or line
        lines = generated_text.strip().split("\n")
        if lines:
            first_line = lines[0].strip()
            # Remove common prefixes
            for prefix in ["Step:", "Next:", "Then:", "- "]:
                if first_line.startswith(prefix):
                    first_line = first_line[len(prefix) :].strip()
            return first_line
        return ""

    def _extract_answer_part(self, generated_text: str) -> str:
        """Extract answer part from generated text"""
        # Take first sentence or up to first newline
        text = generated_text.strip()
        if "\n" in text:
            text = text.split("\n")[0]
        if "." in text:
            text = text.split(".")[0] + "."
        return text

    def _log_search_progress(self):
        """Log search progress"""
        elapsed = time.time() - self.search_start_time
        logging.info(
            f"MCTS Iteration {self.current_iteration}: "
            f"Nodes={self.search_stats['total_nodes']}, "
            f"Depth={self._get_max_tree_depth()}, "
            f"Time={elapsed:.2f}s"
        )


def create_mcts_cot_searcher(
    model, tokenizer, config: Optional[MCTSConfig] = None
) -> MCTSCoTSearcher:
    """Create MCTS CoT searcher with default config"""
    if config is None:
        config = MCTSConfig()

    return MCTSCoTSearcher(model, tokenizer, config)
