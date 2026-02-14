"""
Data Curation and Decomposition
Implements Agent-FLAN style decomposition and Difficulty-Diversity-Quality curation
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging
import ast

class ASTComplexityScanner(ast.NodeVisitor):
    def __init__(self):
        self.complexity = 0
        self.nesting_depth = 0
        self.max_nesting_depth = 0

    def visit_If(self, node):
        self.complexity += 1
        self.nesting_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.nesting_depth)
        self.generic_visit(node)
        self.nesting_depth -= 1

    def visit_For(self, node):
        self.complexity += 1
        self.nesting_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.nesting_depth)
        self.generic_visit(node)
        self.nesting_depth -= 1

    def visit_While(self, node):
        self.complexity += 1
        self.nesting_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.nesting_depth)
        self.generic_visit(node)
        self.nesting_depth -= 1

    def visit_FunctionDef(self, node):
        self.complexity += 1
        self.nesting_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.nesting_depth)
        self.generic_visit(node)
        self.nesting_depth -= 1

class AgentFLANDecomposer:
    """
    Decomposes complex agent trajectories into smaller, manageable learning signals
    """
    def decompose_trajectory(self, trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits a full trajectory into sub-tasks and atomic reasoning steps
        """
        decomposed = []
        for i, step in enumerate(trajectory):
            # Each step becomes a training example with its own prompt and target
            decomposed.append({
                "prompt": self._generate_step_prompt(trajectory[:i], step),
                "target": step["content"],
                "step_idx": i,
                "is_atomic": True
            })
        return decomposed

    def _generate_step_prompt(self, history: List[Dict[str, Any]], current_step: Dict[str, Any]) -> str:
        # Reconstruct context for this specific step
        context = "\n".join([s["content"] for s in history])
        return f"Context: {context}\nNext Step Task: {current_step.get('goal', 'Continue reasoning')}"

class DatasetCurator:
    """
    Validates and filters data based on Difficulty, Diversity, and Quality (DDQ)
    """
    def __init__(self, quality_threshold: float = 0.8):
        self.quality_threshold = quality_threshold

    def filter_by_quality(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filters out low-quality items"""
        return [item for item in dataset if item.get("quality_score", 0) >= self.quality_threshold]

    def select_diverse_subset(self, dataset: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Selects k most diverse items using a greedy approach (mock)"""
        if len(dataset) <= k:
            return dataset
        return dataset[:k] # Simplified

    def estimate_difficulty(self, code: str) -> float:
        """
        Estimates difficulty based on AST complexity and nesting depth.
        """
        try:
            tree = ast.parse(code)
            scanner = ASTComplexityScanner()
            scanner.visit(tree)
            # Combine metrics into a difficulty score [0, 1]
            # Base complexity + 2*max_depth
            raw_score = scanner.complexity + (2 * scanner.max_nesting_depth)
            # Normalize (clamped at 20 for max difficulty)
            return min(1.0, raw_score / 20.0)
        except Exception:
            # Fallback to length-based if AST parsing fails
            return min(1.0, len(code) / 2000.0)

    def balance_by_difficulty(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensures a balanced distribution of problem difficulties"""
        for item in dataset:
            if "difficulty" not in item:
                content = item.get("content", item.get("target", ""))
                item["difficulty"] = self.estimate_difficulty(content)

        # Sort by difficulty to allow curriculum sampling later
        dataset.sort(key=lambda x: x["difficulty"])
        return dataset

def curate_training_corpus(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Applies full curation and decomposition pipeline
    """
    # 1. DDQ Filtering
    curator = DatasetCurator()
    clean_data = curator.filter_by_quality(raw_data)

    # 2. Agent-FLAN Decomposition
    decomposer = AgentFLANDecomposer()
    final_data = []
    for item in clean_data:
        if "trajectory" in item:
            final_data.extend(decomposer.decompose_trajectory(item["trajectory"]))
        else:
            final_data.append(item)

    return final_data
