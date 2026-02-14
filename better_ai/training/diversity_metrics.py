"""
Reasoning Diversity Metrics
Measures and encourages varied solution approaches in RLHF
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import numpy as np

class DiversityMeasurer:
    """
    Computes diversity scores for a group of reasoning trajectories
    """
    def __init__(self, n_gram: int = 2):
        self.n_gram = n_gram
        self.approach_keywords = {
            "brute_force": ["brute force", "bruteforce", "try all", "exhaustive"],
            "greedy": ["greedy", "local optimum", "always pick"],
            "dynamic_programming": ["dp", "dynamic programming", "memoization", "table"],
            "divide_and_conquer": ["divide and conquer", "recursive call", "subproblem"],
            "backtracking": ["backtrack", "dfs", "recursion", "pruning"],
            "sliding_window": ["sliding window", "two pointers", "left pointer"],
            "binary_search": ["binary search", "logarithmic", "midpoint"],
        }

    def classify_approach(self, trajectory: str) -> str:
        """Classifies the reasoning approach based on keywords"""
        text = trajectory.lower()
        for approach, keywords in self.approach_keywords.items():
            if any(kw in text for kw in keywords):
                return approach
        return "unknown"

    def compute_approach_diversity(self, trajectories: List[str]) -> float:
        """Measures diversity of identified solution approaches"""
        if not trajectories:
            return 0.0
        approaches = [self.classify_approach(t) for t in trajectories]
        unique_approaches = set(approaches)
        # Ratio of unique approaches to total trajectories
        return len(unique_approaches) / len(trajectories)

    def compute_n_gram_diversity(self, trajectories: List[str]) -> float:
        """
        Calculates distinct n-grams across all trajectories
        """
        all_ngrams = set()
        total_ngrams = 0

        for traj in trajectories:
            words = traj.split()
            if len(words) < self.n_gram:
                continue

            ngrams = [tuple(words[i:i+self.n_gram]) for i in range(len(words)-self.n_gram+1)]
            all_ngrams.update(ngrams)
            total_ngrams += len(ngrams)

        if total_ngrams == 0:
            return 0.0

        return len(all_ngrams) / total_ngrams

    def compute_embedding_diversity(self, embeddings: torch.Tensor) -> float:
        """
        Calculates diversity using cosine similarity between trajectory embeddings
        """
        # embeddings: [Group_size, Hidden_dim]
        if embeddings.size(0) <= 1:
            return 0.0

        # Compute pairwise cosine similarity
        norm_emb = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.matmul(norm_emb, norm_emb.t())

        # Diversity = 1 - average pairwise similarity (excluding self-similarity)
        mask = torch.eye(sim_matrix.size(0), device=embeddings.device).bool()
        avg_sim = sim_matrix[~mask].mean()

        return (1.0 - avg_sim).item()

    def _split_into_steps(self, trajectory: str) -> List[str]:
        """Splits a trajectory into individual reasoning steps"""
        # Common step markers
        import re
        step_pattern = r'(?:Step\s+\d+:|(?:\n\n)+)'
        steps = re.split(step_pattern, trajectory)
        return [s.strip() for s in steps if s.strip()]

    def measure_step_diversity(self, trajectories: List[str]) -> float:
        """Measures diversity of intermediate reasoning steps across trajectories"""
        if not trajectories:
            return 0.0

        all_trajs_steps = [self._split_into_steps(t) for t in trajectories]

        # Calculate unique steps across all trajectories (normalized)
        all_steps = []
        for steps in all_trajs_steps:
            all_steps.extend(steps)

        if not all_steps:
            return 0.0

        unique_steps = set(all_steps)
        return len(unique_steps) / len(all_steps)

    def label_reasoning_patterns(self, trajectory: str) -> List[str]:
        """Identifies reasoning patterns in a trajectory using heuristics"""
        patterns = []
        text = trajectory.lower()

        heuristics = {
            "verification": ["verify", "check", "confirm", "proof", "validate", "ensure"],
            "backtracking": ["go back", "instead", "wait", "actually", "reconsider", "correction"],
            "exploratory": ["maybe", "try", "perhaps", "could", "hypothesis", "explore"],
            "analytical": ["analyze", "break down", "structure", "components", "logic", "deduce"],
            "mathematical": ["equation", "formula", "calculate", "sum", "product", "derive"]
        }

        for pattern, keywords in heuristics.items():
            if any(kw in text for kw in keywords):
                patterns.append(pattern)

        return patterns if patterns else ["standard"]

    def compute_pattern_diversity(self, trajectories: List[str]) -> float:
        """Measures diversity of reasoning patterns across a group of trajectories"""
        if not trajectories:
            return 0.0

        all_patterns = []
        for t in trajectories:
            all_patterns.extend(self.label_reasoning_patterns(t))

        unique_patterns = set(all_patterns)
        return len(unique_patterns) / 5.0 # Normalized by number of known patterns

def get_diversity_reward(group_trajectories: List[str], group_embeddings: Optional[torch.Tensor] = None) -> float:
    """
    Computes an aggregate diversity reward for a group of rollouts
    """
    if not group_trajectories:
        return 0.0

    measurer = DiversityMeasurer()
    n_gram_div = measurer.compute_n_gram_diversity(group_trajectories)
    approach_div = measurer.compute_approach_diversity(group_trajectories)
    step_div = measurer.measure_step_diversity(group_trajectories)
    pattern_div = measurer.compute_pattern_diversity(group_trajectories)

    # Aggregate base reward from multiple text-based diversity signals
    base_reward = (0.3 * n_gram_div +
                   0.3 * approach_div +
                   0.2 * step_div +
                   0.2 * pattern_div)

    if group_embeddings is not None:
        emb_div = measurer.compute_embedding_diversity(group_embeddings)
        return 0.6 * base_reward + 0.4 * emb_div

    return base_reward
