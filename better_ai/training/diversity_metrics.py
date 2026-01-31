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

def get_diversity_reward(group_trajectories: List[str], group_embeddings: Optional[torch.Tensor] = None) -> float:
    """
    Computes an aggregate diversity reward for a group of rollouts
    """
    measurer = DiversityMeasurer()
    n_gram_div = measurer.compute_n_gram_diversity(group_trajectories)

    if group_embeddings is not None:
        emb_div = measurer.compute_embedding_diversity(group_embeddings)
        return 0.5 * n_gram_div + 0.5 * emb_div

    return n_gram_div
