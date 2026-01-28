"""
Specialized Head for JSON, Math, and Algorithm tasks
Separate from CoTSpecializationHeads to support ratio parameter
"""

import torch
import torch.nn as nn
from typing import Dict


class SpecializedHead(nn.Module):
    """
    Specialized head for specific tasks (JSON, Math, Algorithm)
    Supports ratio parameter for weighted combination with main output
    """

    def __init__(self, hidden_dim: int, internal_dim: int = 256, ratio: float = 0.1, cot_hidden_dim: int | None = None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.internal_dim = internal_dim if cot_hidden_dim is None else cot_hidden_dim
        self.ratio = ratio
        
        # Task-specific layers
        self.task_layers = nn.Sequential(
            nn.Linear(hidden_dim, self.internal_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.internal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Task router to determine task relevance
        self.task_router = nn.Sequential(
            nn.Linear(hidden_dim, self.internal_dim // 2),
            nn.ReLU(),
            nn.Linear(self.internal_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Process through specialized head
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            
        Returns:
            Specialized output (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Apply task-specific layers
        task_output = self.task_layers(hidden_states)
        
        # Get task relevance weights
        router_weights = self.task_router(hidden_states)  # (batch_size, seq_len, 1)
        
        # Combine with router weights
        weighted_output = task_output * router_weights
        
        return weighted_output