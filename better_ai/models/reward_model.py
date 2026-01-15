"""
BR-RM (Branch Reward Model) implementation
Two-turn scoring mechanism with adaptive branching for dimension selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
import math
from ..config import ModelConfig


class BranchRewardModel(nn.Module):
    """
    Branch Reward Model (BR-RM) for coding task evaluation
    Implements two-turn scoring with adaptive branching
    """
    
    def __init__(self, config: ModelConfig, hidden_dim: int = 512):
        super().__init__()
        
        self.config = config
        self.hidden_dim = hidden_dim
        
        # Main reward scoring head
        self.reward_head = nn.Sequential(
            nn.Linear(config.hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Branch selector for adaptive dimension selection
        self.branch_selector = nn.Sequential(
            nn.Linear(config.hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # Select from 4 branches
            nn.Softmax(dim=-1)
        )
        
        # Multiple expert branches for different coding aspects
        self.correctness_head = nn.Sequential(
            nn.Linear(config.hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.efficiency_head = nn.Sequential(
            nn.Linear(config.hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.readability_head = nn.Sequential(
            nn.Linear(config.hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.robustness_head = nn.Sequential(
            nn.Linear(config.hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.heads = [
            self.correctness_head,
            self.efficiency_head,
            self.readability_head,
            self.robustness_head
        ]
        
        # Rethinking module for branch-conditioned reasoning
        self.rethinking_module = nn.GRUCell(config.hidden_dim, config.hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_branch_scores: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute reward score using adaptive branching
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)
            attention_mask: Optional attention mask
            return_branch_scores: If True, also return individual branch scores
        
        Returns:
            reward_score: (batch_size,) reward scores
            or
            (reward_score, branch_dict) if return_branch_scores=True
        """
        
        # Handle sequence dimensions - take last token or mean pooling
        if hidden_states.dim() == 3:
            # Take mean pooling across sequence if attention_mask provided, else last token
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_hidden = (hidden_states * mask_expanded).sum(1)
                sum_mask = mask_expanded.sum(1)
                hidden_repr = sum_hidden / (sum_mask + 1e-9)
            else:
                hidden_repr = hidden_states[:, -1, :]  # Last token
        else:
            hidden_repr = hidden_states
        
        # Get branch selection weights
        branch_weights = self.branch_selector(hidden_repr)  # (batch_size, 4)
        
        # Compute scores from each branch
        correctness_score = self.correctness_head(hidden_repr)
        efficiency_score = self.efficiency_head(hidden_repr)
        readability_score = self.readability_head(hidden_repr)
        robustness_score = self.robustness_head(hidden_repr)
        
        # Combine branch scores using learned weights
        branch_scores = torch.cat([
            correctness_score,
            efficiency_score,
            readability_score,
            robustness_score
        ], dim=-1)  # (batch_size, 4)
        
        # Weighted combination
        combined_score = (branch_scores * branch_weights).sum(dim=-1, keepdim=True)
        
        # Apply rethinking module for adaptive reasoning
        rethinking_hidden = self.rethinking_module(hidden_repr, hidden_repr)
        final_reward = self.reward_head(rethinking_hidden)
        
        # Combine initial scores with rethinking
        final_reward = 0.7 * combined_score + 0.3 * final_reward
        
        if return_branch_scores:
            branch_dict = {
                "correctness": correctness_score.squeeze(-1),
                "efficiency": efficiency_score.squeeze(-1),
                "readability": readability_score.squeeze(-1),
                "robustness": robustness_score.squeeze(-1),
                "branch_weights": branch_weights,
                "combined": combined_score.squeeze(-1),
            }
            return final_reward.squeeze(-1), branch_dict
        
        return final_reward.squeeze(-1)
    
    def score_pair(
        self,
        chosen_hidden: torch.Tensor,
        rejected_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Score a pair of responses (chosen vs rejected)
        
        Returns:
            (chosen_score, rejected_score)
        """
        chosen_score = self.forward(chosen_hidden, attention_mask)
        rejected_score = self.forward(rejected_hidden, attention_mask)
        return chosen_score, rejected_score


class MultiAttributeRewardModel(nn.Module):
    """
    Multi-attribute regression for preference distributions
    Uses quantile regression for modeling uncertainty
    """
    
    def __init__(self, config: ModelConfig, num_attributes: int = 5, num_quantiles: int = 5):
        super().__init__()
        
        self.config = config
        self.num_attributes = num_attributes
        self.num_quantiles = num_quantiles
        
        # Attribute-specific heads with quantile regression
        self.attribute_heads = nn.ModuleList([
            self._build_quantile_head(config.hidden_dim, num_quantiles)
            for _ in range(num_attributes)
        ])
        
        # Attribute names (can be customized)
        self.attribute_names = [
            "correctness",
            "efficiency", 
            "readability",
            "robustness",
            "creativity"
        ][:num_attributes]
    
    def _build_quantile_head(self, hidden_dim: int, num_quantiles: int) -> nn.Module:
        """Build a quantile regression head"""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_quantiles),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-attribute scores with quantiles
        
        Returns:
            Dictionary with per-attribute quantile distributions
        """
        
        # Pool hidden states
        if hidden_states.dim() == 3:
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_hidden = (hidden_states * mask_expanded).sum(1)
                sum_mask = mask_expanded.sum(1)
                hidden_repr = sum_hidden / (sum_mask + 1e-9)
            else:
                hidden_repr = hidden_states[:, -1, :]
        else:
            hidden_repr = hidden_states
        
        results = {}
        for attr_name, head in zip(self.attribute_names, self.attribute_heads):
            quantile_scores = head(hidden_repr)  # (batch_size, num_quantiles)
            results[attr_name] = quantile_scores
        
        # Also compute point estimate (median)
        results["point_estimates"] = torch.stack([
            quantiles[:, self.num_quantiles // 2]
            for quantiles in results.values() if isinstance(quantiles, torch.Tensor)
        ], dim=1)
        
        return results
    
    def quantile_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        quantiles: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Compute quantile regression loss
        
        Args:
            predictions: Dictionary of predicted quantile distributions
            targets: Target values (batch_size, num_attributes)
            quantiles: List of quantile levels [0.25, 0.5, 0.75, ...]
        
        Returns:
            Loss value
        """
        if quantiles is None:
            quantiles = torch.linspace(0.1, 0.9, self.num_quantiles)
        
        total_loss = 0.0
        for i, (attr_name, pred_quantiles) in enumerate(predictions.items()):
            if attr_name == "point_estimates":
                continue
            
            target = targets[:, i] if targets.dim() > 1 else targets
            
            # Quantile loss
            for q, quantile_level in enumerate(quantiles):
                errors = target.unsqueeze(-1) - pred_quantiles
                loss = torch.where(
                    errors >= 0,
                    quantile_level * errors,
                    (quantile_level - 1) * errors
                )
                total_loss += loss.mean()
        
        return total_loss / max(len(self.attribute_names), 1)
