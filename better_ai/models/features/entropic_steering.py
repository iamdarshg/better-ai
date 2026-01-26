
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class EntropicSteering(nn.Module):
    """
    Real-time entropy monitoring and clarifying question insertion
    Detects uncertainty spikes and triggers clarification requests
    """

    def __init__(self, hidden_dim: int, entropy_threshold: float = 2.5):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.entropy_threshold = entropy_threshold

        # Entropy spike detector
        self.spike_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Clarification question generator
        self.clarification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # Generate embedding for clarification
        )

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy of logits"""
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        return entropy

    def forward(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Monitor entropy and trigger clarification

        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            logits: (batch_size, seq_len, vocab_size)

        Returns:
            Dictionary with entropy_scores, spike_detected, clarification_triggers
        """
        # Compute entropy per position
        entropy_scores = self.compute_entropy(logits)  # (batch_size, seq_len)

        # Detect spikes
        entropy_mean = entropy_scores.mean(dim=-1, keepdim=True)
        entropy_std = entropy_scores.std(dim=-1, keepdim=True)
        normalized_entropy = (entropy_scores - entropy_mean) / (entropy_std + 1e-6)

        spike_detected = entropy_scores > self.entropy_threshold

        # Generate clarification triggers
        clarification_embeddings = self.clarification_head(hidden_states)

        # Determine when to ask clarifying questions
        clarification_triggers = self.spike_detector(hidden_states)  # (batch_size, seq_len, 1)
        clarification_triggers = clarification_triggers * spike_detected.unsqueeze(-1).float()

        return {
            "entropy_scores": entropy_scores,
            "spike_detected": spike_detected,
            "clarification_triggers": clarification_triggers,
            "clarification_embeddings": clarification_embeddings,
            "entropy_mean": entropy_mean,
        }
