
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
        weight_entropy: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Monitor entropy and trigger clarification

        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            logits: (batch_size, seq_len, vocab_size)
            weight_entropy: Global model weight entropy for adaptive thresholding

        Returns:
            Dictionary with entropy_scores, spike_detected, clarification_triggers
        """
        # Compute output entropy per position
        output_entropy = self.compute_entropy(logits)  # (batch_size, seq_len)

        # Dynamic threshold adjustment based on weight entropy
        # Lower weight entropy (memorization) makes the model more sensitive to output entropy spikes
        effective_threshold = self.entropy_threshold
        if weight_entropy > 0:
            # Heuristic: adjust threshold based on weight entropy
            # If weight entropy is low, we expect lower output entropy, so we lower the threshold to catch spikes
            effective_threshold = self.entropy_threshold * (weight_entropy / 4.0) # Assume 4.0 is a typical high entropy
            effective_threshold = max(effective_threshold, 0.5)

        # Detect spikes
        spike_detected = output_entropy > effective_threshold

        # Generate clarification triggers
        clarification_embeddings = self.clarification_head(hidden_states)

        # Determine when to ask clarifying questions - incorporate both signals
        # We add a small bias based on weight entropy to the detector
        detector_input = hidden_states
        clarification_triggers = self.spike_detector(detector_input)  # (batch_size, seq_len, 1)

        # Combine output uncertainty (spike) with detector confidence
        clarification_triggers = clarification_triggers * spike_detected.unsqueeze(-1).float()

        return {
            "entropy_scores": output_entropy,
            "spike_detected": spike_detected,
            "clarification_triggers": clarification_triggers,
            "clarification_embeddings": clarification_embeddings,
            "weight_entropy_used": torch.tensor(weight_entropy, device=hidden_states.device),
            "effective_threshold": torch.tensor(effective_threshold, device=hidden_states.device),
        }
