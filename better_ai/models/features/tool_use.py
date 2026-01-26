
import torch
import torch.nn as nn
from typing import Dict


class ToolUseHeads(nn.Module):
    """
    Tool-Use specialization heads for API call prediction
    Separate routing between text generation and function calling
    """

    def __init__(self, hidden_dim: int, tool_vocab_size: int = 1000, tool_hidden_dim: int = 512):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.tool_vocab_size = tool_vocab_size
        self.tool_hidden_dim = tool_hidden_dim

        # Tool/Text router
        self.mode_router = nn.Sequential(
            nn.Linear(hidden_dim, tool_hidden_dim),
            nn.ReLU(),
            nn.Linear(tool_hidden_dim, 1),
            nn.Sigmoid()  # 0 for text generation, 1 for tool use
        )

        # Tool prediction head
        self.tool_head = nn.Sequential(
            nn.Linear(hidden_dim, tool_hidden_dim),
            nn.ReLU(),
            nn.Linear(tool_hidden_dim, tool_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(tool_hidden_dim // 2, tool_vocab_size)
        )

        # Argument prediction head
        self.argument_head = nn.Sequential(
            nn.Linear(hidden_dim, tool_hidden_dim),
            nn.ReLU(),
            nn.Linear(tool_hidden_dim, tool_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(tool_hidden_dim // 2, hidden_dim)  # Embed arguments in hidden space
        )

        # Hallucination prevention
        self.confidence_scorer = nn.Sequential(
            nn.Linear(hidden_dim, tool_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(tool_hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict tool usage and arguments

        Returns:
            Dictionary with tool_logits, argument_embeddings, mode_scores, confidence
        """
        if hidden_states.dim() == 3:
            hidden_repr = hidden_states[:, -1, :]
        else:
            hidden_repr = hidden_states

        # Determine mode (tool vs text)
        mode_score = self.mode_router(hidden_repr)  # (batch_size, 1)

        # Predict tool
        tool_logits = self.tool_head(hidden_repr)  # (batch_size, tool_vocab_size)

        # Predict arguments
        argument_embeddings = self.argument_head(hidden_repr)  # (batch_size, hidden_dim)

        # Confidence score (prevent hallucination)
        confidence = self.confidence_scorer(hidden_repr)  # (batch_size, 1)

        # Suppress low-confidence predictions
        tool_logits = tool_logits * confidence

        return {
            "tool_logits": tool_logits,
            "argument_embeddings": argument_embeddings,
            "mode_score": mode_score,  # 0 for text, 1 for tool
            "confidence": confidence,
        }
