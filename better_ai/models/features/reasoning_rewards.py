"""
Reasoning Rewards for DeepSeek Model
Includes Trace Validity, Structural Signal, and AHA-Moment rewards
"""

import torch
import torch.nn as nn
import re
from typing import Dict, List, Any, Optional, Tuple

class TraceValidityScorer:
    """
    Scores reasoning traces based on logical consistency and goal alignment
    """
    def __init__(self, model: nn.Module):
        self.model = model

    def score_trace(self, trace: List[str], goal: str) -> float:
        """
        Assigns a validity score to a sequence of reasoning steps
        """
        score = 0.5 # Baseline

        if len(trace) < 2:
            score -= 0.1

        reasoning_keywords = ["therefore", "because", "so", "consequently", "implies"]
        keyword_count = sum(1 for step in trace if any(kw in step.lower() for kw in reasoning_keywords))

        score += 0.1 * min(keyword_count, 5)

        goal_keywords = goal.lower().split()
        if trace and any(kw in trace[-1].lower() for kw in goal_keywords):
            score += 0.2

        return min(1.0, max(0.0, score))

class StructuralSignalReward:
    """
    Computes rewards based on the structural signals and tags in a trajectory
    """
    def __init__(self, required_tags: Optional[List[str]] = None):
        self.required_tags = required_tags or ["thought", "action", "observation"]
        self.tag_patterns = {tag: rf"<{tag}>.*?</{tag}>" for tag in self.required_tags}

    def compute_reward(self, text: str) -> float:
        """
        Calculates a structural compliance score for the given text
        """
        score = 0.0
        total_tags = len(self.required_tags)

        present_tags = 0
        for tag, pattern in self.tag_patterns.items():
            if re.search(pattern, text, re.DOTALL):
                present_tags += 1

        score += 0.5 * (present_tags / total_tags)

        order_score = 1.0
        thought_indices = [m.start() for m in re.finditer(r"<thought>", text)]
        action_indices = [m.start() for m in re.finditer(r"<action>", text)]

        if thought_indices and action_indices:
            if action_indices[0] < thought_indices[0]:
                order_score -= 0.2

        score += 0.3 * order_score

        balance_score = 1.0
        for tag in self.required_tags:
            opens = text.count(f"<{tag}>")
            closes = text.count(f"</{tag}>")
            if opens != closes:
                balance_score -= 0.1 * abs(opens - closes)

        score += 0.2 * max(0.0, balance_score)

        return score

class AHAMomentDetector:
    """
    Detects patterns indicative of breakthrough insights or major self-corrections
    """
    def __init__(self):
        self.patterns = [
            r"Wait, (I see|that's not right|actually)",
            r"Oh! (I should|now I see)",
            r"Wait a minute\.",
            r"Instead of .* I should .*",
            r"A better approach would be",
            r"Hold on, let me re-evaluate",
            r"I just realized that",
            r"This implies that .* which means .*"
        ]

    def detect_aha_moments(self, trace: str) -> List[Dict[str, Any]]:
        moments = []
        for pattern in self.patterns:
            for match in re.finditer(pattern, trace, re.IGNORECASE):
                moments.append({
                    "pattern": pattern,
                    "matched_text": match.group(0),
                    "start": match.start(),
                    "end": match.end()
                })
        return moments

    def compute_aha_reward(self, trace: str) -> float:
        moments = self.detect_aha_moments(trace)
        if not moments:
            return 0.0

        base_reward = 0.2
        count_reward = 0.1 * min(len(moments), 3)

        return base_reward + count_reward
