"""
CLEANER: Self-Purified Trajectories with SAAR
Similarity-Aware Adaptive Rollback for eliminating error-contaminated context during data collection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Any, Union
import numpy as np
import difflib
import logging


class SemanticSimilarityCalculator:
    """
    Computes semantic similarity between text segments for SAAR
    """

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold

    def compute_textual_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity using multiple metrics"""
        # Sequence matcher similarity
        seq_similarity = difflib.SequenceMatcher(None, text1, text2).ratio()

        # Word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if len(words1) == 0 and len(words2) == 0:
            word_similarity = 1.0
        else:
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            word_similarity = len(intersection) / len(union) if union else 0.0

        # Weighted combination
        combined_similarity = 0.6 * seq_similarity + 0.4 * word_similarity
        return combined_similarity

    def compute_code_similarity(self, code1: str, code2: str) -> float:
        """Enhanced similarity for code segments"""
        # Extract structural elements
        struct1 = self._extract_code_structure(code1)
        struct2 = self._extract_code_structure(code2)

        # Compare structures
        struct_similarity = difflib.SequenceMatcher(None, struct1, struct2).ratio()

        # Token-level similarity
        tokens1 = set(code1.replace("\n", " ").split())
        tokens2 = set(code2.replace("\n", " ").split())

        if len(tokens1) == 0 and len(tokens2) == 0:
            token_similarity = 1.0
        else:
            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)
            token_similarity = len(intersection) / len(union) if union else 0.0

        return 0.5 * struct_similarity + 0.5 * token_similarity

    def _extract_code_structure(self, code: str) -> str:
        """Extract structural elements from code"""
        # Remove variable content, keep structure
        import re

        # Replace literals with placeholders
        code = re.sub(r'"[^"]*"', '"X"', code)
        code = re.sub(r"'[^']*'", "'X'", code)
        code = re.sub(r"\d+", "N", code)

        # Remove whitespace
        code = " ".join(code.split())

        return code


class RollbackGranularityEstimator:
    """
    Estimates the appropriate rollback granularity based on error type
    """

    def __init__(self):
        self.error_patterns = {
            "syntax_error": ["SyntaxError", "Invalid syntax", "unexpected EOF"],
            "name_error": ["NameError", "not defined", "undefined"],
            "type_error": ["TypeError", "incompatible", "expected"],
            "runtime_error": ["RuntimeError", "execution failed"],
            "import_error": ["ImportError", "module not found", "no module"],
        }

    def estimate_granularity(
        self, error_text: str, failed_segment: str, successful_correction: str
    ) -> str:
        """
        Estimate rollback granularity: shallow, medium, or deep

        Returns: "shallow", "medium", or "deep"
        """
        error_lower = error_text.lower()

        # Check error type
        error_type = "unknown"
        for err_type, patterns in self.error_patterns.items():
            if any(pattern.lower() in error_lower for pattern in patterns):
                error_type = err_type
                break

        # Determine granularity based on error type and length
        failed_len = len(failed_segment.split())
        success_len = len(successful_correction.split())

        if error_type == "syntax_error":
            return "shallow"  # Usually small syntax fixes
        elif error_type == "name_error" or error_type == "type_error":
            return "medium"  # Variable/logic issues
        elif failed_len > success_len * 2:
            return "deep"  # Major simplification needed
        elif error_type == "runtime_error":
            return "medium"  # Logic error fixes
        else:
            return "medium"  # Default to medium


class SAARController:
    """
    Similarity-Aware Adaptive Rollback controller
    """

    def __init__(
        self,
        similarity_calculator: SemanticSimilarityCalculator,
        granularity_estimator: RollbackGranularityEstimator,
        min_similarity: float = 0.5,
        max_rollback_ratio: float = 0.8,
    ):
        self.similarity_calculator = similarity_calculator
        self.granularity_estimator = granularity_estimator
        self.min_similarity = min_similarity
        self.max_rollback_ratio = max_rollback_ratio

        # Statistics tracking
        self.total_corrections = 0
        self.successful_corrections = 0
        self.rollback_history = []

    def should_rollback(
        self, failed_segment: str, correction_candidate: str, error_text: str
    ) -> Tuple[bool, str, float]:
        """
        Determine if rollback should be performed

        Returns:
            (should_rollback, granularity, similarity_score)
        """
        # Compute similarity
        if self._is_code_segment(failed_segment):
            similarity = self.similarity_calculator.compute_code_similarity(
                failed_segment, correction_candidate
            )
        else:
            similarity = self.similarity_calculator.compute_textual_similarity(
                failed_segment, correction_candidate
            )

        # Check minimum similarity threshold
        if similarity < self.min_similarity:
            return False, "none", similarity

        # Estimate granularity
        granularity = self.granularity_estimator.estimate_granularity(
            error_text, failed_segment, correction_candidate
        )

        return True, granularity, similarity

    def apply_rollback(
        self,
        trajectory: List[Dict[str, Any]],
        error_position: int,
        correction: str,
        granularity: str,
    ) -> List[Dict[str, Any]]:
        """
        Apply rollback with specified granularity
        """
        if granularity == "shallow":
            return self._shallow_rollback(trajectory, error_position, correction)
        elif granularity == "medium":
            return self._medium_rollback(trajectory, error_position, correction)
        elif granularity == "deep":
            return self._deep_rollback(trajectory, error_position, correction)
        else:
            return trajectory

    def _shallow_rollback(
        self, trajectory: List[Dict[str, Any]], error_position: int, correction: str
    ) -> List[Dict[str, Any]]:
        """Shallow rollback: replace only the error segment"""
        if error_position < len(trajectory):
            trajectory[error_position]["content"] = correction
            trajectory[error_position]["corrected"] = True
            trajectory[error_position]["rollback_type"] = "shallow"

        return trajectory

    def _medium_rollback(
        self, trajectory: List[Dict[str, Any]], error_position: int, correction: str
    ) -> List[Dict[str, Any]]:
        """Medium rollback: replace error segment and previous step"""
        rollback_start = max(0, error_position - 1)

        for i in range(rollback_start, min(error_position + 1, len(trajectory))):
            if i == error_position:
                trajectory[i]["content"] = correction
                trajectory[i]["corrected"] = True
                trajectory[i]["rollback_type"] = "medium"
            else:
                trajectory[i]["content"] = self._simplify_content(
                    trajectory[i]["content"]
                )
                trajectory[i]["rollback_type"] = "medium_simplify"

        return trajectory

    def _deep_rollback(
        self, trajectory: List[Dict[str, Any]], error_position: int, correction: str
    ) -> List[Dict[str, Any]]:
        """Deep rollback: replace error segment and multiple previous steps"""
        rollback_start = max(0, error_position - 3)

        for i in range(rollback_start, min(error_position + 1, len(trajectory))):
            if i == error_position:
                trajectory[i]["content"] = correction
                trajectory[i]["corrected"] = True
                trajectory[i]["rollback_type"] = "deep"
            else:
                trajectory[i]["content"] = self._simplify_content(
                    trajectory[i]["content"]
                )
                trajectory[i]["rollback_type"] = "deep_simplify"

        return trajectory

    def _is_code_segment(self, segment: str) -> bool:
        """Check if segment contains code"""
        code_indicators = [
            "def ",
            "import ",
            "class ",
            "for ",
            "if ",
            "while ",
            "return ",
            "print(",
        ]
        return any(indicator in segment for indicator in code_indicators)

    def _simplify_content(self, content: str) -> str:
        """Simplify content during rollback"""
        # Remove complex parts, keep core structure
        lines = content.split("\n")
        simplified_lines = []

        for line in lines:
            line = line.strip()
            # Keep simple assignments and function definitions
            if (
                line
                and not line.startswith("#")
                and not line.startswith('"""')
                and len(line) < 100
            ):
                simplified_lines.append(line)

        return "\n".join(simplified_lines)


class CLEANERDataCollector:
    """
    Data collector with self-purification capabilities
    """

    def __init__(
        self, saar_controller: SAARController, purification_enabled: bool = True
    ):
        self.saar_controller = saar_controller
        self.purification_enabled = purification_enabled

        # Statistics
        self.trajectories_processed = 0
        self.trajectories_purified = 0
        self.errors_corrected = 0

        logging.info(
            f"CLEANER data collector initialized, purification: {purification_enabled}"
        )

    def process_trajectory(
        self, raw_trajectory: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process trajectory with error detection and purification

        Args:
            raw_trajectory: List of steps with content, errors, corrections

        Returns:
            Purified trajectory
        """
        self.trajectories_processed += 1

        if not self.purification_enabled:
            return raw_trajectory

        # Detect errors and corrections
        purified_trajectory = self._detect_and_correct_errors(raw_trajectory)

        if purified_trajectory != raw_trajectory:
            self.trajectories_purified += 1
            logging.info(f"Trajectory purified: {len(purified_trajectory)} steps")

        return purified_trajectory

    def _detect_and_correct_errors(
        self, trajectory: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect errors and apply corrections using SAAR"""
        purified_trajectory = trajectory.copy()
        errors_found = 0

        for i, step in enumerate(trajectory):
            # Check for errors
            error_info = step.get("error", {})
            if not error_info:
                continue

            error_text = error_info.get("message", "")
            failed_content = step.get("content", "")
            correction_candidate = step.get("correction", "")

            if not correction_candidate:
                continue

            # Should we rollback?
            should_rollback, granularity, similarity = (
                self.saar_controller.should_rollback(
                    failed_content, correction_candidate, error_text
                )
            )

            if should_rollback:
                # Apply rollback
                purified_trajectory = self.saar_controller.apply_rollback(
                    purified_trajectory, i, correction_candidate, granularity
                )

                errors_found += 1
                self.errors_corrected += 1

                logging.debug(
                    f"Applied {granularity} rollback at step {i}, similarity: {similarity:.3f}"
                )

        # Update SAAR statistics
        self.saar_controller.total_corrections += errors_found
        if errors_found > 0:
            self.saar_controller.successful_corrections += 1

        return purified_trajectory

    def collect_batch(
        self, raw_trajectories: List[List[Dict[str, Any]]]
    ) -> List[List[Dict[str, Any]]]:
        """Process a batch of trajectories"""
        purified_trajectories = []

        for trajectory in raw_trajectories:
            purified = self.process_trajectory(trajectory)
            purified_trajectories.append(purified)

        return purified_trajectories

    def get_statistics(self) -> Dict[str, Any]:
        """Get collection and purification statistics"""
        return {
            "trajectories_processed": self.trajectories_processed,
            "trajectories_purified": self.trajectories_purified,
            "errors_corrected": self.errors_corrected,
            "purification_rate": (
                self.trajectories_purified / max(1, self.trajectories_processed)
            ),
            "saar_stats": {
                "total_corrections": self.saar_controller.total_corrections,
                "successful_corrections": self.saar_controller.successful_corrections,
                "success_rate": (
                    self.saar_controller.successful_corrections
                    / max(1, self.saar_controller.total_corrections)
                ),
            },
        }


def create_cleaner_pipeline(
    min_similarity: float = 0.5,
    max_rollback_ratio: float = 0.8,
    purification_enabled: bool = True,
) -> CLEANERDataCollector:
    """
    Factory function to create a complete CLEANER pipeline
    """
    similarity_calculator = SemanticSimilarityCalculator(min_similarity)
    granularity_estimator = RollbackGranularityEstimator()
    saar_controller = SAARController(
        similarity_calculator, granularity_estimator, min_similarity, max_rollback_ratio
    )

    return CLEANERDataCollector(saar_controller, purification_enabled)
