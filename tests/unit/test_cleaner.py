"""
Unit tests for CLEANER (Self-Purified Trajectories)
"""

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from better_ai.training.cleaner import (
    SemanticSimilarityCalculator,
    RollbackGranularityEstimator,
    SAARController,
    CLEANERDataCollector,
    create_cleaner_pipeline,
)


class TestSemanticSimilarityCalculator:
    """Test semantic similarity calculations"""

    def test_textual_similarity(self):
        calculator = SemanticSimilarityCalculator()

        # Identical text
        text1 = "hello world"
        text2 = "hello world"
        similarity = calculator.compute_textual_similarity(text1, text2)
        assert similarity == pytest.approx(1.0, rel=1e-3)

        # Different text
        text3 = "goodbye world"
        similarity = calculator.compute_textual_similarity(text1, text3)
        assert 0.0 < similarity < 1.0

    def test_code_similarity(self):
        calculator = SemanticSimilarityCalculator()

        # Similar code with different variable names
        code1 = "def add(a, b): return a + b"
        code2 = "def add(x, y): return x + y"
        similarity = calculator.compute_code_similarity(code1, code2)
        assert similarity > 0.5  # Should be quite similar

        # Different functions
        code3 = "def multiply(a, b): return a * b"
        similarity = calculator.compute_code_similarity(code1, code3)
        assert 0.0 < similarity < 0.8

    def test_code_structure_extraction(self):
        calculator = SemanticSimilarityCalculator()

        code = 'def test(x): return "hello" + str(42)'
        structure = calculator._extract_code_structure(code)

        # Should preserve structure but replace literals
        assert "def test(" in structure
        assert "return" in structure
        assert '"hello"' not in structure  # Should be replaced
        assert "42" not in structure  # Should be replaced


class TestRollbackGranularityEstimator:
    """Test rollback granularity estimation"""

    def test_syntax_error_granularity(self):
        estimator = RollbackGranularityEstimator()

        error_text = "SyntaxError: invalid syntax"
        failed_segment = "def broken_function("
        success_correction = "def fixed_function():"

        granularity = estimator.estimate_granularity(
            error_text, failed_segment, success_correction
        )
        assert granularity == "shallow"

    def test_name_error_granularity(self):
        estimator = RollbackGranularityEstimator()

        error_text = "NameError: name 'x' is not defined"
        failed_segment = "return x + y"
        success_correction = "return a + b"

        granularity = estimator.estimate_granularity(
            error_text, failed_segment, success_correction
        )
        assert granularity == "medium"

    def test_deep_rollback_condition(self):
        estimator = RollbackGranularityEstimator()

        error_text = "RuntimeError: execution failed"
        failed_segment = (
            "very long complex line with many variables and operations" * 10
        )
        success_correction = "simple result"

        granularity = estimator.estimate_granularity(
            error_text, failed_segment, success_correction
        )
        assert granularity == "deep"


class TestSAARController:
    """Test SAAR rollback control"""

    @pytest.fixture
    def similarity_calculator(self):
        return SemanticSimilarityCalculator(min_similarity=0.3)

    @pytest.fixture
    def granularity_estimator(self):
        return RollbackGranularityEstimator()

    @pytest.fixture
    def saar_controller(self, similarity_calculator, granularity_estimator):
        return SAARController(
            similarity_calculator,
            granularity_estimator,
            min_similarity=0.3,
            max_rollback_ratio=0.7,
        )

    def test_should_rollback_true(self, saar_controller):
        failed_segment = "def broken():"
        correction = "def fixed(): pass"
        error_text = "SyntaxError: invalid syntax"

        should_rollback, granularity, similarity = saar_controller.should_rollback(
            failed_segment, correction, error_text
        )

        assert should_rollback == True
        assert granularity in ["shallow", "medium", "deep"]
        assert similarity >= 0.3

    def test_should_rollback_false_low_similarity(self, saar_controller):
        failed_segment = "print('hello')"
        correction = "completely_different_function()"
        error_text = "small error"

        should_rollback, granularity, similarity = saar_controller.should_rollback(
            failed_segment, correction, error_text
        )

        assert should_rollback == False
        assert granularity == "none"
        assert similarity < 0.3

    def test_shallow_rollback(self, saar_controller):
        trajectory = [
            {"content": "step 1", "corrected": False},
            {"content": "step 2 with error", "corrected": False},
            {"content": "step 3", "corrected": False},
        ]

        corrected = saar_controller.apply_rollback(
            trajectory, 1, "fixed step 2", "shallow"
        )

        assert len(corrected) == 3
        assert corrected[1]["content"] == "fixed step 2"
        assert corrected[1]["corrected"] == True
        assert corrected[1]["rollback_type"] == "shallow"
        assert corrected[0]["content"] == "step 1"  # Unchanged

    def test_medium_rollback(self, saar_controller):
        trajectory = [
            {"content": "step 1", "corrected": False},
            {"content": "step 2 with error", "corrected": False},
            {"content": "step 3", "corrected": False},
        ]

        corrected = saar_controller.apply_rollback(
            trajectory, 1, "fixed step 2", "medium"
        )

        assert len(corrected) == 3
        assert corrected[1]["content"] == "fixed step 2"
        assert corrected[1]["corrected"] == True
        assert corrected[1]["rollback_type"] == "medium"
        assert corrected[0]["rollback_type"] == "medium_simplify"  # Also simplified


class TestCLEANERDataCollector:
    """Test CLEANER data collection"""

    @pytest.fixture
    def cleaner_collector(self):
        saar_controller = SAARController(
            SemanticSimilarityCalculator(), RollbackGranularityEstimator()
        )
        return CLEANERDataCollector(saar_controller, purification_enabled=True)

    def test_process_trajectory_without_errors(self, cleaner_collector):
        clean_trajectory = [
            {"content": "step 1", "error": {}},
            {"content": "step 2", "error": {}},
            {"content": "step 3", "error": {}},
        ]

        result = cleaner_collector.process_trajectory(clean_trajectory)

        assert result == clean_trajectory  # No changes
        assert cleaner_collector.trajectories_processed == 1
        assert cleaner_collector.trajectories_purified == 0

    def test_process_trajectory_with_errors(self, cleaner_collector):
        error_trajectory = [
            {
                "content": "def broken(",
                "error": {"message": "SyntaxError"},
                "correction": "def fixed(): pass",
            },
            {
                "content": "print(x)",
                "error": {"message": "NameError"},
                "correction": "print('hello')",
            },
            {"content": "step 3", "error": {}},
        ]

        result = cleaner_collector.process_trajectory(error_trajectory)

        assert result != error_trajectory  # Should be modified
        assert cleaner_collector.trajectories_processed == 1
        assert cleaner_collector.trajectories_purified == 1
        assert cleaner_collector.errors_corrected >= 1

    def test_collect_batch(self, cleaner_collector):
        trajectories = [
            [{"content": "clean trajectory"}],  # Clean
            [
                {
                    "content": "error",
                    "error": {"message": "error"},
                    "correction": "fixed",
                }
            ],  # With error
        ]

        results = cleaner_collector.collect_batch(trajectories)

        assert len(results) == 2
        assert results[0] == trajectories[0]  # Unchanged
        assert results[1] != trajectories[1]  # Modified

        stats = cleaner_collector.get_statistics()
        assert stats["trajectories_processed"] == 1  # First trajectory in batch
        assert stats["trajectories_purified"] == 1


class TestCleanerPipeline:
    """Test CLEANER pipeline creation"""

    def test_create_cleaner_pipeline(self):
        pipeline = create_cleaner_pipeline(
            min_similarity=0.4, max_rollback_ratio=0.6, purification_enabled=True
        )

        assert pipeline.purification_enabled == True
        assert pipeline.saar_controller.min_similarity == 0.4
        assert pipeline.saar_controller.max_rollback_ratio == 0.6
        assert isinstance(
            pipeline.saar_controller.similarity_calculator, SemanticSimilarityCalculator
        )
        assert isinstance(
            pipeline.saar_controller.granularity_estimator, RollbackGranularityEstimator
        )

    def test_create_cleaner_pipeline_disabled(self):
        pipeline = create_cleaner_pipeline(purification_enabled=False)

        assert pipeline.purification_enabled == False


if __name__ == "__main__":
    pytest.main([__file__])
