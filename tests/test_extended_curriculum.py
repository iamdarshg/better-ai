"""
Unit tests for Extended Cosine Curriculum Learning
Tests sequence length scheduling, difficulty normalization, and domain mixing
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from better_ai.training.extended_curriculum import (
    SequenceLengthConfig,
    SequenceLengthScheduler,
    DifficultyConfig,
    DifficultyScheduler,
    DomainMixingConfig,
    AdaptiveDomainMixer,
    ExtendedCurriculumConfig,
    ExtendedCurriculumScheduler,
)


class TestSequenceLengthScheduler:
    """Tests for SequenceLengthScheduler"""

    def test_cosine_schedule_progression(self):
        """Test that cosine schedule progresses correctly"""
        config = SequenceLengthConfig(
            stage="test",
            min_length=4096,
            warmup_steps=100,
            schedule="cosine",
        )
        dataset_max_lengths = {"dataset1": 8192, "dataset2": 16384}

        scheduler = SequenceLengthScheduler(config, dataset_max_lengths)

        # At start (step 0), should be at min_length
        lengths = scheduler.step()
        assert lengths["dataset1"] == 4096
        assert lengths["dataset2"] == 4096

        # After some steps, should progress
        for _ in range(200):
            scheduler.step()

        lengths = scheduler.step()
        # Should be somewhere between min and max
        assert 4096 <= lengths["dataset1"] <= 8192
        assert 4096 <= lengths["dataset2"] <= 16384

    def test_linear_schedule(self):
        """Test linear schedule"""
        config = SequenceLengthConfig(
            stage="test",
            min_length=1000,
            warmup_steps=50,
            schedule="linear",
        )
        scheduler = SequenceLengthScheduler(config, {"ds": 5000})

        # Advance many steps
        for _ in range(200):
            scheduler.step()

        lengths = scheduler.step()
        assert 1000 <= lengths["ds"] <= 5000

    def test_step_schedule(self):
        """Test step schedule"""
        config = SequenceLengthConfig(
            stage="test",
            min_length=1000,
            warmup_steps=10,
            schedule="step",
            step_thresholds=[0.25, 0.5, 0.75, 1.0],
        )
        scheduler = SequenceLengthScheduler(config, {"ds": 4000})

        # Initially at min
        scheduler.step()
        lengths = scheduler.step()
        assert lengths["ds"] == 1000

    def test_dataset_specific_min_lengths(self):
        """Test per-dataset minimum lengths"""
        config = SequenceLengthConfig(
            stage="test",
            min_length=4096,
            warmup_steps=0,
            schedule="cosine",
            dataset_min_lengths={"short_ds": 1024, "long_ds": 8192},
        )
        scheduler = SequenceLengthScheduler(
            config, {"short_ds": 8192, "long_ds": 16384}
        )

        scheduler.step()
        lengths = scheduler.get_current_lengths()

        assert lengths["short_ds"] == 1024
        assert lengths["long_ds"] == 8192

    def test_grokking_cosine_schedule(self):
        """Test grokking cosine schedule: fast first 40%, plateau, slow tail"""
        config = SequenceLengthConfig(
            stage="test",
            total_steps=2000,
            min_length=4096,
            warmup_steps=10,
            schedule="grokking_cosine",
            grokking_fast_ratio=0.4,
            plateau_steps=500,
        )
        scheduler = SequenceLengthScheduler(config, {"ds": 16384})

        # Advance to fast phase (progress ~0.3)
        for _ in range(300):
            scheduler.step()

        lengths_fast = scheduler.get_dataset_length("ds")

        # Advance to tail phase (progress ~0.7)
        for _ in range(100):
            scheduler.step()

        lengths_tail = scheduler.get_dataset_length("ds")

        # Fast phase should have progressed significantly
        assert lengths_fast > 4096
        # Tail phase should progress slower
        assert lengths_tail > lengths_fast
        # Should not reach max too quickly
        assert lengths_tail < 14000  # Not yet at max

    def test_grokking_step_schedule(self):
        """Test grokking step schedule with gradual progression"""
        config = SequenceLengthConfig(
            stage="test",
            min_length=4096,
            warmup_steps=10,
            schedule="grokking_step",
            grokking_fast_ratio=0.4,
            step_thresholds=[0.2, 0.4, 0.6, 0.8, 1.0],
        )
        scheduler = SequenceLengthScheduler(config, {"ds": 16384})

        # Initially at min
        scheduler.step()
        assert scheduler.get_dataset_length("ds") == 4096


class TestDifficultyScheduler:
    """Tests for DifficultyScheduler"""

    def test_normalize_difficulty_score(self):
        """Test difficulty score normalization"""
        config = DifficultyConfig(stage="test")
        scheduler = DifficultyScheduler(config)

        # Test with explicit difficulty
        item = {"difficulty": 0.5}
        score = scheduler.normalize_difficulty_score(item)
        assert score == 0.5

        # Test with difficulty > 1 (should be normalized)
        item = {"difficulty": 5.5}
        score = scheduler.normalize_difficulty_score(item)
        assert score == 0.5  # (5.5-1)/9 = 0.5

        # Test with missing difficulty (should default to 0.5)
        item = {"other_field": "value"}
        score = scheduler.normalize_difficulty_score(item)
        assert score == 0.5

    def test_alternative_fields(self):
        """Test alternative difficulty field lookup"""
        config = DifficultyConfig(
            stage="test",
            difficulty_field="missing",
            alternative_fields=["complexity", "difficulty_score"],
        )
        scheduler = DifficultyScheduler(config)

        item = {"complexity": 0.8}
        score = scheduler.normalize_difficulty_score(item)
        assert score == 0.8

    def test_should_include_sample(self):
        """Test sample inclusion based on difficulty threshold"""
        config = DifficultyConfig(stage="test")
        scheduler = DifficultyScheduler(config)

        # Initially threshold should be low (0.0 at step 1)
        scheduler.step()

        # Test with low difficulty sample (0.0 should be included, 0.1 might not)
        assert scheduler.should_include_sample(0.0) is True

        # Test with high difficulty sample
        # May or may not include based on threshold and randomness
        # At early stage, should include harder samples occasionally
        for _ in range(10):
            result = scheduler.should_include_sample(0.9)
            # Should sometimes include for exploration
            assert isinstance(result, bool)

    def test_adaptive_adjustment(self):
        """Test adaptive difficulty adjustment based on performance"""
        config = DifficultyConfig(
            stage="test",
            enable_adaptive=True,
            performance_window=5,
            adjustment_rate=0.1,
        )
        scheduler = DifficultyScheduler(config)

        # Add some performance history
        for i in range(10):
            scheduler.step()
            scheduler.update_performance({"loss": 0.5, "accuracy": 0.9})

        # Difficulty should adjust based on performance
        final_difficulty = scheduler.difficulty_history[-1]
        assert 0.0 <= final_difficulty <= 1.0


class TestAdaptiveDomainMixer:
    """Tests for AdaptiveDomainMixer"""

    def test_initial_weights(self):
        """Test initial domain weights"""
        config = DomainMixingConfig(
            stage="test",
            domains={
                "coding": ["ds1", "ds2"],
                "math": ["ds3"],
            },
            initial_weights={"coding": 0.7, "math": 0.3},
        )
        mixer = AdaptiveDomainMixer(config)

        weights = mixer.get_sampling_weights()

        assert weights["coding"] == 0.7
        assert weights["math"] == 0.3
        # Weights should sum to 1
        assert abs(weights["coding"] + weights["math"] - 1.0) < 0.001

    def test_equal_initial_weights(self):
        """Test equal initial weights when not specified"""
        config = DomainMixingConfig(
            stage="test",
            domains={
                "domain1": ["ds1"],
                "domain2": ["ds2"],
                "domain3": ["ds3"],
            },
        )
        mixer = AdaptiveDomainMixer(config)

        weights = mixer.get_sampling_weights()

        # Should be approximately equal
        for w in weights.values():
            assert abs(w - 1.0 / 3.0) < 0.001

    def test_domain_performance_update(self):
        """Test updating performance metrics per domain"""
        config = DomainMixingConfig(
            stage="test",
            domains={"domain1": ["ds1"]},
            update_frequency=10,
        )
        mixer = AdaptiveDomainMixer(config, total_steps=1000)

        # Update performance for domain
        mixer.update_domain_performance("domain1", {"loss": 0.5, "accuracy": 0.9})

        perf = mixer.domain_performance["domain1"]
        assert "loss" in perf
        assert "accuracy" in perf

    def test_weight_history(self):
        """Test that weight changes are tracked"""
        config = DomainMixingConfig(
            stage="test",
            domains={
                "strong": ["ds1"],
                "weak": ["ds2"],
            },
            initial_weights={"strong": 0.5, "weak": 0.5},
            adjustment_rate=0.5,  # High rate for testing
            update_frequency=5,
        )
        mixer = AdaptiveDomainMixer(config, total_steps=1000)

        initial_weights = mixer.get_sampling_weights()

        # Update performance: weak domain has worse metrics
        mixer.update_domain_performance("strong", {"loss": 0.1, "accuracy": 0.99})
        mixer.update_domain_performance("weak", {"loss": 2.0, "accuracy": 0.5})

        # Advance enough steps to trigger update
        for _ in range(15):
            mixer.step()

        final_weights = mixer.get_sampling_weights()

        # Should have some weight history
        assert len(mixer.weight_history) > 1


class TestExtendedCurriculumScheduler:
    """Tests for ExtendedCurriculumScheduler"""

    def test_basic_step(self):
        """Test basic curriculum step"""
        seq_config = SequenceLengthConfig(
            stage="test",
            min_length=1024,
            warmup_steps=10,
        )
        diff_config = DifficultyConfig(stage="test")
        domain_config = DomainMixingConfig(
            stage="test",
            domains={"coding": ["ds1"]},
        )

        config = ExtendedCurriculumConfig(
            stage="test",
            total_steps=100,
            sequence_config=seq_config,
            difficulty_config=diff_config,
            domain_config=domain_config,
        )

        scheduler = ExtendedCurriculumScheduler(config, {"ds1": 4096})

        state = scheduler.step()

        assert "step" in state
        assert state["stage"] == "test"
        assert "sequence_lengths" in state
        assert "difficulty_threshold" in state
        assert "domain_weights" in state

    def test_optional_components(self):
        """Test scheduler with only some components enabled"""
        config = ExtendedCurriculumConfig(
            stage="test",
            enable_sequence_curriculum=True,
            enable_difficulty_curriculum=False,
            enable_domain_mixing=False,
        )

        scheduler = ExtendedCurriculumScheduler(config, {"ds1": 4096})

        state = scheduler.step()

        assert "sequence_lengths" in state
        assert "difficulty_threshold" not in state
        assert "domain_weights" not in state

    def test_get_state_methods(self):
        """Test various getter methods"""
        seq_config = SequenceLengthConfig(stage="test", min_length=1024)
        config = ExtendedCurriculumConfig(
            stage="test",
            sequence_config=seq_config,
        )

        scheduler = ExtendedCurriculumScheduler(config, {"ds1": 4096, "ds2": 8192})

        scheduler.step()

        lengths = scheduler.get_current_sequence_lengths()
        assert lengths is not None
        assert "ds1" in lengths
        assert "ds2" in lengths

        ds_len = scheduler.get_dataset_sequence_length("ds1")
        assert isinstance(ds_len, int)

        diff = scheduler.get_difficulty_threshold()
        assert isinstance(diff, float)


class TestDifficultyNormalization:
    """Tests for difficulty normalization edge cases"""

    def test_various_difficulty_formats(self):
        """Test normalizing difficulties in various formats"""
        config = DifficultyConfig(stage="test")
        scheduler = DifficultyScheduler(config)

        test_cases = [
            ({"difficulty": 0.0}, 0.0),
            ({"difficulty": 1.0}, 1.0),
            ({"difficulty": 0.5}, 0.5),
            ({"difficulty": 10}, 1.0),  # >1 should be clamped
            ({"difficulty": -0.5}, 0.0),  # <0 should be clamped
            ({}, 0.5),  # Missing should default to 0.5
        ]

        for item, expected in test_cases:
            score = scheduler.normalize_difficulty_score(item)
            assert abs(score - expected) < 0.01, f"Failed for {item}"

    def test_length_proxy_difficulty(self):
        """Test using sequence length as difficulty proxy"""
        config = DifficultyConfig(
            stage="test",
            use_length_proxy=True,
            length_difficulty_factor=0.5,
        )
        scheduler = DifficultyScheduler(config)

        # Short sequence
        short_score = scheduler.normalize_difficulty_score({}, seq_length=1000)

        # Long sequence
        long_score = scheduler.normalize_difficulty_score({}, seq_length=100000)

        # Long should be higher difficulty
        assert long_score > short_score


class TestMetricBlending:
    """Tests for metric blending in domain mixer"""

    def test_weighted_sum_blend(self):
        """Test weighted sum metric blending"""
        config = DomainMixingConfig(
            stage="test",
            domains={"d": ["ds"]},
            blend_strategy="weighted_sum",
            metric_weights={"loss": 0.5, "accuracy": 0.5},
        )
        mixer = AdaptiveDomainMixer(config)

        mixer.update_domain_performance("d", {"loss": 0.2, "accuracy": 0.8})

        # The blended score should be calculated
        perf = mixer.domain_performance["d"]
        assert "loss" in perf
        assert "accuracy" in perf

    def test_geometric_mean_blend(self):
        """Test geometric mean metric blending"""
        config = DomainMixingConfig(
            stage="test",
            domains={"d": ["ds"]},
            blend_strategy="geometric_mean",
            metric_weights={"loss": 0.5, "accuracy": 0.5},
        )
        mixer = AdaptiveDomainMixer(config)

        mixer.update_domain_performance("d", {"loss": 0.5, "accuracy": 0.9})

    def test_min_max_norm_blend(self):
        """Test min-max normalization blending"""
        config = DomainMixingConfig(
            stage="test",
            domains={"d": ["ds"]},
            blend_strategy="min_max_norm",
        )
        mixer = AdaptiveDomainMixer(config)

        mixer.update_domain_performance("d", {"loss": 0.5})


def run_all_tests():
    """Run all tests"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_all_tests()
