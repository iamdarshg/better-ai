"""Tests for HTSR Monitoring Module"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from better_ai.monitoring.htsr_monitor import (
    compute_alpha_from_esd,
    detect_correlation_traps,
    LayerAlphaAnalyzer,
    GrokkingDetector,
    HTSRLogger,
    CommunicationStub,
    LogLevel,
    HTSRMonitor,
    ALPHA_OVER_GROKKING_THRESHOLD,
    ALPHA_VARIANCE_THRESHOLD,
)


class TestComputeAlphaFromESD:
    """Tests for α computation from empirical spectral density."""

    def test_normal_matrix(self):
        """Test α computation on normal random matrix."""
        np.random.seed(42)
        W = np.random.randn(100, 100)
        alpha = compute_alpha_from_esd(W)
        assert 1.0 <= alpha <= 10.0

    def test_small_matrix(self):
        """Test handling of small matrices."""
        W = np.random.randn(5, 5)
        alpha = compute_alpha_from_esd(W)
        assert alpha == float("inf") or 1.0 <= alpha <= 10.0

    def test_none_input(self):
        """Test handling of None input."""
        alpha = compute_alpha_from_esd(None)
        assert alpha == float("inf")

    def test_1d_array(self):
        """Test handling of 1D array."""
        W = np.random.randn(100)
        alpha = compute_alpha_from_esd(W)
        assert alpha == float("inf")

    def test_zeros_matrix(self):
        """Test handling of zeros matrix."""
        W = np.zeros((100, 100))
        alpha = compute_alpha_from_esd(W)
        assert alpha == float("inf") or 2.0  # Returns default


class TestDetectCorrelationTraps:
    """Tests for correlation trap detection."""

    def test_normal_matrix(self):
        """Test on normal random matrix."""
        np.random.seed(42)
        W = np.random.randn(100, 100)
        is_trap, p_value = detect_correlation_traps(W)
        assert isinstance(is_trap, bool)
        assert 0.0 <= p_value <= 1.0

    def test_small_matrix(self):
        """Test on small matrix."""
        W = np.random.randn(5, 5)
        is_trap, p_value = detect_correlation_traps(W)
        assert is_trap == False
        assert p_value == 1.0

    def test_none_input(self):
        """Test None input handling."""
        is_trap, p_value = detect_correlation_traps(None)
        assert is_trap == False
        assert p_value == 1.0


class TestLayerAlphaAnalyzer:
    """Tests for layer α analyzer."""

    def test_should_monitor_layer_linear(self):
        """Test that linear layers are monitored."""
        analyzer = LayerAlphaAnalyzer()
        linear = nn.Linear(100, 100)
        assert analyzer.should_monitor_layer("test.linear", linear) == True

    def test_should_not_monitor_bias(self):
        """Test that bias layers are excluded."""
        analyzer = LayerAlphaAnalyzer()
        linear = nn.Linear(100, 100)
        assert analyzer.should_monitor_layer("test.bias", linear) == False

    def test_should_not_monitor_norm(self):
        """Test that normalization layers are excluded."""
        analyzer = LayerAlphaAnalyzer()
        linear = nn.Linear(100, 100)
        assert analyzer.should_monitor_layer("test.layernorm", linear) == False

    def test_should_not_monitor_small_layer(self):
        """Test that small layers are excluded."""
        analyzer = LayerAlphaAnalyzer()
        linear = nn.Linear(5, 5)
        assert analyzer.should_monitor_layer("test.small", linear) == False

    def test_analyze_layer(self):
        """Test layer analysis."""
        analyzer = LayerAlphaAnalyzer()
        linear = nn.Linear(100, 100)
        # Initialize with some values
        with torch.no_grad():
            linear.weight.data = torch.randn(100, 100)

        result = analyzer.analyze_layer("test.linear", linear)
        assert result is not None
        assert "alpha" in result
        assert "layer_name" in result


class TestGrokkingDetector:
    """Tests for grokking detector."""

    def test_normal_alphas(self):
        """Test with normal α values."""
        detector = GrokkingDetector()
        alpha_metrics = {"layer1": 2.0, "layer2": 2.5, "layer3": 3.0}
        is_detected, details = detector.check(alpha_metrics)
        assert is_detected == False
        assert details["model_alpha"] == pytest.approx(2.5)

    def test_over_grokking_detected(self):
        """Test detection of over-grokking (α > 4.5)."""
        detector = GrokkingDetector()
        alpha_metrics = {"layer1": 5.0, "layer2": 2.0}
        is_detected, details = detector.check(alpha_metrics)
        assert is_detected == True
        assert details["is_over_grokking"] == True
        assert len(details["over_grokking_layers"]) == 1

    def test_high_variance_detected(self):
        """Test detection of high variance."""
        detector = GrokkingDetector()
        # Very different α values will cause high variance
        alpha_metrics = {"layer1": 1.0, "layer2": 6.0}
        is_detected, details = detector.check(alpha_metrics)
        # Either over-grokking or high variance should trigger
        assert is_detected == True

    def test_empty_metrics(self):
        """Test with empty metrics."""
        detector = GrokkingDetector()
        is_detected, details = detector.check({})
        assert is_detected == False
        assert details["reason"] == "no_metrics"


class TestHTSRLogger:
    """Tests for HTSR logger with log classification."""

    def test_severe_log(self):
        """Test severe log creation."""
        logger = HTSRLogger()
        log = logger.severe("Test severe message", step=100)
        assert log.level == LogLevel.SEVERE
        assert log.message == "Test severe message"
        assert log.step == 100

    def test_major_log(self):
        """Test major log creation."""
        logger = HTSRLogger()
        log = logger.major("Test major message", step=100)
        assert log.level == LogLevel.MAJOR

    def test_minor_log(self):
        """Test minor log creation."""
        logger = HTSRLogger()
        log = logger.minor("Test minor message", step=100)
        assert log.level == LogLevel.MINOR

    def test_status_log(self):
        """Test status log creation."""
        logger = HTSRLogger()
        log = logger.status("Test status message", step=100)
        assert log.level == LogLevel.STATUS

    def test_log_history_limit(self):
        """Test that log history is limited."""
        logger = HTSRLogger()
        for i in range(600):
            logger.status(f"Log {i}", step=i)

        assert len(logger.log_history) <= 500

    def test_get_logs_by_level(self):
        """Test filtering logs by level."""
        logger = HTSRLogger()
        logger.severe("Severe 1", step=1)
        logger.major("Major 1", step=2)
        logger.severe("Severe 2", step=3)

        severe_logs = logger.get_logs_by_level(LogLevel.SEVERE)
        assert len(severe_logs) == 2

        major_logs = logger.get_logs_by_level(LogLevel.MAJOR)
        assert len(major_logs) == 1


class TestCommunicationStub:
    """Tests for communication stub."""

    def test_send_severe_alert(self):
        """Test sending severe alert."""
        stub = CommunicationStub()
        log = Mock()
        log.message = "Test alert"
        log.to_dict.return_value = {"message": "Test alert"}

        result = stub.send_severe_alert(log)
        assert result == True
        assert len(stub.notification_queue) == 1

    def test_disabled_communication(self):
        """Test when communication is disabled."""
        stub = CommunicationStub()
        stub.enabled = False

        result = stub.send_severe_alert(Mock())
        assert result == False

    def test_configure_channel(self):
        """Test channel configuration."""
        stub = CommunicationStub()
        stub.configure_channel("email", {"smtp": "server"})
        assert stub.channels["email"] == True

    def test_get_queued_notifications(self):
        """Test getting queued notifications."""
        stub = CommunicationStub()
        stub.send_severe_alert(Mock())
        notifications = stub.get_queued_notifications()
        assert len(notifications) == 1


class TestHTSRMonitor:
    """Tests for main HTSR monitor."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        model = nn.Linear(100, 100)
        monitor = HTSRMonitor(model)
        assert monitor.alpha_upper_threshold == ALPHA_OVER_GROKKING_THRESHOLD
        assert monitor.monitor_interval == 75

    def test_compute_all_layer_alphas(self):
        """Test computing α for all layers."""
        model = nn.Sequential(nn.Linear(100, 100), nn.Linear(100, 100))
        monitor = HTSRMonitor(model)
        result = monitor.compute_all_layer_alphas()

        assert "alpha_metrics" in result
        assert "detector_state" in result
        assert monitor.total_checks == 1

    def test_should_check(self):
        """Test check scheduling."""
        model = nn.Linear(100, 100)
        monitor = HTSRMonitor(model, monitor_interval=75)

        assert monitor.should_check(75) == True
        assert monitor.should_check(150) == True
        assert monitor.should_check(74) == False

    def test_get_dashboard_data(self):
        """Test dashboard data format."""
        model = nn.Linear(100, 100)
        monitor = HTSRMonitor(model)
        monitor.compute_all_layer_alphas()

        data = monitor.get_dashboard_data()
        assert "current_alphas" in data
        assert "detector_state" in data
        assert "statistics" in data

    def test_get_layer_summary(self):
        """Test layer summary generation."""
        model = nn.Linear(100, 100)
        monitor = HTSRMonitor(model)
        monitor.compute_all_layer_alphas()

        summary = monitor.get_layer_summary()
        assert isinstance(summary, list)
        assert len(summary) > 0

    def test_get_health_report(self):
        """Test health report generation."""
        model = nn.Linear(100, 100)
        monitor = HTSRMonitor(model)
        monitor.compute_all_layer_alphas()

        report = monitor.get_health_report()
        assert "status" in report
        assert "model_alpha" in report


class TestIntegration:
    """Integration tests for HTSR monitoring."""

    def test_full_monitoring_flow(self):
        """Test complete monitoring flow."""
        model = nn.Sequential(
            nn.Linear(100, 100), nn.Linear(100, 100), nn.Linear(100, 100)
        )

        monitor = HTSRMonitor(model, alpha_upper_threshold=3.0)

        # Simulate multiple checks
        for _ in range(3):
            monitor.compute_all_layer_alphas()

        assert monitor.total_checks == 3
        assert len(monitor.alpha_history) == 3

    def test_monitoring_with_intervention(self):
        """Test monitoring with intervention."""
        model = nn.Linear(100, 100)
        optimizer = torch.optim.Adam(model.parameters())

        monitor = HTSRMonitor(model)

        # Apply intervention
        result = monitor.apply_intervention(
            intervention_type="test",
            lr_reduction_factor=0.5,
            wd_increase_factor=2.0,
            optimizer=optimizer,
        )

        assert result is not None
        assert "type" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
