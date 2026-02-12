"""HTSR Monitoring Package for Anti-Grokking Detection"""

from .htsr_monitor import (
    HTSRMonitor,
    compute_alpha_from_esd,
    detect_correlation_traps,
    AntiGrokkingDetector,
    LayerAlphaAnalyzer,
)

from .dashboard import HTMLDashboard

__all__ = [
    "HTSRMonitor",
    "compute_alpha_from_esd",
    "detect_correlation_traps",
    "AntiGrokkingDetector",
    "LayerAlphaAnalyzer",
    "HTMLDashboard",
]
