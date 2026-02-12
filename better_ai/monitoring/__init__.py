"""HTSR Monitoring Package for Grokking Detection"""

from .htsr_monitor import (
    HTSRMonitor,
    compute_alpha_from_esd,
    detect_correlation_traps,
    GrokkingDetector,
    LayerAlphaAnalyzer,
    HTSRLogger,
    CommunicationStub,
    LogLevel,
    HTSRLog,
    ALPHA_OVER_GROKKING_THRESHOLD,
    ALPHA_VARIANCE_THRESHOLD,
)

from .dashboard import HTMLDashboard, SecureAuthenticator, LogLevel as DashboardLogLevel

__all__ = [
    "HTSRMonitor",
    "compute_alpha_from_esd",
    "detect_correlation_traps",
    "GrokkingDetector",
    "LayerAlphaAnalyzer",
    "HTSRLogger",
    "CommunicationStub",
    "LogLevel",
    "HTSRLog",
    "ALPHA_OVER_GROKKING_THRESHOLD",
    "ALPHA_VARIANCE_THRESHOLD",
    "HTMLDashboard",
    "SecureAuthenticator",
]
