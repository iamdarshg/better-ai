"""
Custom exceptions for the Better AI project.
"""

class BetterAIError(Exception):
    """Base exception class for all Better AI errors."""
    pass

class ConfigError(BetterAIError):
    """Raised for configuration-related errors."""
    pass

class DataError(BetterAIError):
    """Raised for data loading or processing errors."""
    pass

class ModelError(BetterAIError):
    """Raised for model-related errors."""
    pass

class TrainingError(BetterAIError):
    """Raised for training-related errors."""
    pass

class EvaluationError(BetterAIError):
    """Raised for evaluation-related errors."""
    pass
