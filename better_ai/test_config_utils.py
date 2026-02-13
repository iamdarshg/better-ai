"""
Test utilities for reduced configuration fixtures.
Provides a function-based approach for unittest compatibility.
"""

from better_ai.config import ModelConfig, TrainingConfig, InferenceConfig


def get_small_model_config():
    """
    Get a reduced ModelConfig for testing.
    Deduplicated and now uses the central ModelConfig.get_small_model_config().
    """
    return ModelConfig.get_small_model_config()


def get_small_training_config():
    """
    Get a reduced TrainingConfig for testing.
    Deduplicated and now uses TrainingConfig.get_small_training_config().
    """
    return TrainingConfig.get_small_training_config()


def get_small_inference_config():
    """
    Get a reduced InferenceConfig for testing.
    Deduplicated and now uses InferenceConfig.get_small_inference_config().
    """
    return InferenceConfig.get_small_inference_config()
