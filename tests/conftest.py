"""
Testing configuration and fixtures for pytest
Delegates to test_config_utils.py for configuration to avoid duplication
"""

import pytest


@pytest.fixture
def small_model_config():
    """
    Reduced ModelConfig for testing - 32-64x smaller parameters.
    Suitable for component verification without significant compute.
    """
    from better_ai.test_config_utils import get_small_model_config

    return get_small_model_config()


@pytest.fixture
def small_training_config():
    """
    Reduced TrainingConfig for testing.
    Suitable for verifying training loops with minimal compute.
    """
    from better_ai.test_config_utils import get_small_training_config

    return get_small_training_config()


@pytest.fixture
def small_inference_config():
    """
    Reduced InferenceConfig for testing.
    """
    from better_ai.test_config_utils import get_small_inference_config

    return get_small_inference_config()


@pytest.fixture
def model_config():
    """Default model config for tests using small defaults for CI safety."""
    from better_ai.test_config_utils import get_small_model_config

    return get_small_model_config()


@pytest.fixture
def training_config():
    """Default training config for tests using small defaults for CI safety."""
    from better_ai.test_config_utils import get_small_training_config

    return get_small_training_config()


@pytest.fixture
def inference_config():
    """Default inference config for tests using small defaults for CI safety."""
    from better_ai.test_config_utils import get_small_inference_config

    return get_small_inference_config()
