"""
DEPRECATED: This file has been consolidated into test_rlhf_components.py

All component tests have been moved to test_rlhf_components.py for unified testing
of both RLHF and base model components.

Please run: python -m pytest tests/test_rlhf_components.py
"""

import warnings

warnings.warn(
    "test_components.py is deprecated. Use test_rlhf_components.py instead.",
    DeprecationWarning,
    stacklevel=2
)

# This file is kept for backwards compatibility but should not be used