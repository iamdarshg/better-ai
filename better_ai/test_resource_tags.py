"""
Resource tagging utilities for tests.

These decorators annotate unittest.TestCase classes (and their methods) with
resource usage metadata to enable automatic segregation and profiling.
"""

import unittest


def high_resource(obj=None):
    """Mark a test or test class as high-resource."""

    def decorator(o):
        # Set class-level attribute
        setattr(o, "_is_high_resource", True)

        # Also patch setUp to set instance attribute for runtime access
        original_setUp = getattr(o, "setUp", None)

        def setUp(self_instance):
            setattr(self_instance, "_is_high_resource", True)
            if original_setUp:
                original_setUp(self_instance)

        o.setUp = setUp

        return o

    if obj is None:
        return decorator
    return decorator(obj)


def low_resource(obj=None):
    """Mark a test or test class as low-resource."""

    def decorator(o):
        # Set class-level attribute
        setattr(o, "_is_low_resource", True)

        # Also patch setUp to set instance attribute for runtime access
        original_setUp = getattr(o, "setUp", None)

        def setUp(self_instance):
            setattr(self_instance, "_is_low_resource", True)
            if original_setUp:
                original_setUp(self_instance)

        o.setUp = setUp

        return o

    if obj is None:
        return decorator
    return decorator(obj)


def low_resource(obj=None):
    """Mark a test or test class as low-resource."""

    def decorator(o):
        # Set class-level attribute
        setattr(o, "_is_low_resource", True)

        # Also patch setUp to set instance attribute for runtime access
        original_setUp = getattr(o, "setUp", None)

        def setUp(self):
            setattr(self, "_is_low_resource", True)
            if original_setUp:
                original_setUp(self)

        o.setUp = setUp

        return o

    if obj is None:
        return decorator
    return decorator(obj)
