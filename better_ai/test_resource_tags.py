"""
Resource tagging utilities for tests.

These decorators annotate unittest.TestCase classes (and their methods) with
resource usage metadata to enable automatic segregation and profiling.
"""


def high_resource(obj=None):
    """Mark a test or test class as high-resource."""


def decorator(o):
    setattr(o, "_high_resource", True)
    return o

    if obj is None:
        return decorator
    return decorator(obj)


def low_resource(obj=None):
    """Mark a test or test class as low-resource."""


def decorator(o):
    setattr(o, "_high_resource", False)
    return o

    if obj is None:
        return decorator
    return decorator(obj)
