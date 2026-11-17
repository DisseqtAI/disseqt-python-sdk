"""Errors raised by the agent-testing data models."""


class ModelValidationError(ValueError):
    """A model was built (or loaded) with an invalid value.

    Messages are ``"<Class>.<field>: <problem>"`` (or a dotted path when loading data), so the
    author of a test case sees exactly which value to fix.
    """
