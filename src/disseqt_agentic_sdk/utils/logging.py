"""
Structured logging utilities for the SDK.
"""

import logging
import sys

# Default logger name
DEFAULT_LOGGER_NAME = "disseqt_agentic_sdk"


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance with structured formatting.

    Args:
        name: Logger name (defaults to SDK logger name)

    Returns:
        Configured logger instance
    """
    logger_name = name or DEFAULT_LOGGER_NAME
    logger = logging.getLogger(logger_name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler with structured format
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.INFO)

        # Structured formatter: timestamp level logger message
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def set_log_level(level: str | int) -> None:
    """
    Set the logging level for the SDK.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) as string or int
    """
    # Handle both string and int inputs
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    elif isinstance(level, int):
        numeric_level = level
    else:
        # Default to INFO if invalid type
        numeric_level = logging.INFO

    # Set level on root SDK logger
    root_logger = get_logger()
    root_logger.setLevel(numeric_level)
    for handler in root_logger.handlers:
        handler.setLevel(numeric_level)

    # Also set level on all existing SDK loggers (since they have propagate=False)
    for name in logging.root.manager.loggerDict:
        if name.startswith(DEFAULT_LOGGER_NAME):
            logger = logging.getLogger(name)
            logger.setLevel(numeric_level)
            for handler in logger.handlers:
                handler.setLevel(numeric_level)
