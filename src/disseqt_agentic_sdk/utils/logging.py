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


def set_log_level(level: str) -> None:
    """
    Set the logging level for the SDK.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger = get_logger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    for handler in logger.handlers:
        handler.setLevel(numeric_level)
