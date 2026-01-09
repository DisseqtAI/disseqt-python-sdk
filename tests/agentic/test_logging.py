"""
Unit tests for logging utilities.
"""

import logging
from io import StringIO

from disseqt_agentic_sdk.utils.logging import get_logger, set_log_level


class TestLogging:
    """Tests for logging utilities."""

    def test_get_logger(self):
        """Test getting logger instance."""
        logger = get_logger()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "disseqt_agentic_sdk"

    def test_get_logger_custom_name(self):
        """Test getting logger with custom name."""
        logger = get_logger("custom_logger")
        assert logger.name == "custom_logger"

    def test_logger_has_handler(self):
        """Test logger has handler configured."""
        logger = get_logger()
        assert len(logger.handlers) > 0
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_logger_output(self):
        """Test logger outputs messages."""
        logger = get_logger("test_logger")

        # Capture output
        handler = logging.StreamHandler(StringIO())
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info("Test message")

        # Verify handler received message
        assert len(handler.stream.getvalue()) > 0

    def test_set_log_level(self):
        """Test setting log level."""
        logger = get_logger()

        set_log_level("DEBUG")
        assert logger.level == logging.DEBUG

        set_log_level("INFO")
        assert logger.level == logging.INFO

        set_log_level("WARNING")
        assert logger.level == logging.WARNING

        set_log_level("ERROR")
        assert logger.level == logging.ERROR

    def test_logger_structure(self):
        """Test logger uses structured format."""
        logger = get_logger("structured_test")

        # Check formatter exists
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                formatter = handler.formatter
                assert formatter is not None
                # Format should include timestamp, level, name, message
                assert "%(asctime)s" in formatter._fmt
                assert "%(levelname)s" in formatter._fmt
                assert "%(name)s" in formatter._fmt
                assert "%(message)s" in formatter._fmt
