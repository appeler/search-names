"""
Tests for logging_config module
"""

import logging
import unittest
from unittest.mock import MagicMock, patch

from search_names.logging_config import get_logger, setup_logging


class TestLoggingConfig(unittest.TestCase):
    def setUp(self):
        # Clear any existing loggers
        for logger_name in list(logging.Logger.manager.loggerDict.keys()):
            if logger_name.startswith("search_names"):
                del logging.Logger.manager.loggerDict[logger_name]

    def test_setup_logging_default(self):
        """Test setup_logging with default parameters."""
        logger = setup_logging()

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "search_names")
        self.assertEqual(logger.level, logging.INFO)
        self.assertGreater(len(logger.handlers), 0)

    def test_setup_logging_custom_level(self):
        """Test setup_logging with custom log level."""
        logger = setup_logging(level="DEBUG")

        self.assertEqual(logger.level, logging.DEBUG)

    def test_setup_logging_invalid_level(self):
        """Test setup_logging with invalid log level defaults to INFO."""
        logger = setup_logging(level="INVALID")

        self.assertEqual(logger.level, logging.INFO)

    def test_get_logger_default(self):
        """Test get_logger with no name."""
        setup_logging()
        logger = get_logger()

        self.assertEqual(logger.name, "search_names")

    def test_get_logger_with_name(self):
        """Test get_logger with specific name."""
        setup_logging()
        logger = get_logger("test_module")

        self.assertEqual(logger.name, "search_names.test_module")

    @patch("search_names.logging_config.Console")
    @patch("search_names.logging_config.RichHandler")
    def test_rich_handler_configuration(self, mock_rich_handler, mock_console):
        """Test that RichHandler is configured correctly."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        mock_handler = MagicMock()
        mock_rich_handler.return_value = mock_handler

        setup_logging(
            level="DEBUG", rich_tracebacks=True, show_time=True, show_path=False
        )

        # Verify Console was created
        mock_console.assert_called_once_with(stderr=True)

        # Verify RichHandler was created with correct parameters
        mock_rich_handler.assert_called_once_with(
            console=mock_console_instance,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )

    def test_logger_propagation(self):
        """Test that logger propagation is disabled."""
        logger = setup_logging()
        self.assertFalse(logger.propagate)

    def test_multiple_setup_calls(self):
        """Test that multiple setup_logging calls don't create duplicate handlers."""
        logger1 = setup_logging()
        handler_count1 = len(logger1.handlers)

        logger2 = setup_logging()
        handler_count2 = len(logger2.handlers)

        self.assertEqual(handler_count1, handler_count2)
        self.assertEqual(logger1, logger2)


if __name__ == "__main__":
    unittest.main()
