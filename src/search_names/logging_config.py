"""Logging configuration for search_names package."""

import logging

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    rich_tracebacks: bool = True,
    show_time: bool = True,
    show_path: bool = False,
) -> logging.Logger:
    """Set up logging with rich formatting.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rich_tracebacks: Whether to use rich formatted tracebacks
        show_time: Whether to show timestamps
        show_path: Whether to show file paths in logs

    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create console for rich output
    console = Console(stderr=True)

    # Configure rich handler
    rich_handler = RichHandler(
        console=console,
        show_time=show_time,
        show_path=show_path,
        rich_tracebacks=rich_tracebacks,
        markup=True,
    )

    # Set format
    rich_handler.setFormatter(
        logging.Formatter(
            fmt="%(message)s",
            datefmt="[%X]",
        )
    )

    # Configure root logger
    logger = logging.getLogger("search_names")
    logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add rich handler
    logger.addHandler(rich_handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Optional logger name. If None, returns the main package logger.

    Returns:
        Logger instance
    """
    if name is None:
        return logging.getLogger("search_names")
    return logging.getLogger(f"search_names.{name}")
