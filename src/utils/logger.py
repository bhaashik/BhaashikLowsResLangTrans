"""Logging configuration for the translation system."""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    console: bool = True,
    format_string: Optional[str] = None
) -> None:
    """
    Set up logging configuration.

    Args:
        log_file: Path to log file. If None, only console logging is enabled.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Whether to log to console
        format_string: Custom format string. If None, uses default.
    """
    # Remove default logger
    logger.remove()

    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add console handler
    if console:
        logger.add(
            sys.stderr,
            format=format_string,
            level=level,
            colorize=True
        )

    # Add file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation="100 MB",
            retention="30 days",
            compression="zip"
        )

    logger.info(f"Logger initialized with level {level}")


def get_logger(name: str = None):
    """
    Get a logger instance.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger
