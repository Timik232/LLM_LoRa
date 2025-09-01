"""File for configuring logging with colored output."""

import logging
from logging import Formatter, LogRecord, StreamHandler

"""
Enhanced logging configuration and utilities for LLM LoRa training framework.

This module provides:
- Centralized logging configuration with colored output
- Named logger instances for better debugging
- Utility functions for structured logging
- Standardized training progress logging

Usage:
    from training_model.logging_config import configure_logging, get_logger

    configure_logging(logging.DEBUG)
    logger = get_logger(__name__)
    logger.info("Module initialized")
"""
LOG_COLORS: dict[str, str] = {
    "DEBUG": "#4b8bf5",  # Light blue
    "INFO": "#2ecc71",  # Green
    "WARNING": "#f1c40f",  # Yellow
    "ERROR": "#e74c3c",  # Red
    "CRITICAL": "#8b0000",  # Dark red
}
RESET_COLOR = "\x1b[0m"


def hex_to_ansi(hex_color: str) -> str:
    """Convert hexadecimal color code to ANSI escape sequence.

    Args:
        hex_color (str): Hexadecimal color code in format '#RRGGBB'

    Returns:
        str: ANSI escape sequence for the color, or empty string if conversion fails
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return ""

    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError:
        return ""

    return f"\x1b[38;2;{r};{g};{b}m"


class ColoredFormatter(Formatter):
    """Custom formatter that adds color to log messages using ANSI escape codes.
    The colors are determined by the LOG_COLORS mapping based on log level."""

    def format(self, record: LogRecord) -> str:
        """Format the specified log record with color.

        Args:
            record (LogRecord): The log record to be formatted

        Returns:
            str: Formatted log message with color codes
        """
        color_code = hex_to_ansi(LOG_COLORS.get(record.levelname, ""))
        message = super().format(record)
        return f"{color_code}{message}{RESET_COLOR}"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with colored output handler.

    Args:
        level (int): Logging level to set (logging.INFO or logging.DEBUG).
            Defaults to logging.INFO.

    Raises:
        ValueError: If level is not logging.INFO or logging.DEBUG
    """
    if level != logging.INFO and level != logging.DEBUG:
        raise ValueError("You can use only logging.info or logging.debug")

    # Clear existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = StreamHandler()
    handler.setFormatter(
        ColoredFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    # Suppress third-party library noise
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance.

    Args:
        name (str): Logger name, typically __name__ of the calling module

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


def log_dict(logger: logging.Logger, data: dict, level: int = logging.INFO, prefix: str = ""):
    """Log dictionary contents in a structured format.

    Args:
        logger: Logger instance to use
        data: Dictionary to log
        level: Log level to use
        prefix: Optional prefix for the log message
    """
    if not data:
        return

    logger.log(level, f"{prefix}Configuration:" if prefix else "Configuration:")
    for key, value in data.items():
        if isinstance(value, dict):
            logger.log(level, f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.log(level, f"    {sub_key}: {sub_value}")
        else:
            logger.log(level, f"  {key}: {value}")


def log_training_progress(
    logger: logging.Logger,
    step: int,
    total_steps: int,
    loss: float = None,
    metrics: dict = None,
):
    """Log training progress in a standardized format.

    Args:
        logger: Logger instance to use
        step: Current training step
        total_steps: Total number of training steps
        loss: Current loss value (optional)
        metrics: Additional metrics dictionary (optional)
    """
    progress_pct = (step / total_steps) * 100 if total_steps > 0 else 0
    base_msg = f"Step {step}/{total_steps} ({progress_pct:.1f}%)"

    if loss is not None:
        base_msg += f" - Loss: {loss:.4f}"

    if metrics:
        metric_strs = [
            f"{k}: {v:.4f}" if isinstance(v, int | float) else f"{k}: {v}"
            for k, v in metrics.items()
        ]
        if metric_strs:
            base_msg += f" - {', '.join(metric_strs)}"

    logger.info(base_msg)
