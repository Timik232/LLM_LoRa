"""File for configuring logging with colored output."""
import logging
from logging import Formatter, LogRecord, StreamHandler
from typing import Dict

LOG_COLORS: Dict[str, str] = {
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
    handler = StreamHandler()
    handler.setFormatter(
        ColoredFormatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(handler)
