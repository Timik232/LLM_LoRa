import logging
from logging import Formatter, StreamHandler
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
    """Convert hexadecimal color code to ANSI escape sequence"""
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
    """Custom formatter with true color support"""

    def format(self, record):
        color_code = hex_to_ansi(LOG_COLORS.get(record.levelname, ""))
        message = super().format(record)
        return f"{color_code}{message}{RESET_COLOR}"


def configure_logging(level=logging.INFO):
    """Set up colored logging configuration\n
    level should be like: logging:INFO or logging.DEBUG"""
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
