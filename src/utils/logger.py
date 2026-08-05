"""
Logger utility for AIC Video Retrieval System.
Uses loguru for structured, colored logging.
"""

import sys
from loguru import logger as _logger


def get_logger(name: str):
    """Return a named loguru logger with module context."""
    _logger.remove()
    _logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[name]}</cyan> — {message}",
        level="INFO",
        colorize=True,
    )
    return _logger.bind(name=name)
