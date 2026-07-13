"""AuraTrainer Logging System."""

import logging
import sys
from pathlib import Path
from typing import Optional


class ColorFormatter(logging.Formatter):
    """Colored console formatter."""

    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        record.color = color
        record.reset = self.RESET
        return super().format(record)


def setup_logger(
    name: str = "AuraTrainer",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: str = "logs",
) -> logging.Logger:
    """Setup and configure logger.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional log file path.
        log_dir: Directory for log files.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    console_fmt = ColorFormatter(
        fmt="%(color)s[%(asctime)s] %(levelname)-8s%(reset)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    if log_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_path = Path(log_dir) / log_file
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_fmt = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "AuraTrainer") -> logging.Logger:
    """Get existing logger or create default one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
