"""
Logging configuration for easy-oww
"""
import logging
from pathlib import Path
from rich.logging import RichHandler


def setup_logger(
    name: str = 'easy-oww',
    level: int = logging.INFO,
    log_file: Path = None
) -> logging.Logger:
    """
    Configure logging with rich handler

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Rich console handler
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=False,
        show_path=False
    )
    console_formatter = logging.Formatter(
        '%(message)s',
        datefmt='[%X]'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = 'easy-oww') -> logging.Logger:
    """
    Get logger instance

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
