"""
Logging utilities for the ML Options Trading System.

Provides structured logging with file and console handlers.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Module-level logger cache
_loggers: dict[str, logging.Logger] = {}


def setup_logger(
    name: str = "trading_system",
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name (typically module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (optional)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console

    Returns:
        Configured logger instance
    """
    # Return cached logger if exists
    if name in _loggers:
        return _loggers[name]

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear any existing handlers
    logger.handlers = []

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    simple_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_to_file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Daily log file
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"{name}_{date_str}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    # Cache and return
    _loggers[name] = logger
    return logger


def get_logger(name: str = "trading_system") -> logging.Logger:
    """
    Get a logger instance.

    If the logger doesn't exist, creates a basic one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]

    # Create a basic logger if not set up
    return setup_logger(name, log_to_file=False)


class LogContext:
    """
    Context manager for adding context to log messages.

    Usage:
        with LogContext(logger, ticker="SPY", action="fetch"):
            logger.info("Fetching data")  # Will include ticker and action
    """

    def __init__(self, logger: logging.Logger, **context):
        """
        Initialize log context.

        Args:
            logger: Logger instance
            **context: Key-value pairs to add to log messages
        """
        self.logger = logger
        self.context = context
        self.old_factory = None

    def __enter__(self):
        """Add context to log records."""
        old_factory = logging.getLogRecordFactory()
        context = self.context

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            for key, value in context.items():
                setattr(record, key, value)
            return record

        self.old_factory = old_factory
        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original log record factory."""
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)


def log_execution_time(logger: logging.Logger):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance

    Returns:
        Decorator function
    """
    import functools
    import time

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"Starting {func.__name__}")

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.debug(f"Completed {func.__name__} in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Failed {func.__name__} after {elapsed:.2f}s: {e}")
                raise

        return wrapper
    return decorator


def log_dataframe_info(logger: logging.Logger, df, name: str = "DataFrame"):
    """
    Log information about a pandas DataFrame.

    Args:
        logger: Logger instance
        df: pandas DataFrame
        name: Name to use in log message
    """
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            logger.info(
                f"{name}: {len(df)} rows, {len(df.columns)} columns, "
                f"memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
            )
            if len(df) > 0:
                logger.debug(f"{name} columns: {list(df.columns)}")
                logger.debug(f"{name} date range: {df.index.min()} to {df.index.max()}")
    except Exception as e:
        logger.warning(f"Could not log DataFrame info: {e}")
