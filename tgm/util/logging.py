import functools
import logging
import time
from pathlib import Path
from typing import Any, Callable, List


def enable_logging(
    *,
    console_log_level: int = logging.INFO,
    file_log_level: int = logging.DEBUG,
    log_file_path: str | Path | None = None,
) -> None:
    """Enable library-wide logging to stdout and optionally to a file.

    Args:
        console_log_level (int): Logging level for console stream handler (default = logging.INFO).
        file_log_level (int): Logging level for file handler if configured (default = logging.DEBUG).
        log_file_path (Optional[str | Path]): Optional path to a log file.
    """
    logger = logging.getLogger('tgm')
    logger.handlers.clear()  # Clear existing handlers, making this idempotent

    console_formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(console_formatter)

    handlers: List[logging.Handler] = [console_handler]
    if log_file_path is not None:
        file_formatter = logging.Formatter(
            '[%(asctime)s.%(msecs)03d] %(name)s - %(levelname)s '
            '[%(processName)s %(threadName)s %(name)s.%(funcName)s:%(lineno)d] '
            '%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        file_handler = logging.FileHandler(filename=log_file_path, mode='a')
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    for handler in handlers:
        logger.addHandler(handler)

    logger.setLevel(min(console_log_level, file_log_level))
    logger.propagate = False  # Don't spam user's root logger


def log_latency(_func: Callable | None = None, *, level: int = logging.INFO) -> Any:
    """Function decorator to log latency at configurable log level.

    Usage:
        - @log_latency # Logs at logging.INFO
        - @log_latency() # Logs at logging.INFO
        - @log_latency=level=logging.DEBUG) # Logs at logging.DEBUG

    Returns:
        The output of calling func.
    """

    def decorator(func: Callable) -> Callable:
        logger = logging.getLogger('tgm')

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            latency = time.perf_counter() - start_time
            logger.log(level, 'Function %s executed in %.4fs', func.__name__, latency)
            return result

        return wrapper

    # If _func is None, decorator was called with parens
    if _func is None:
        return decorator
    else:
        # Decorator used without parens
        return decorator(_func)


def _get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


logger = _get_logger(__name__)


class _cached_property_log_cache_activity(functools.cached_property):
    def __set_name__(self, owner: type, name: str) -> None:
        self.attrname = name

    def __get__(self, instance: Any, owner: Any | None = None) -> Any:
        if instance is None:
            return self
        if self.attrname in instance.__dict__:
            logger.debug('%s Cache hit: %s', instance.__class__.__name__, self.attrname)
        else:
            logger.debug(
                '%s Cache miss: %s', instance.__class__.__name__, self.attrname
            )
        return super().__get__(instance, owner)
