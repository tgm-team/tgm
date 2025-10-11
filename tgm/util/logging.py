import functools
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch

_TGM_LOGGING_ENABLED: bool = os.getenv('TGM_LOGGING_ENABLED', '0').lower() in (
    '1',
    'true',
)


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
    global _TGM_LOGGING_ENABLED
    _TGM_LOGGING_ENABLED = True

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

    Logs human-readable info at `level`, and JSON-formatted debug log at DEBUG.

    Usage:
        - @log_latency                      # Logs at logging.INFO
        - @log_latency()                    # Logs at logging.INFO
        - @log_latency=level=logging.DEBUG) # Logs at logging.DEBUG (JSON included)

    Returns:
        The output of calling func.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _TGM_LOGGING_ENABLED:
                return func(*args, **kwargs)

            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            latency = time.perf_counter() - start_time
            util_logger.log(
                level, 'Function %s executed in %.4fs', func.__name__, latency
            )

            if util_logger.isEnabledFor(logging.DEBUG):
                log_entry = {
                    'metric': f'{func.__name__} latency',
                    'value': latency,
                }
                util_logger.debug(json.dumps(log_entry))
            return result

        return wrapper

    # If _func is None, decorator was called with parens
    if _func is None:
        return decorator
    else:
        # Decorator used without parens
        return decorator(_func)


def log_gpu(_func: Callable | None = None, *, level: int = logging.INFO) -> Any:
    """Function decorator to log GPU memory usage during a function call.

    Logs human-readable info at `level`, and JSON-formatted debug log at DEBUG.

    Usage:
        - @log_gpu                       # Logs at logging.INFO
        - @log_gpu()                     # Logs at logging.INFO
        - @log_gpu(level=logging.DEBUG)  # Logs at DEBUG (JSON included)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _TGM_LOGGING_ENABLED:
                return func(*args, **kwargs)

            cuda_available = torch.cuda.is_available()
            if cuda_available:
                torch.cuda.reset_peak_memory_stats()
                start_mem = torch.cuda.memory_allocated() / (1024**2)
            else:
                start_mem = 0.0

            result = func(*args, **kwargs)

            if cuda_available:
                peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
                mem_diff = peak_mem - start_mem
            else:
                peak_mem = mem_diff = 0.0

            util_logger.log(
                level,
                'Function %s GPU memory (CUDA available=%s) [MB]: peak=%.2f, alloc=%.2f',
                func.__name__,
                cuda_available,
                peak_mem,
                mem_diff,
            )

            if util_logger.isEnabledFor(logging.DEBUG):
                log_entry = {
                    'metric': f'{func.__name__} peak_gpu_mb',
                    'value': peak_mem,
                }
                util_logger.debug(json.dumps(log_entry))

                log_entry = {
                    'metric': f'{func.__name__} alloc_gpu_mb',
                    'value': mem_diff,
                }
                util_logger.debug(json.dumps(log_entry))
            return result

        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)


def log_metrics_dict(
    metrics_dict: Dict[str, Any],
    *,
    epoch: int | None = None,
    level: int = logging.INFO,
    extra: Dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Log a set of metric with optional epoch and structured JSON output.

    Logs human-readable info at `level`, and JSON-formatted debug log at DEBUG.

    Note: This is equivalent to calling log_metric for each key-value pair.

    Args:
        metrics_dict (Dict[str, Any]): Dictionary of metric_name: metric_value pairs.
        epoch (Optional[int]): Optional epoch number.
        level (int): Logging level for human-readable log (default INFO)
        extra (Dict[str, Any]): Optional dictionary of extra metadata to include in JSON.
        logger (Optional[logging.Logger]): Logger to log to, defaults to tgm.util logger.
    """
    for metric_name, metric_value in metrics_dict.items():
        log_metric(
            metric_name,
            metric_value,
            epoch=epoch,
            level=level,
            extra=extra,
            logger=logger,
        )


def log_metric(
    metric_name: str,
    metric_value: Any,
    *,
    epoch: int | None = None,
    level: int = logging.INFO,
    extra: Dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Log a metric with optional epoch and structured JSON output.

    Logs human-readable info at `level`, and JSON-formatted debug log at DEBUG.

    Args:
        metric_name (str): Name of the metric to log.
        metric_value (Any): Value of the metric to log.
        epoch (Optional[int]): Optional epoch number.
        level (int): Logging level for human-readable log (default INFO)
        extra (Dict[str, Any]): Optional dictionary of extra metadata to include in JSON.
        logger (Optional[logging.Logger]): Logger to log to, defaults to tgm.util logger.
    """
    if not _TGM_LOGGING_ENABLED:
        return

    logger = logger or util_logger

    display_value = (
        round(metric_value, 4) if isinstance(metric_value, float) else metric_value
    )
    parts = []
    if epoch is not None:
        parts.append(f'Epoch={epoch:02d}')
    parts.append(f'{metric_name}={display_value}')
    msg = ' '.join(parts)
    logger.log(level, msg)

    if logger.isEnabledFor(logging.DEBUG):
        if epoch is not None:
            metric_name += f' epoch {epoch}'
        log_entry = {'metric': metric_name, 'value': metric_value}
        if extra is not None:
            log_entry.update(extra)
        logger.debug(json.dumps(log_entry))


def _get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


util_logger = _get_logger(__name__)


def pretty_number_format(x: int | float) -> str:
    try:
        # Handle inf and nan
        if isinstance(x, float) and (x == float('inf') or x == float('-inf') or x != x):
            return str(x)

        n = float(x)

        # Small numbers: comma separated
        if abs(n) < 1_000_000:
            if n.is_integer():
                return f'{int(n):,}'
            else:
                return f'{n:,.2f}'

        # Large numbers: suffix
        suffixes = ['', 'K', 'M', 'B', 'T']
        magnitude = 0
        while abs(n) >= 1000 and magnitude < len(suffixes) - 1:
            magnitude += 1
            n /= 1000.0
        return f'{n:.2f}{suffixes[magnitude]}'

    except Exception:
        return str(x)


class _logged_cached_property(functools.cached_property):
    def __set_name__(self, owner: type, name: str) -> None:
        self.attrname = name

    def __get__(self, instance: Any, owner: Any | None = None) -> Any:
        if instance is None:
            return self
        if self.attrname in instance.__dict__:
            util_logger.debug(
                '%s Cache hit: %s', instance.__class__.__name__, self.attrname
            )
        else:
            util_logger.debug(
                '%s Cache miss: %s', instance.__class__.__name__, self.attrname
            )
        return super().__get__(instance, owner)
