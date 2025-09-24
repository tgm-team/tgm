from typing import Any, Callable


def suppress_output(func: Callable, *args: Any, **kwargs: Any) -> Any:
    import builtins

    SILENCE_PREFIXES = [
        'raw file found',
        'Dataset directory is',
        'loading processed file',
    ]

    original_print = builtins.print

    def filtered_print(*p_args: Any, **p_kwargs: Any) -> None:
        if not p_args:
            return
        msg = str(p_args[0])
        if any(msg.startswith(prefix) for prefix in SILENCE_PREFIXES):
            return
        original_print(*p_args, **p_kwargs)

    try:
        builtins.print = filtered_print
        return func(*args, **kwargs)
    finally:
        builtins.print = original_print
