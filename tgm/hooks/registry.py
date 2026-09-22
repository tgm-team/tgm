from typing import List, Type, TypeVar

from tgm.hooks.base import DGHook

T = TypeVar('T', bound=type)

_HOOK_REGISTRY: List[Type[DGHook]] = []


def hook(cls: T) -> T:
    """Decorator to register a hook class into the global registry.

    Example:
        @hook
        class BatchAnalyticsHook(StatelessHook):
            ...
    """
    _HOOK_REGISTRY.append(cls)  # ty: ignore[invalid-argument-type]
    return cls


def list_hooks() -> List[Type[DGHook]]:
    """List all registered hooks and their metadata."""
    return _HOOK_REGISTRY
