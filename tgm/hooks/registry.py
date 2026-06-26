from typing import List, Type

from tgm.hooks import DGHook

_HOOK_REGISTRY: List[Type[DGHook]] = []


def hook(cls: Type[DGHook]) -> Type[DGHook]:
    """Decorator to register a hook class into the global registry.

    Example:
        @hook
        class BatchAnalyticsHook(StatelessHook):
            ...
    """
    _HOOK_REGISTRY.append(cls)
    return cls


def list_hooks() -> List[Type[DGHook]]:
    """List all registered hooks and their metadata."""
    return _HOOK_REGISTRY
