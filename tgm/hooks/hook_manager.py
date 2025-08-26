from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Set

from tgm import DGBatch, DGraph
from tgm.hooks import DeduplicationHook, DGHook


class HookManager:
    def __init__(self) -> None:
        self._key_to_hooks: Dict[str, List[DGHook]] = {}  # Topological order
        self._shared_hooks: List[DGHook] = []
        self._active_key: str | None = None

        # Implicitly add deduplication
        self._shared_hooks.append(DeduplicationHook())

    def register_shared(self, hook: DGHook) -> None:
        self._ensure_valid_hook(hook)
        if self._active_key is not None:
            raise RuntimeError(
                'Cannot register hooks while a key is active. Register hooks before using `activate`.'
            )
        self._shared_hooks.append(hook)

        # Recompute execution order for all keys
        for key, hooks in self._key_to_hooks.items():
            self._key_to_hooks[key] = self._topological_sort_hooks(
                self._shared_hooks + hooks
            )

    def register(self, key: str, hook: DGHook) -> None:
        self._ensure_valid_hook(hook)
        if self._active_key is not None:
            raise RuntimeError(
                'Cannot register hooks while a key is active. Register hooks before using `activate`.'
            )

        if key not in self._key_to_hooks:
            self._key_to_hooks[key] = []
        self._key_to_hooks[key].append(hook)

        # Precompute execution order for the key
        self._key_to_hooks[key] = self._topological_sort_hooks(
            self._shared_hooks + self._key_to_hooks[key]
        )

    def get_active_hooks(self) -> List[DGHook]:
        if self._active_key is None:
            raise RuntimeError('No active key set. Use active() context manager.')
        return self._key_to_hooks[self._active_key]

    def set_active_hooks(self, key: str) -> None:
        self._active_key = key

    def execute_active_hooks(self, dg: DGraph) -> DGBatch:
        batch = dg.materialize()

        for hook in self.get_active_hooks():
            batch = hook(dg, batch)
        return batch

    def reset_state(self, key: str | None = None) -> None:
        for hook in self._shared_hooks:
            hook.reset_state()

        keys_to_reset = [key] if key is not None else list(self._key_to_hooks.keys())
        for k in keys_to_reset:
            for h in self._key_to_hooks.get(k, []):
                h.reset_state()

    @contextmanager
    def activate(self, key: str) -> Iterator[None]:
        prev_key = self._active_key
        self.set_active_hooks(key)
        try:
            yield
        finally:
            self._active_key = prev_key  # Restore previous active key

    def _ensure_valid_hook(self, hook: Any) -> None:
        if not isinstance(hook, DGHook):
            raise TypeError(
                f'Cannot register hook {type(hook).__name__}: must implement __call__(dg: DGraph, batch: DGBatch) -> DGBatch and reset_state()'
            )

    def _topological_sort_hooks(self, hooks: List[DGHook]) -> List[DGHook]:
        produced: Set[str] = set()
        ordered: List[DGHook] = []
        remaining = hooks[:]  # Shalllow copy to avoid mutating

        while remaining:
            ready_hooks = [h for h in remaining if h.requires.issubset(produced)]

            if not ready_hooks:
                missing = {h: h.requires - produced for h in remaining}
                raise ValueError(f'Cannot resolve hook dependencies: {missing}')

            # Schedule ready hooks
            for hook in ready_hooks:
                ordered.append(hook)
                produced |= hook.produces
                remaining.remove(hook)

        return ordered
