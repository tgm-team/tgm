from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

from tgm import DGBatch, DGraph
from tgm.hooks import DeduplicationHook, DGHook


class HookManager:
    def __init__(self, keys: List[str]) -> None:
        if not len(keys):
            raise ValueError('HookManager keys list must be non-empty')

        # Topological order
        self._key_to_hooks: Dict[str, List[DGHook]] = {k: [] for k in keys}
        self._shared_hooks: List[DGHook] = []
        self._active_key: str | None = None

        # Implicitly add deduplication shared hook
        self._shared_hooks.append(DeduplicationHook())

    def __str__(self) -> str:
        def _stringify_hook(h: DGHook) -> str:
            return f'    - {h.__class__.__name__} (requires={h.requires}, produces={h.produces})'

        lines = ['HookManager:']
        lines.append('  Shared hooks:')
        for h in self._shared_hooks:
            lines.append(_stringify_hook(h))

        lines.append(f'  Active key: {self._active_key}')
        if self._key_to_hooks:
            lines.append('  Keyed hooks:')
            for key, hooks in self._key_to_hooks.items():
                lines.append(f'    {key}:')
                for h in hooks:
                    lines.append(_stringify_hook(h))
        else:
            lines.append('  No keyed hooks registered.')

        return '\n'.join(lines)

    def register_shared(self, hook: DGHook) -> None:
        self._ensure_valid_hook(hook)
        self._ensure_no_active_key()

        self._shared_hooks.append(hook)

        # Recompute execution order for all keys
        for key, hooks in self._key_to_hooks.items():
            # Only include keyed hooks that are not shared
            keyed_only = [h for h in hooks if h not in self._shared_hooks]
            self._key_to_hooks[key] = self._topological_sort_hooks(
                self._shared_hooks + keyed_only
            )

    def register(self, key: str, hook: DGHook) -> None:
        self._ensure_valid_key(key)
        self._ensure_valid_hook(hook)
        self._ensure_no_active_key()

        self._key_to_hooks[key].append(hook)

        # Precompute execution order including shared hooks once
        keyed_only = [h for h in self._key_to_hooks[key] if h not in self._shared_hooks]
        self._key_to_hooks[key] = self._topological_sort_hooks(
            self._shared_hooks + keyed_only
        )

    def get_active_hooks(self) -> List[DGHook]:
        if self._active_key is None:
            raise RuntimeError('No active key set. Use activate() context manager.')
        return self._key_to_hooks[self._active_key]

    def set_active_hooks(self, key: str) -> None:
        self._ensure_valid_key(key)
        self._active_key = key

    def execute_active_hooks(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        for hook in self.get_active_hooks():
            batch = hook(dg, batch)
        return batch

    def reset_state(self, key: str | None = None) -> None:
        if key is not None:
            self._ensure_valid_key(key)

        for hook in self._shared_hooks:
            hook.reset_state()

        keys_to_reset = [key] if key is not None else list(self._key_to_hooks.keys())
        for k in keys_to_reset:
            for h in self._key_to_hooks[k]:
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

    def _ensure_no_active_key(self) -> None:
        if self._active_key is not None:
            raise RuntimeError(
                'Cannot register hooks while a key is active. Register hooks before using `activate`.'
            )

    def _ensure_valid_key(self, key: str) -> None:
        if key not in self._key_to_hooks:
            raise KeyError(f'{key} was not a declared key in the hook manager')

    @staticmethod
    def _topological_sort_hooks(hooks: List[DGHook]) -> List[DGHook]:
        # Before producing a valid hook ordering, we need to ensure
        # that all the required attributes are produced by *some* hook.
        all_produced = set().union(*(h.produces for h in hooks))
        missing = set()
        for h in hooks:
            missing |= h.requires - all_produced
        if missing:
            raise ValueError(
                f'Cannot resolve hook dependencies: required attributes not produced by any hook: {missing}'
            )

        # Build adjacency list and then run Kahn's algorithmk jbs
        adj_list: Dict[DGHook, List[DGHook]] = defaultdict(list)
        for i, h1 in enumerate(hooks):
            for j, h2 in enumerate(hooks):
                if i != j and h1.produces & h2.requires:
                    # If h2 requires something h1 produces, add edge
                    adj_list[h1].append(h2)

        indegree: Dict[DGHook, int] = {h: 0 for h in hooks}
        for u in adj_list:
            for v in adj_list[u]:
                indegree[v] += 1

        queue = deque([h for h in hooks if indegree[h] == 0])
        ordered: List['DGHook'] = []

        while queue:
            u = queue.popleft()
            ordered.append(u)
            for v in adj_list.get(u, []):
                indegree[v] -= 1
                if indegree[v] == 0:
                    queue.append(v)

        if len(ordered) != len(hooks):
            unresolved = [h for h in hooks if h not in ordered]

            # For each unresolved hook, show which .requires are unsatisfied
            err_msg = f'Cannot resolve hook dependencies:\n'
            for h in unresolved:
                missing = h.requires - set().union(*[u.produces for u in ordered])
                err_msg += f'\n - {repr(h)} requires {missing} but not produced (or stuck in cycle)'
            raise ValueError(err_msg)

        return ordered
