from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

import torch

from tgm import DGBatch, DGraph
from tgm.hooks import DeduplicationHook, DeviceTransferHook, DGHook


class HookManager:
    def __init__(self, device: str | torch.device = 'cpu') -> None:
        self._key_to_hooks: Dict[str, List[DGHook]] = {}  # Topological order
        self._shared_hooks: List[DGHook] = []
        self._active_key: str | None = None

        # Implicitly add deduplication shared hook
        self._shared_hooks.append(DeduplicationHook())

        # Device transfer hook is a special case. It is always run first,
        # and triggers even when no _active_key is present. This happens
        # if a user want to run data loader on gpu, without any addition hooks (e.g. negatives, neighbors)
        self._device_hook = DeviceTransferHook(device)

    def __str__(self) -> str:
        lines = ['HookManager:']
        lines.append(
            f'  DeviceHook: {self._device_hook.__class__.__name__} '
            f'(requires={self._device_hook.requires}, produces={self._device_hook.produces})'
        )

        lines.append('  Shared hooks:')
        for h in self._shared_hooks:
            lines.append(
                f'    - {h.__class__.__name__} '
                f'(requires={h.requires}, produces={h.produces})'
            )

        lines.append(f'  Active key: {self._active_key}')

        if self._key_to_hooks:
            lines.append('  Keyed hooks:')
            for key, hooks in self._key_to_hooks.items():
                lines.append(f'    {key}:')
                for h in hooks:
                    lines.append(
                        f'      - {h.__class__.__name__} '
                        f'(requires={h.requires}, produces={h.produces})'
                    )
        else:
            lines.append('  No keyed hooks registered.')

        return '\n'.join(lines)

    def register_shared(self, hook: DGHook) -> None:
        self._ensure_valid_hook(hook)
        if self._active_key is not None:
            raise RuntimeError(
                'Cannot register hooks while a key is active. Register hooks before using `activate`.'
            )
        self._shared_hooks.append(hook)

        # Recompute execution order for all keys
        for key, hooks in self._key_to_hooks.items():
            # Only include keyed hooks that are not shared
            keyed_only = [h for h in hooks if h not in self._shared_hooks]
            self._key_to_hooks[key] = self._topological_sort_hooks(
                self._shared_hooks + keyed_only
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

        # Precompute execution order including shared hooks once
        keyed_only = [h for h in self._key_to_hooks[key] if h not in self._shared_hooks]
        self._key_to_hooks[key] = self._topological_sort_hooks(
            self._shared_hooks + keyed_only
        )

    def get_active_hooks(self) -> List[DGHook]:
        if self._active_key is None:
            raise RuntimeError('No active key set. Use active() context manager.')
        return self._key_to_hooks[self._active_key]

    def set_active_hooks(self, key: str) -> None:
        self._active_key = key

    def execute_active_hooks(self, dg: DGraph) -> DGBatch:
        batch = dg.materialize()

        # Always run device hook first
        batch = self._device_hook(dg, batch)

        if self._active_key is None:
            return batch

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

    @staticmethod
    def _topological_sort_hooks(hooks: List[DGHook]) -> List[DGHook]:
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
