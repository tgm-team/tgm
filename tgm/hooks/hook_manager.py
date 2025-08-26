from __future__ import annotations

from typing import List, Set

from tgm import DGBatch, DGraph
from tgm.hooks import (
    DeduplicationHook,
    DeviceTransferHook,
    DGHook,
    PinMemoryHook,
)


class HookManager:
    def __init__(self, dg: DGraph, hooks: List[DGHook]) -> None:
        if not isinstance(hooks, list):
            raise TypeError(f'Invalid hook type: {type(hooks)}')
        bad_hook_names = [type(h).__name__ for h in hooks if not isinstance(h, DGHook)]
        if len(bad_hook_names):
            raise TypeError(
                f'These hooks do not correctly implement the DGHook protocol: {bad_hook_names}, '
                'ensure there is a __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch implemented'
            )

        # Implicitly add dedup hook after all user-defined hooks and before device transfer
        hooks.append(DeduplicationHook())

        if dg.device.type != 'cpu':
            hooks.append(PinMemoryHook())
            hooks.append(DeviceTransferHook(dg.device))

        self.hooks = hooks
        self._validate_hook_dependencies()

    def reset_state(self) -> None:
        for hook in self.hooks:
            hook.reset_state()

    @classmethod
    def from_any(
        cls, dg: DGraph, hook_like: HookManager | DGHook | List[DGHook] | None
    ) -> HookManager:
        if isinstance(hook_like, cls):
            return hook_like
        elif hook_like is None:
            return cls(dg, hooks=[])
        elif isinstance(hook_like, DGHook):
            return cls(dg, hooks=[hook_like])
        elif isinstance(hook_like, list):
            return cls(dg, hooks=hook_like)
        else:
            raise TypeError(f'Invalid hook type: {type(hook_like)}')

    def __call__(self, dg: DGraph) -> DGBatch:
        batch = dg.materialize()
        for hook in self.hooks:
            batch = hook(dg, batch)
        return batch

    def _validate_hook_dependencies(self) -> None:
        produced: Set[str] = set()
        for hook in self.hooks:
            missing = hook.requires - produced
            if missing:
                raise ValueError(
                    f'{hook.__class__.__name__} is missing required fields: {missing}'
                )
            produced |= hook.produces
