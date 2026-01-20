from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

from tgm import DGBatch, DGraph
from tgm.exceptions import (
    BadHookProtocolError,
    UnresolvableHookDependenciesError,
)
from tgm.hooks import DGHook
from tgm.util.logging import _get_logger, log_latency

logger = _get_logger(__name__)


class HookManager:
    """Manages hooks (`DGHook`s) for a `DGraph`, supporting both shared and key-specific hooks.

    This class allows you to register hooks that modify or enrich batches of graph events
    via transformation on the current, and optionally past history of the temporal graph.
    The hook manager executed these transformation transparently to the user during data
    iteration. Hooks can be shared across all keys or specific to a key.
    Dependencies between hooks are automatically resolved using topological
    sorting based on their `requires` and `produces` attributes.

    Args:
        keys (List[str]): List of valid keys for key-specific hooks. Each key can
            have its own set of hooks, in addition to shared hooks.

    Raises:
        ValueError: If `keys` is empty.
    """

    def __init__(self, keys: List[str]) -> None:
        if not len(keys):
            raise ValueError('HookManager keys list must be non-empty')

        self._dirty: Dict[str, bool] = {k: False for k in keys}
        self._key_to_hooks: Dict[str, List[DGHook]] = {k: [] for k in keys}
        self._shared_hooks: List[DGHook] = []
        self._active_key: str | None = None
        self._registered_key = keys

    @property
    def keys(self) -> List:
        return self._registered_key

    def __str__(self) -> str:
        def _stringify_hook(h: DGHook) -> str:
            return f'    - {h.__class__.__name__} (requires={h.requires}, produces={h.produces})'

        lines = ['HookManager:']
        lines.append('  Shared hooks:')
        for h in self._shared_hooks:
            lines.append(_stringify_hook(h))

        lines.append(f'  Active key: {self._active_key}')
        lines.append('  Keyed hooks:')
        for key, hooks in self._key_to_hooks.items():
            lines.append(f'    {key}:')
            for h in hooks:
                lines.append(_stringify_hook(h))

        return '\n'.join(lines)

    def register_shared(self, hook: DGHook) -> None:
        """Registers a shared hook that runs for all keys.

        Args:
            hook (DGHook): The hook to register.

        Raises:
            BadHookProtocolError: If `hook` does not implement the `DGHook` protocol.
            RuntimeError: If called while a key is active.
        """
        self._ensure_valid_hook(hook)
        self._ensure_no_active_key()
        self._shared_hooks.append(hook)
        for k in self._dirty:  # Mark all keys as 'dirty'
            self._dirty[k] = True
        logger.debug('Registered shared hook: %s', hook.__class__.__name__)

    def register(self, key: str, hook: DGHook) -> None:
        """Registers a key-specific hook.

        Args:
            key (str): The key to associate the hook with.
            hook (DGHook): The hook to register.

        Raises:
            KeyError: If `key` is not a declared key.
            BadHookProtocolError: If `hook` does not implement the `DGHook` protocol.
            RuntimeError: If called while a key is active.
        """
        self._ensure_valid_key(key)
        self._ensure_valid_hook(hook)
        self._ensure_no_active_key()
        self._key_to_hooks[key].append(hook)
        self._dirty[key] = True  # Mark registered key as 'dirty'
        logger.debug('Registered hook: %s for key: %s', hook.__class__.__name__, key)

    def set_active_hooks(self, key: str) -> None:
        """Sets the currently active key for executing hooks.

        Args:
            key (str): The key to activate.

        Raises:
            KeyError: If `key` is not a declared key.
        """
        self._ensure_valid_key(key)
        self._active_key = key
        logger.debug('Set active hooks to key: %s', key)

    @log_latency(level=logging.DEBUG)
    def execute_active_hooks(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        """Executes all hooks (shared + key-specific) for the active key on a batch.

        Args:
            dg (DGraph): The graph on which hooks operate.
            batch (DGBatch): The batch of events to modify.

        Returns:
            DGBatch: The modified batch after all hooks are applied.

        Raises:
            RuntimeError: If no active key is set.
            UnresolvableHookDependenciesError: If required attributes are missing or hooks form a cyclic dependency.
        """
        if self._active_key is None:
            raise RuntimeError('No active key set. Use activate() context manager.')

        # Lazily validate and topological sort if needed
        key = self._active_key
        if self._dirty[key]:
            self.resolve_hooks(key)

        for hook in self._key_to_hooks[key]:
            start_time = time.perf_counter()
            batch = hook(dg, batch)
            latency = time.perf_counter() - start_time
            logger.debug('%s hook executed in %.4fs', hook.__class__.__name__, latency)

        return batch

    def reset_state(self, key: str | None = None) -> None:
        """Resets the internal state of all stateful hooks.

        Args:
            key (str | None): If specified, resets only hooks for this key.
                Otherwise resets all keys and shared hooks.
        """
        logger.debug('Resetting hooks for key: %s', key)
        if key is not None:
            self._ensure_valid_key(key)

        for hook in self._shared_hooks:
            logger.debug('Resetting state for shared hook: %s', hook.__class__.__name__)
            hook.reset_state()

        keys_to_reset = [key] if key is not None else list(self._key_to_hooks.keys())
        for k in keys_to_reset:
            for h in self._key_to_hooks[k]:
                logger.debug('Resetting state for keyed hook: %s', h.__class__.__name__)
                h.reset_state()

    def resolve_hooks(self, key: str | None = None) -> None:
        """Resolves hook execution order by topologically sorting them based on dependencies.

        Args:
            key (str | None): If specified, resolves hooks only for this key.
                Otherwise resolves all keys.

        Raises:
            KeyError: If `key` is invalid.
            UnresolvableHookDependenciesError: If required attributes are missing or hooks form a cyclic dependency.
        """
        if key is not None:
            self._ensure_valid_key(key)

        keys_to_validate = [key] if key else list(self._key_to_hooks.keys())
        for k in keys_to_validate:
            hooks = self._shared_hooks + [
                h for h in self._key_to_hooks[k] if h not in self._shared_hooks
            ]

            # @TODO: Need a better solution when requirement at instance-level is introduced to DGHook
            filter_hooks = [h for h in hooks if not self._is_dedup(h)]
            dedups = list(set(hooks) - set(filter_hooks))
            hooks = filter_hooks

            logger.debug('Running topological sort to resolve hooks for key: %s', k)
            self._key_to_hooks[k] = self._topological_sort_hooks(hooks)
            if len(dedups) > 0:
                logger.debug(
                    'Identify DeduplicationHook. Enforce DeduplicationHook to run at the end.'
                )
                self._key_to_hooks[k].extend(dedups)
            self._dirty[k] = False

    @contextmanager
    def activate(self, key: str) -> Iterator[None]:
        """Context manager to temporarily set a key as active for hook execution.

        Args:
            key (str): The key to activate.
        """
        prev_key = self._active_key
        self.set_active_hooks(key)
        try:
            yield
        finally:
            self._active_key = prev_key  # Restore previous active key

    def _ensure_valid_hook(self, hook: Any) -> None:
        if not isinstance(hook, DGHook):
            raise BadHookProtocolError(
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
        logger.debug('Hook ordering BEFORE topological sort:')
        for h in hooks:
            logger.debug(f'  - {h.__class__.__name__}')

        # Before producing a valid hook ordering, we need to ensure
        # that all the required attributes are produced by *some* hook.
        # Note, we assume that the edge index and node event attributes are *always*
        # present in the materialized batch (no hook explicitly produces them, but they
        # are marked as *required*). This could still lead to runtime issues if a batch
        # does not have, e.g. node events but they are marked as required by a registered hook.
        # We cannot guard against this here since determining whether or not a batch has
        # edge/node events can only be inferred during data loading.
        all_produced = set(
            [
                'edge_src',
                'edge_dst',
                'edge_time',
                'edge_type',
                'node_x_time',
                'node_x_nids',
                'node_y_time',
                'node_y_nids',
                'node_type',
            ]
        ).union(*(h.produces for h in hooks))
        missing = set()
        for h in hooks:
            missing |= h.requires - all_produced
        if missing:
            raise UnresolvableHookDependenciesError(
                f'Cannot resolve hook dependencies: required attributes not produced by any hook: {missing}'
            )

        # Build adjacency list and then run Kahn's algorithm
        adj_list: Dict[DGHook, List[DGHook]] = defaultdict(list)
        for i, h1 in enumerate(hooks):
            for j, h2 in enumerate(hooks):
                if i != j and h1.produces & h2.requires:
                    # If h2 requires something h1 produces, add edge
                    adj_list[h1].append(h2)

                # TODO: This is a hacky short term fix for implicit hook ordering constraints.
                # If both a negative hook and a neighbor hook are present, it is crucial
                # that the negatives come first (so that we sample neighbors for the negatives).
                # But since neighbor sampler does not explicitly require negatives, the topological
                # sort may put these out of order. In order to fix this, we add an extra edge
                # into the DAG before sorting. Long term, we need to think about how to avoid
                # things like this, and make it seamless for the user.
                is_neg_hook = lambda h: 'neg' in h.produces
                is_nbr_hook = lambda h: 'nbr_nids' in h.produces
                if is_neg_hook(h1) and is_nbr_hook(h2):
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
            raise UnresolvableHookDependenciesError(err_msg)

        logger.debug('Hook ordering AFTER topological sort:')
        for h in hooks:
            logger.debug(f'  - {h.__class__.__name__}')

        return ordered

    def _is_dedup(self, hook: DGHook) -> bool:
        return 'Deduplication' in hook.__class__.__name__
