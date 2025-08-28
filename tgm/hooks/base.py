from __future__ import annotations

from typing import Protocol, Set, runtime_checkable

from tgm import DGBatch, DGraph


@runtime_checkable
class DGHook(Protocol):
    r"""The behaviours to be executed on a DGraph before materializing."""

    requires: Set[str]
    produces: Set[str]
    has_state: bool

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch: ...

    def reset_state(self) -> None: ...


class StatelessHook:
    """Base class for hooks without internal state."""

    requires: Set[str] = set()
    produces: Set[str] = set()
    has_state: bool = False

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        raise NotImplementedError

    def reset_state(self) -> None:
        pass


class StatefulHook:
    """Base class for hooks that maintain internal state."""

    requires: Set[str] = set()
    produces: Set[str] = set()
    has_state: bool = True
