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

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...


class StatelessHook:
    """Base class for hooks without internal state."""

    requires: Set[str] = set()
    produces: Set[str] = set()
    has_state: bool = False

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        raise NotImplementedError

    def reset_state(self) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


class StatefulHook:
    """Base class for hooks that maintain internal state."""

    requires: Set[str] = set()
    produces: Set[str] = set()
    has_state: bool = True

    def state_dict(self) -> dict:
        """Return the hook's state as a serializable dict."""
        raise NotImplementedError(
            f'{self.__class__.__name__} has has_state=True '
            f'but did not implement state_dict(). '
            f'implement state_dict() to support checkpointing.'
        )

    def load_state_dict(self, state: dict) -> None:
        """Restore the hook's state from a dict returned by state_dict()."""
        raise NotImplementedError(
            f'{self.__class__.__name__} has has_state=True '
            f'but did not implement load_state_dict(). '
            f'implement load_state_dict() to support checkpointing.'
        )
