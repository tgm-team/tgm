from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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


@dataclass(eq=False)
class BaseDGHook(ABC):
    """Base class for hooks."""

    requires: Set[str] = field(default_factory=set)
    produces: Set[str] = field(default_factory=set)
    id: str | None = None
    has_state: bool = False

    def __post_init__(self) -> None:
        if self.id:
            self.produces = {f'{p}_{self.id}' for p in self.produces}

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        if self.id:
            cls_name = f'{cls_name}_{self.id}'
        return cls_name

    @abstractmethod
    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        raise NotImplementedError

    def reset_state(self) -> None:
        pass


class StatelessHook(BaseDGHook):
    """Base class for hooks without internal state."""

    has_state: bool = False


class StatefulHook(BaseDGHook):
    """Base class for hooks that maintain internal state."""

    has_state: bool = True
