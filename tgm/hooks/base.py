from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Protocol, Set, runtime_checkable

from tgm import DGBatch, DGraph


@runtime_checkable
class DGHook(Protocol):
    r"""The behaviours to be executed on a DGraph before materializing."""

    has_state: bool

    @property
    def requires(self) -> Set[str]: ...

    @property
    def produces(self) -> Set[str]: ...

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch: ...

    def reset_state(self) -> None: ...


@dataclass(eq=False)
class BaseDGHook(ABC):
    """Base class for hooks."""

    _cls_requires: Set[str] = field(default_factory=set, init=False, repr=False)
    _cls_produces: Set[str] = field(default_factory=set, init=False, repr=False)

    _requires: Set[str] = field(default_factory=set)
    _produces: Set[str] = field(default_factory=set)

    _id: str | None = None

    has_state: bool = False

    def __post_init__(self) -> None:
        self._requires.update(type(self).__dict__.get('_cls_requires', set()))
        self._produces.update(type(self).__dict__.get('_cls_produces', set()))

    @property
    def produces(self) -> Set[str]:
        if self._id is None:
            return self._produces
        else:
            return {f'{p}_{self._id}' for p in self._produces}

    @property
    def requires(self) -> Set[str]:
        return self._requires

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        if self._id:
            cls_name = f'{cls_name}_{self._id}'
        return cls_name

    @abstractmethod
    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        raise NotImplementedError

    def reset_state(self) -> None:
        pass

    def add_attribute_to_batch(self, batch: DGBatch, name: str, value: Any) -> None:
        """Add a new attribute to providede batch.

        If `_id` is specified, the new attribute name will be appended with the given `id` as a suffix.
        """
        if self._id:
            name = f'{name}_{self._id}'
        setattr(batch, name, value)


class StatelessHook(BaseDGHook):
    """Base class for hooks without internal state."""

    has_state: bool = False


class StatefulHook(BaseDGHook):
    """Base class for hooks that maintain internal state."""

    has_state: bool = True


@dataclass(eq=False)
class SeedableHook(BaseDGHook):
    seed_keys: List[str] | None = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._requires.update(self.seed_keys if self.seed_keys else [])
