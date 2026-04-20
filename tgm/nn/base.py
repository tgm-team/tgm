from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, Set, runtime_checkable

from torch import nn


@runtime_checkable
class NNModule(Protocol):
    r"""Expected behaviors to be executed for all neural network models."""

    @property
    def requires(self) -> Set[str]: ...

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    r"""Runs the forward pass of the module."""


@dataclass(eq=False)
class BaseNN(ABC, nn.Module):
    r"""Base NN for all neural network models."""

    _cls_requires: ClassVar[Set[str]] = set()
    _requires: Set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        nn.Module.__init__(self)
        for cls in type(self).__mro__:
            cls_requires = cls.__dict__.get('_cls_requires', set())
            self._requires.update(cls_requires)

    @property
    def requires(self) -> Set[str]:
        return self._requires

    @abstractmethod
    def forward(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def validate_inputs(self, kwargs: dict) -> None:
        missing = self._requires - kwargs.keys()
        if missing:
            raise ValueError(
                f'[{self.__class__.__name__}] Missing required input keys: {missing}\n'
                f'Expected: {self._requires}'
            )


@dataclass(eq=False)
class EncoderBase(BaseNN):
    r"""Base NN for all encoders."""

    _cls_requires: ClassVar[Set[str]] = {'edge_index', 'X'}
