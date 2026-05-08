from abc import ABC, abstractmethod
from typing import Any, Protocol, Set, runtime_checkable

from torch import nn

from tgm import DGBatch


@runtime_checkable
class NNModule(Protocol):
    r"""Expected behaviors to be executed for module."""

    @property
    def requires(self) -> Set[str]: ...

    def __call__(self, batch: DGBatch, *args: Any, **kwargs: Any) -> Any: ...

    r"""Runs the forward pass of the module."""


class BaseNN(ABC, nn.Module):
    r"""Base NN for all neural network models."""

    def __init__(self) -> None:
        super().__init__()
        self._requires: Set[str] = set()

    @property
    def requires(self) -> Set[str]:
        return self._requires

    @abstractmethod
    def forward(self, batch: DGBatch, *args: Any, **kwargs: Any) -> Any: ...
