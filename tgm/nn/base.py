from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class NNModule(Protocol):
    r"""Expected behaviors to be executed for module."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    r"""Runs the forward pass of the module."""


class BaseNN(ABC):
    r"""Base NN for all neural network models."""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...


class LinkDecoderBase(BaseNN):
    r"""Base NN for all link decoder."""

    @abstractmethod
    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor, **kwargs: Any) -> Any:
        raise NotImplementedError
