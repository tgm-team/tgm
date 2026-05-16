from typing import Any, Protocol, Set, runtime_checkable

from tgm import DGBatch


@runtime_checkable
class EncoderModule(Protocol):
    r"""Expected behaviors to be executed for encoder module."""

    requires: Set[str]

    def __call__(self, batch: DGBatch, *args: Any, **kwargs: Any) -> Any: ...

    r"""Runs the forward pass of the module."""
