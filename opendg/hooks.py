from typing import Protocol

from opendg.graph import DGraph


class DGHook(Protocol):
    r"""The behaviours to be executed on a DGraph before materializing."""

    def __call__(self, batch: DGraph) -> DGraph: ...
