from typing import Protocol

from opendg.graph import DGBatch


class DGHook(Protocol):
    def __call__(self, batch: DGBatch) -> DGBatch: ...
