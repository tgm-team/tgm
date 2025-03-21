from __future__ import annotations

from opendg.graph import DGraph
from opendg.loader.base import DGBaseLoader


class DGDataLoader(DGBaseLoader):
    r"""Load data from DGraph without any sampling."""

    def pre_yield(self, batch: DGraph) -> DGraph:
        return batch
