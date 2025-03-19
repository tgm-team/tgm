from __future__ import annotations

from opendg.graph import DGraph
from opendg.loader.base import DGBaseLoader


class DGDataLoader(DGBaseLoader):
    r"""Load data from DGraph without any sampling."""

    def sample(self, batch: DGraph) -> DGraph:
        return batch
