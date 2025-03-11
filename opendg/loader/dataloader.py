from opendg.graph import DGraph
from opendg.loader.base import DGBaseLoader


class DGDataLoader(DGBaseLoader):
    r"""Load data from DGraph without any sampling."""

    def sample(self, batch: 'DGraph') -> 'DGraph':
        r"""DGDataLoader performs no subsampling. Returns the full batch.

        Args:
            batch (DGraph): Incoming batch of data. May not be materialized.

        Returns:
            (DGraph): The input batch of data.
        """
        return batch
