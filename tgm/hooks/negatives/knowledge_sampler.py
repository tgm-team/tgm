from __future__ import annotations

from typing import Any

from tgm.core import DGBatch
from tgm.hooks.negatives.tgb_base_sampler import TGBNegativeEdgeSamplerBase
from tgm.hooks.registry import hook
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


@hook
class TGBTKGNegativeEdgeSamplerHook(TGBNegativeEdgeSamplerBase):
    """Load data from DGraph using pre-generated TGB negative samples for knowledge graph.
    Make sure to perform `dataset.load_val_ns()` or `dataset.load_test_ns()` before using this hook.

    Args:
        dataset_name (str): The name of the TGB dataset to produce sampler for.
        split_mode (str): The split mode to use for sampling, either 'val' or 'test'.
        first_dst_id (int): identity of the first destination node
        last_dst_id (int): identity of the last destination node
        id (str): A unique identifier for the hook. The hook’s name and all attributes it produces will be suffixed with this `id`.


    Raises:
        ValueError: If neg_sampler is not provided.

    Key words: knowledge graph, link prediction, evaluation, negative sampler, tkgl.
    """

    _cls_requires = {'edge_src', 'edge_dst', 'edge_time', 'edge_type'}
    _cls_produces = {'neg', 'neg_batch_list', 'neg_time'}
    _dataset_prefix = 'tkgl'

    def __init__(
        self,
        dataset_name: str,
        split_mode: str,
        first_dst_id: int,
        last_dst_id: int,
        id: str | None = None,
    ) -> None:
        if first_dst_id < 0 or last_dst_id < 0:
            raise ValueError('First and last ID of node must be positive')
        self._first_dst_id = first_dst_id
        self._last_dst_id = last_dst_id
        super().__init__(dataset_name, split_mode, id)

    def _build_sampler(self, dataset_name: str) -> Any:
        try:
            from tgb.linkproppred.tkg_negative_sampler import (
                TKGNegativeEdgeSampler,
            )
        except ImportError:
            raise ImportError(
                f'TGB required for {self.__class__.__name__}, try `pip install py-tgb`'
            )
        return TKGNegativeEdgeSampler(
            dataset_name=dataset_name,
            first_dst_id=self._first_dst_id,
            last_dst_id=self._last_dst_id,
        )

    def _query_batch(self, batch: DGBatch) -> list:
        return self.neg_sampler.query_batch(
            batch.edge_src,
            batch.edge_dst,
            batch.edge_time,
            batch.edge_type,
            split_mode=self.split_mode,
        )
