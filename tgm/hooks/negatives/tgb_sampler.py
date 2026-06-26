from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tgm.core import DGBatch, DGraph
from tgm.hooks.base import StatelessHook
from tgm.hooks.registry import hook
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


@hook
class TGBNegativeEdgeSamplerBase(StatelessHook):
    """Base class for TGB pre-generated negative edge sampler hooks.

    Handles common logic for loading evaluation sets, querying negative samples,
    and assembling batch attributes. Subclasses must implement ``_build_sampler``
    and ``_query_batch``.

    Args:
        dataset_name (str): The name of the TGB dataset.
        split_mode (str): The split mode to use for sampling, either 'val' or 'test'.
        id (str | None): A unique identifier for the hook.

    Attributes produced:
        neg (Tensor[int32]): Unique negative destination node ids across the batch.
        neg_batch_list (list[Tensor[int32]]): Per-edge negative candidate lists,
            aligned with ``batch.edge_src``.
        neg_time (Tensor[int64]): Randomly sampled timestamps for each negative,
            drawn uniformly from ``[batch.edge_time.min(), batch.edge_time.max()]``
            with a fixed seed for reproducibility.

    """

    _dataset_prefix: str

    def __init__(
        self, dataset_name: str, split_mode: str, id: str | None = None
    ) -> None:
        super().__init__()
        if split_mode not in ['val', 'test']:
            raise ValueError(f'split_mode must be "val" or "test", got: {split_mode}')

        try:
            from tgb.utils.info import DATA_VERSION_DICT, PROJ_DIR
        except ImportError:
            raise ImportError(
                f'TGB required for {self.__class__.__name__}, try `pip install py-tgb`'
            )

        if not dataset_name.startswith(f'{self._dataset_prefix}-'):
            raise ValueError(
                'TGBNegativeEdgeSamplerHook should only be registered for '
                f'"{self._dataset_prefix}-xxx" datasets, but got: {dataset_name}'
            )

        neg_sampler = self._build_sampler(dataset_name)

        # Load evaluation sets
        root = Path(PROJ_DIR + 'datasets') / dataset_name.replace('-', '_')
        if DATA_VERSION_DICT.get(dataset_name, 1) > 1:
            version_suffix = f'_v{DATA_VERSION_DICT[dataset_name]}'
        else:
            version_suffix = ''

        ns_fname = root / f'{dataset_name}_{split_mode}_ns{version_suffix}.pkl'
        logger.debug(
            'Loading %s split (neg_sampler.load_eval_set) for dataset: %s from file: %s',
            split_mode,
            dataset_name,
            ns_fname,
        )
        neg_sampler.load_eval_set(fname=str(ns_fname), split_mode=split_mode)

        self.neg_sampler = neg_sampler
        self.split_mode = split_mode
        self._id = id
        self.__post_init__()

    def _build_sampler(self, dataset_name: str) -> Any:
        """Instantiate and return the TGB negative sampler. Must be implemented by subclasses."""
        raise NotImplementedError

    def _query_batch(self, batch: DGBatch) -> list:
        """Query the sampler for a batch. Must be implemented by subclasses."""
        raise NotImplementedError

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        if batch.edge_src.size(0) == 0:
            batch_neg = torch.empty(
                batch.edge_src.size(0), dtype=torch.int32, device=dg.device
            )
            batch_neg_time = torch.empty(
                batch.edge_src.size(0), dtype=torch.int64, device=dg.device
            )
            batch_neg_batch_list = []
        else:
            try:
                neg_batch_list = self._query_batch(batch)
            except ValueError as e:
                raise ValueError(
                    f'{self._dataset_prefix.upper()} Negative sampling failed for split_mode={self.split_mode}. Try updating your TGB package: `pip install --upgrade py-tgb`'
                ) from e

            batch_neg_batch_list = [
                torch.tensor(neg_batch, dtype=torch.int32, device=dg.device)
                for neg_batch in neg_batch_list
            ]
            batch_neg = torch.unique(torch.cat(batch_neg_batch_list))

            # This is a heuristic. For our fake (negative) link times,
            # we pick random time stamps within [batch.start_time, batch.end_time].
            # Using random times on the whole graph will likely produce information
            # leakage, making the prediction easier than it should be.

            # Use generator to local constrain rng for reproducibility
            gen = torch.Generator(device=dg.device)
            gen.manual_seed(0)
            batch_neg_time = torch.randint(
                int(batch.edge_time.min().item()),
                int(batch.edge_time.max().item()) + 1,
                (batch_neg.size(0),),
                device=dg.device,
                generator=gen,
            )

        self.add_batch_attribute(batch, 'neg', batch_neg)
        self.add_batch_attribute(batch, 'neg_batch_list', batch_neg_batch_list)
        self.add_batch_attribute(batch, 'neg_time', batch_neg_time)
        return batch


@hook
class TGBNegativeEdgeSamplerHook(TGBNegativeEdgeSamplerBase):
    """Load data from DGraph using pre-generated TGB negative samples.
    Make sure to perform `dataset.load_val_ns()` or `dataset.load_test_ns()` before using this hook.

    Args:
        dataset_name (str): The name of the TGB dataset to produce sampler for.
        split_mode (str): The split mode to use for sampling, either 'val' or 'test'.
        id (str): A unique identifier for the hook. The hook’s name and all attributes it produces will be suffixed with this `id`.

    Raises:
        ValueError: If neg_sampler is not provided.

    Key words: negative sampler, evaluation, tgbl, link prediction.
    """

    _cls_requires = {'edge_src', 'edge_dst', 'edge_time'}
    _cls_produces = {'neg', 'neg_batch_list', 'neg_time'}
    _dataset_prefix = 'tgbl'

    def _build_sampler(self, dataset_name: str) -> Any:
        try:
            from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
        except ImportError:
            raise ImportError(
                f'TGB required for {self.__class__.__name__}, try `pip install py-tgb`'
            )
        return NegativeEdgeSampler(dataset_name=dataset_name)

    def _query_batch(self, batch: DGBatch) -> list:
        return self.neg_sampler.query_batch(
            batch.edge_src,
            batch.edge_dst,
            batch.edge_time,
            split_mode=self.split_mode,
        )


@hook
class TGBTHGNegativeEdgeSamplerHook(TGBNegativeEdgeSamplerBase):
    """Load data from DGraph using pre-generated TGB negative samples for heterogeneous graph.
    Make sure to perform `dataset.load_val_ns()` or `dataset.load_test_ns()` before using this hook.

    Args:
        dataset_name (str): The name of the TGB dataset to produce sampler for.
        split_mode (str): The split mode to use for sampling, either 'val' or 'test'.
        first_node_id (int): identity of the first node
        last_node_id (int): identity of the last destination node
        node_type (Tensor): the node type of each node
        id (str): A unique identifier for the hook. The hook’s name and all attributes it produces will be suffixed with this `id`.



    Raises:
        ValueError: If neg_sampler is not provided.

    Key words: heterogeneous graph, link prediction, evaluation, negative sampler, thgl.
    """

    _cls_requires = {'edge_src', 'edge_dst', 'edge_time', 'edge_type'}
    _cls_produces = {'neg', 'neg_batch_list', 'neg_time'}
    _dataset_prefix = 'thgl'

    def __init__(
        self,
        dataset_name: str,
        split_mode: str,
        first_node_id: int,
        last_node_id: int,
        node_type: torch.Tensor,
        id: str | None = None,
    ) -> None:
        if first_node_id < 0 or last_node_id < 0:
            raise ValueError('First and last ID of node must be positive')

        if node_type is None:
            raise ValueError('Node type must not be None')

        if node_type.shape[0] < last_node_id:
            raise ValueError(f'last_node_id {last_node_id} must be within node_type')

        self._first_node_id = first_node_id
        self._last_node_id = last_node_id
        self._node_type = node_type
        super().__init__(dataset_name, split_mode, id)

    def _build_sampler(self, dataset_name: str) -> Any:
        try:
            from tgb.linkproppred.thg_negative_sampler import (
                THGNegativeEdgeSampler,
            )
        except ImportError:
            raise ImportError(
                f'TGB required for {self.__class__.__name__}, try `pip install py-tgb`'
            )

        return THGNegativeEdgeSampler(
            dataset_name=dataset_name,
            first_node_id=self._first_node_id,
            last_node_id=self._last_node_id,
            node_type=self._node_type.numpy(),
        )

    def _query_batch(self, batch: DGBatch) -> list:
        return self.neg_sampler.query_batch(
            batch.edge_src,
            batch.edge_dst,
            batch.edge_time,
            batch.edge_type,
            split_mode=self.split_mode,
        )


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
