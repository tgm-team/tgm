from __future__ import annotations

from pathlib import Path

import torch

from tgm import DGBatch, DGraph
from tgm.hooks import StatelessHook
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


class NegativeEdgeSamplerHook(StatelessHook):
    """Sample negative edges for dynamic link prediction.

    Args:
        low (int): The minimum node id to sample
        high (int) : The maximum node id to sample
        neg_ratio (float): The ratio of sampled negative destination nodes
            to the number of positive destination nodes (default = 1.0).
    """

    requires = {'edge_src', 'edge_dst', 'edge_time'}
    produces = {'neg', 'neg_time'}

    def __init__(self, low: int, high: int, neg_ratio: float = 1.0) -> None:
        if not 0 < neg_ratio <= 1:
            raise ValueError(f'neg_ratio must be in (0, 1], got: {neg_ratio}')
        if not low < high:
            raise ValueError(f'low ({low}) must be strictly less than high ({high})')
        self.low = low
        self.high = high
        self.neg_ratio = neg_ratio

    # TODO: Historical vs. random
    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        size = (round(self.neg_ratio * batch.edge_dst.size(0)),)
        if size[0] == 0:
            batch.neg = torch.empty(size, dtype=torch.int32, device=dg.device)  # type: ignore
            batch.neg_time = torch.empty(size, dtype=torch.int64, device=dg.device)  # type: ignore
        else:
            batch.neg = torch.randint(  # type: ignore
                self.low, self.high, size, dtype=torch.int32, device=dg.device
            )
            batch.neg_time = batch.edge_time.clone()  # type: ignore
        return batch


class TGBNegativeEdgeSamplerHook(StatelessHook):
    """Load data from DGraph using pre-generated TGB negative samples.
    Make sure to perform `dataset.load_val_ns()` or `dataset.load_test_ns()` before using this hook.

    Args:
        dataset_name (str): The name of the TGB dataset to produce sampler for.
        split_mode (str): The split mode to use for sampling, either 'val' or 'test'.

    Raises:
        ValueError: If neg_sampler is not provided.
    """

    requires = {'edge_src', 'edge_dst', 'edge_time'}
    produces = {'neg', 'neg_batch_list', 'neg_time'}

    def __init__(self, dataset_name: str, split_mode: str) -> None:
        if split_mode not in ['val', 'test']:
            raise ValueError(f'split_mode must be "val" or "test", got: {split_mode}')

        try:
            from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
            from tgb.utils.info import DATA_VERSION_DICT, PROJ_DIR
        except ImportError:
            raise ImportError(
                f'TGB required for {self.__class__.__name__}, try `pip install py-tgb`'
            )

        if not dataset_name.startswith('tgbl-'):
            raise ValueError(
                'TGBNegativeEdgeSamplerHook should only be registered for '
                f'"tgbl-xxx" datasets, but got: {dataset_name}'
            )

        neg_sampler = NegativeEdgeSampler(dataset_name=dataset_name)

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

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        if batch.edge_src.size(0) == 0:
            batch.neg = torch.empty(  # type: ignore
                batch.edge_src.size(0), dtype=torch.int32, device=dg.device
            )
            batch.neg_time = torch.empty(  # type: ignore
                batch.edge_src.size(0), dtype=torch.int64, device=dg.device
            )
            batch.neg_batch_list = []  # type: ignore
            return batch  # empty batch
        try:
            neg_batch_list = self.neg_sampler.query_batch(
                batch.edge_src,
                batch.edge_dst,
                batch.edge_time,
                split_mode=self.split_mode,
            )
        except ValueError as e:
            raise ValueError(
                f'Negative sampling failed for split_mode={self.split_mode}. Try updating your TGB package: `pip install --upgrade py-tgb`'
            ) from e

        batch.neg_batch_list = [  # type: ignore
            torch.tensor(neg_batch, dtype=torch.int32, device=dg.device)
            for neg_batch in neg_batch_list
        ]
        batch.neg = torch.unique(torch.cat(batch.neg_batch_list))  # type: ignore

        # This is a heuristic. For our fake (negative) link times,
        # we pick random time stamps within [batch.start_time, batch.end_time].
        # Using random times on the whole graph will likely produce information
        # leakage, making the prediction easier than it should be.

        # Use generator to local constrain rng for reproducibility
        gen = torch.Generator(device=dg.device)
        gen.manual_seed(0)
        batch.neg_time = torch.randint(  # type: ignore
            int(batch.edge_time.min().item()),
            int(batch.edge_time.max().item()) + 1,
            (batch.neg.size(0),),  # type: ignore
            device=dg.device,
            generator=gen,
        )
        return batch


class TGBTHGNegativeEdgeSamplerHook(StatelessHook):
    """Load data from DGraph using pre-generated TGB negative samples for heterogeneous graph.
    Make sure to perform `dataset.load_val_ns()` or `dataset.load_test_ns()` before using this hook.

    Args:
        dataset_name (str): The name of the TGB dataset to produce sampler for.
        split_mode (str): The split mode to use for sampling, either 'val' or 'test'.
        first_node_id (int): identity of the first node
        last_node_id (int): identity of the last destination node
        node_type (Tensor): the node type of each node


    Raises:
        ValueError: If neg_sampler is not provided.
    """

    requires = {'edge_src', 'edge_dst', 'edge_time', 'edge_type'}
    produces = {'neg', 'neg_batch_list', 'neg_time'}

    def __init__(
        self,
        dataset_name: str,
        split_mode: str,
        first_node_id: int,
        last_node_id: int,
        node_type: torch.Tensor,
    ) -> None:
        if split_mode not in ['val', 'test']:
            raise ValueError(f'split_mode must be "val" or "test", got: {split_mode}')

        if first_node_id < 0 or last_node_id < 0:
            raise ValueError('First and last ID of node must be positive')

        if node_type is None:
            raise ValueError('Node type must not be None')

        if node_type.shape[0] < last_node_id:
            raise ValueError(f'last_node_id {last_node_id} must be within node_type')

        try:
            from tgb.linkproppred.thg_negative_sampler import (
                THGNegativeEdgeSampler,
            )
            from tgb.utils.info import DATA_VERSION_DICT, PROJ_DIR
        except ImportError:
            raise ImportError(
                f'TGB required for {self.__class__.__name__}, try `pip install py-tgb`'
            )

        if not dataset_name.startswith('thgl-'):
            raise ValueError(
                'TGBTHGNegativeEdgeSamplerHook should only be registered for '
                f'"thgl-xxx" datasets, but got: {dataset_name}'
            )

        neg_sampler = THGNegativeEdgeSampler(
            dataset_name=dataset_name,
            first_node_id=first_node_id,
            last_node_id=last_node_id,
            node_type=node_type.numpy(),
        )

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

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        if batch.edge_src.size(0) == 0:
            batch.neg = torch.empty(  # type: ignore
                batch.edge_src.size(0), dtype=torch.int32, device=dg.device
            )
            batch.neg_time = torch.empty(  # type: ignore
                batch.edge_src.size(0), dtype=torch.int64, device=dg.device
            )
            batch.neg_batch_list = []  # type: ignore
            return batch  # empty batch
        try:
            neg_batch_list = self.neg_sampler.query_batch(
                batch.edge_src,
                batch.edge_dst,
                batch.edge_time,
                batch.edge_type,
                split_mode=self.split_mode,
            )
        except ValueError as e:
            raise ValueError(
                f'THGL Negative sampling failed for split_mode={self.split_mode}. Try updating your TGB package: `pip install --upgrade py-tgb`'
            ) from e

        batch.neg_batch_list = [  # type: ignore
            torch.tensor(neg_batch, dtype=torch.int32, device=dg.device)
            for neg_batch in neg_batch_list
        ]
        batch.neg = torch.unique(torch.cat(batch.neg_batch_list))  # type: ignore

        # This is a heuristic. For our fake (negative) link times,
        # we pick random time stamps within [batch.start_time, batch.end_time].
        # Using random times on the whole graph will likely produce information
        # leakage, making the prediction easier than it should be.

        # Use generator to local constrain rng for reproducibility
        gen = torch.Generator(device=dg.device)
        gen.manual_seed(0)
        batch.neg_time = torch.randint(  # type: ignore
            int(batch.edge_time.min().item()),
            int(batch.edge_time.max().item()) + 1,
            (batch.neg.size(0),),  # type: ignore
            device=dg.device,
            generator=gen,
        )
        return batch
