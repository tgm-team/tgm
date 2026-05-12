from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import torch

from tgm import DGBatch, DGraph
from tgm.constants import PADDED_NODE_ID
from tgm.hooks import StatefulHook, StatelessHook
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


class NegativeEdgeSamplerHook(StatefulHook):
    """Sample negative edges for dynamic link prediction.

    Args:
        low (int): The minimum node id to sample
        high (int) : The maximum node id to sample
        neg_ratio (float): The ratio of sampled negative destination nodes
            to the number of positive destination nodes (default = 1.0).
        id (str): A unique identifier for the hook. The hook’s name and all attributes it produces will be suffixed with this `id`.
    """

    _cls_requires = {'edge_src', 'edge_dst', 'edge_time'}
    _cls_produces = {'neg', 'neg_time'}
    VALID_STRATEGY = ['rnd', 'hist_rnd']

    def __init__(
        self,
        low: int,
        high: int,
        neg_ratio: float = 1.0,
        id: str | None = None,
        strategy: Literal['rnd', 'hist_rnd'] = 'rnd',
    ) -> None:
        super().__init__()
        if strategy not in self.VALID_STRATEGY:
            raise ValueError(
                f'{strategy} is not supported. Valid sampling strategies: {self.VALID_STRATEGY}'
            )
        if not 0 < neg_ratio <= 1:
            raise ValueError(f'neg_ratio must be in (0, 1], got: {neg_ratio}')
        if not low < high:
            raise ValueError(f'low ({low}) must be strictly less than high ({high})')
        self.low = low
        self.high = high
        self.neg_ratio = neg_ratio
        self._id = id
        self.strategy = strategy

        if strategy == 'hist_rnd':
            self._memory: torch.Tensor | None = None
            self._count: int = 0
        self.__post_init__()

    def reset_state(self) -> None:
        self._memory = None
        self._count = 0

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        size = (round(self.neg_ratio * batch.edge_dst.size(0)),)

        if size[0] == 0:
            neg = torch.empty(size, dtype=torch.int32, device=dg.device)
            neg_time = torch.empty(size, dtype=torch.int64, device=dg.device)
        elif self.strategy == 'rnd':
            neg, neg_time = self._random_sampling(dg, batch)

            neg, neg_time = neg[: size[0]], neg_time[: size[0]]

        elif self.strategy == 'hist_rnd':
            if self._count == 0:
                neg, neg_time = self._random_sampling(dg, batch)
                neg, neg_time = neg[: size[0]], neg_time[: size[0]]

            else:
                rnd_size = round(size[0] * 0.5)
                hst_size = size[0] - rnd_size
                neg_rnd, neg_time_rnd = self._random_sampling(dg, batch)
                neg_hst, neg_time_hst = self._random_hist_sampling(dg, batch)

                original_valid_mask = neg_hst != PADDED_NODE_ID
                valid_idx = torch.where(original_valid_mask)[0]
                cutoff = min(hst_size, valid_idx.size(0))

                neg = neg_rnd.clone()
                neg_time = neg_time_rnd.clone()

                chosen = valid_idx[:cutoff]
                neg[chosen] = neg_hst[chosen]
                neg_time[chosen] = neg_time_hst[chosen]

            self._update_hst_memory(dg, batch)

        self.add_batch_attribute(batch, 'neg', neg)
        self.add_batch_attribute(batch, 'neg_time', neg_time)
        return batch

    def _random_sampling(
        self, dg: DGraph, batch: DGBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample negative destination nodes uniformly at random.

        For each edge in the batch, samples a random destination node from the
        range [low, high) as the negative sample, paired with the original edge
        timestamp.

        Args:
            dg (DGraph): The dynamic graph, used to determine the target device.
            batch (DGBatch): The current batch of edges.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of:
                - neg (torch.Tensor): Randomly sampled negative destination nodes
                    of shape (batch_size,) and dtype int32.
                - neg_time (torch.Tensor): Timestamps corresponding to each negative
                    sample, cloned from batch.edge_time.
        """
        size = (batch.edge_dst.size(0),)
        neg = torch.randint(
            self.low, self.high, size, dtype=torch.int32, device=dg.device
        )
        neg_time = batch.edge_time.clone()

        return neg, neg_time

    def _random_hist_sampling(
        self, dg: DGraph, batch: DGBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample negative destination nodes from each source node's historical interactions.

        For each source node in the batch, randomly selects a destination node from
        its past interactions stored in memory. If a source node has no recorded past
        interactions, its corresponding negative sample is set to PADDED_NODE_ID as
        a sentinel value indicating no history is available.

        The random selection is performed via a vectorized scatter-max over random
        weights assigned to each historical edge, avoiding explicit loops.

        Args:
            dg (DGraph): The dynamic graph, used to determine the target device.
            batch (DGBatch): The current batch of edges.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of:
                - neg (torch.Tensor): Historically sampled negative destination nodes
                    of shape (batch_size,) and dtype int32. Nodes with no historical
                    interactions are set to PADDED_NODE_ID.
                - neg_time (torch.Tensor): Timestamps corresponding to each negative
                    sample, cloned from batch.edge_time.

        Note:
            Assumes self._memory is a tensor of shape (2, num_observed_edges) where
            row 0 contains source nodes and row 1 contains destination nodes of all
            previously observed edges.
        """
        assert self._memory is not None

        mask = torch.isin(self._memory[0], batch.edge_src)
        sampling_edges = self._memory[:, mask]

        # Group duplicate srcs: for each unique src, collect all batch positions
        unique_srcs, inverse = torch.unique(batch.edge_src, return_inverse=True)

        unique_src_to_idx = torch.zeros(
            (int(batch.edge_src.max().item()) + 1,), dtype=torch.long, device=dg.device
        )
        unique_src_to_idx[unique_srcs] = torch.arange(
            unique_srcs.size(0), device=dg.device
        )

        edge_to_unique_idx = unique_src_to_idx[sampling_edges[0]]

        sampling_edges_random_weights = torch.rand(
            sampling_edges.size(1), device=dg.device
        )
        best_weights = torch.full((unique_srcs.size(0),), -1.0, device=dg.device)

        best_weights.scatter_reduce_(
            0, edge_to_unique_idx, sampling_edges_random_weights, reduce='amax'
        )
        best_edge_mask = (
            sampling_edges_random_weights == best_weights[edge_to_unique_idx]
        )

        # Sample one neg per unique src
        neg_per_unique = torch.full(
            (unique_srcs.size(0),),
            PADDED_NODE_ID,
            dtype=sampling_edges.dtype,
            device=dg.device,
        )
        neg_per_unique[edge_to_unique_idx[best_edge_mask]] = sampling_edges[
            1, best_edge_mask
        ]

        # Broadcast back to all batch positions (duplicates get the same sampled neg)
        neg = neg_per_unique[inverse]
        neg_time = batch.edge_time.clone()
        return neg, neg_time

    def _update_hst_memory(self, dg: DGraph, batch: DGBatch) -> None:
        """Append the current batch of edges to the historical memory buffer.

        Maintains a dynamically resizing memory buffer of observed edges for use
        in historical negative sampling. The buffer doubles in size when capacity
        is exceeded, ensuring expected O(1) time complexity insertion and amortized O(E) space complexity
        where E is the total number of observed edges.

        Args:
            dg (DGraph): The dynamic graph, used to determine the target device.
            batch (DGBatch): The current batch of edges whose source and destination
                nodes will be appended to memory.

        Note:
            - Memory is lazily initialized on the first call with twice the initial
            batch size as the starting capacity.
            - When the buffer is full, it is expanded to the maximum of twice its
            current size or twice the required size, ensuring no immediate
            back-to-back resizes even for large batches.
            - Only source and destination nodes are stored; edge timestamps are
            not retained in memory.
            - This scale linear w.r.t the number of interaction event rather than number of edges.
            Since _memory can contain duplicated edges.
        """
        batch_size = batch.edge_src.size(0)

        if self._memory is None:
            self._memory = torch.zeros(
                (2, batch_size * 2), dtype=torch.int32, device=dg.device
            )

        if self._count + batch_size > self._memory.size(1):
            new_size = max(self._memory.size(1) * 2, (self._count + batch_size) * 2)

            edge_buffer = torch.zeros(
                (2, new_size - self._memory.size(1)),
                dtype=torch.int32,
                device=dg.device,
            )
            self._memory = torch.cat([self._memory, edge_buffer], dim=1)

        self._memory[0, self._count : self._count + batch_size] = batch.edge_src
        self._memory[1, self._count : self._count + batch_size] = batch.edge_dst

        self._count += batch_size


class TGBNegativeEdgeSamplerHook(StatelessHook):
    """Load data from DGraph using pre-generated TGB negative samples.
    Make sure to perform `dataset.load_val_ns()` or `dataset.load_test_ns()` before using this hook.

    Args:
        dataset_name (str): The name of the TGB dataset to produce sampler for.
        split_mode (str): The split mode to use for sampling, either 'val' or 'test'.
        id (str): A unique identifier for the hook. The hook’s name and all attributes it produces will be suffixed with this `id`.

    Raises:
        ValueError: If neg_sampler is not provided.
    """

    _cls_requires = {'edge_src', 'edge_dst', 'edge_time'}
    _cls_produces = {'neg', 'neg_batch_list', 'neg_time'}

    def __init__(
        self, dataset_name: str, split_mode: str, id: str | None = None
    ) -> None:
        super().__init__()
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
        self._id = id
        self.__post_init__()

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        if batch.edge_src.size(0) == 0:
            batch_neg = torch.empty(
                batch.edge_src.size(0), dtype=torch.int32, device=dg.device
            )
            batch_neg_time = torch.empty(
                batch.edge_src.size(0), dtype=torch.int64, device=dg.device
            )
            batch_neg_batch_list = []
            # return batch  # empty batch
        else:
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


class TGBTHGNegativeEdgeSamplerHook(StatelessHook):
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
    """

    _cls_requires = {'edge_src', 'edge_dst', 'edge_time', 'edge_type'}
    _cls_produces = {'neg', 'neg_batch_list', 'neg_time'}

    def __init__(
        self,
        dataset_name: str,
        split_mode: str,
        first_node_id: int,
        last_node_id: int,
        node_type: torch.Tensor,
        id: str | None = None,
    ) -> None:
        super().__init__()
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

        self._id = id

        self.__post_init__()

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


class TGBTKGNegativeEdgeSamplerHook(StatelessHook):
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
    """

    _cls_requires = {'edge_src', 'edge_dst', 'edge_time', 'edge_type'}
    _cls_produces = {'neg', 'neg_batch_list', 'neg_time'}

    def __init__(
        self,
        dataset_name: str,
        split_mode: str,
        first_dst_id: int,
        last_dst_id: int,
        id: str | None = None,
    ) -> None:
        super().__init__()
        if split_mode not in ['val', 'test']:
            raise ValueError(f'split_mode must be "val" or "test", got: {split_mode}')

        if first_dst_id < 0 or last_dst_id < 0:
            raise ValueError('First and last ID of node must be positive')

        try:
            from tgb.linkproppred.tkg_negative_sampler import (
                TKGNegativeEdgeSampler,
            )
            from tgb.utils.info import DATA_VERSION_DICT, PROJ_DIR
        except ImportError:
            raise ImportError(
                f'TGB required for {self.__class__.__name__}, try `pip install py-tgb`'
            )

        if not dataset_name.startswith('tkgl-'):
            raise ValueError(
                'TGBTKGNegativeEdgeSamplerHook should only be registered for '
                f'"tkgl-xxx" datasets, but got: {dataset_name}'
            )

        neg_sampler = TKGNegativeEdgeSampler(
            dataset_name=dataset_name,
            first_dst_id=first_dst_id,
            last_dst_id=last_dst_id,
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
        self._id = id
        self.__post_init__()

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
                neg_batch_list = self.neg_sampler.query_batch(
                    batch.edge_src,
                    batch.edge_dst,
                    batch.edge_time,
                    batch.edge_type,
                    split_mode=self.split_mode,
                )
            except ValueError as e:
                raise ValueError(
                    f'TKGL Negative sampling failed for split_mode={self.split_mode}. Try updating your TGB package: `pip install --upgrade py-tgb`'
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
