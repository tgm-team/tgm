from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor

from tgm.util.logging import _get_logger, pretty_number_format

logger = _get_logger(__name__)


class SplitStrategy(ABC):
    """Abstract base class for splitting temporal graph datasets.

    Implementations of this class define the logic for dividing a `DGData` object
    into one or more subsets (train/val/test) based on temporal information.
    """

    @abstractmethod
    def apply(self, data: 'DGData') -> Tuple['DGData', ...]:  # type: ignore
        r"""Split the dataset and return one or more subsets.

        Args:
            data (DGData): The temporal graph dataset to split.

        Returns:
            Tuple[DGData, ...]: Split datasets according to the strategy.
        """

    def _masked_copy(
        self,
        data: 'DGData',  # type: ignore
        edge_mask: Tensor,
        node_mask: Tensor | None = None,
    ) -> 'DGData':  # type: ignore
        from tgm.data import DGData  # avoid circular dependency

        edge_index = data.edge_index[edge_mask]
        edge_feats = data.edge_feats[edge_mask] if data.edge_feats is not None else None
        edge_timestamps = data.timestamps[data.edge_event_idx[edge_mask]]

        node_ids, dynamic_node_feats, node_timestamps = None, None, None
        if data.node_ids is not None:
            if node_mask is None:
                node_mask = torch.ones(data.node_ids.shape[0], dtype=torch.bool)
            node_ids = data.node_ids[node_mask]
            node_timestamps = data.timestamps[data.node_event_idx[node_mask]]
            if data.dynamic_node_feats is not None:
                dynamic_node_feats = data.dynamic_node_feats[node_mask]

        # Static features are shared across splits, do not clone
        static_node_feats = data.static_node_feats

        # In case we masked out to the point of empty node events, change to None
        if node_ids is not None and node_ids.numel() == 0:
            logger.warning(
                'All nodes masked out, resetting node_ids/node_timestamps/dynamic_node_feats to None'
            )
            node_ids = node_timestamps = dynamic_node_feats = None

        return DGData.from_raw(
            time_delta=data.time_delta,
            edge_timestamps=edge_timestamps,
            edge_index=edge_index,
            edge_feats=edge_feats,
            node_timestamps=node_timestamps,
            node_ids=node_ids,
            dynamic_node_feats=dynamic_node_feats,
            static_node_feats=static_node_feats,
        )


@dataclass
class TemporalSplit(SplitStrategy):
    """Split a temporal graph dataset based on absolute timestamp boundaries.

    Args:
        val_time (int): The timestamp separating training and validation data.
        test_time (int): The timestamp separating validation and test data.

    Raises:
        ValueError: If not 0 <= val_time <= test_time.

    Note:
        This strategy assigns edges and nodes to splits based on whether their
        timestamps fall within the corresponding intervals:
        - Train: (-inf, val_time)
        - Validation: [val_time, test_time)
        - Test: [test_time, inf)
    """

    val_time: int
    test_time: int

    def __post_init__(self) -> None:
        if not (0 <= self.val_time <= self.test_time):
            raise ValueError(
                f'Expected 0 <= val_time <= test_time, got {self.val_time}, {self.test_time}'
            )

    def apply(self, data: 'DGData') -> Tuple['DGData', ...]:  # type: ignore
        edge_times = data.timestamps[data.edge_event_idx]
        node_times = None
        if data.node_ids is not None:
            node_times = data.timestamps[data.node_event_idx]

        ranges = {
            'train': (-float('inf'), self.val_time),
            'val': (self.val_time, self.test_time),
            'test': (self.test_time, float('inf')),
        }

        splits = []
        for split_name, (start, end) in ranges.items():
            edge_mask = (edge_times >= start) & (edge_times < end)
            if not edge_mask.any():
                logger.warning(
                    'No edges found in %s split time range [%s, %s)',
                    split_name,
                    pretty_number_format(start),
                    pretty_number_format(end),
                )

            node_mask = None
            if node_times is not None:
                node_mask = (node_times >= start) & (node_times < end)
            split_data = self._masked_copy(data, edge_mask, node_mask)
            splits.append(split_data)
            logger.info(
                '%s split time range: [%s, %s),  %s edge events, %s node events',
                split_name,
                pretty_number_format(start),
                pretty_number_format(end),
                pretty_number_format(split_data.edge_index.size(0)),
                pretty_number_format(
                    0 if split_data.node_ids is None else split_data.node_ids.size(0)
                ),
            )

        return tuple(splits)


@dataclass
class TemporalRatioSplit(SplitStrategy):
    """Split a temporal graph dataset according to relative ratios of time.

    Args:
        train_ratio (float, optional): Fraction of data to assign to training. Defaults to 0.7.
        val_ratio (float, optional): Fraction of data to assign to validation. Defaults to 0.15.
        test_ratio (float, optional): Fraction of data to assign to test. Defaults to 0.15.

    Raises:
        ValueError: If any ratio is negative.
        ValueError: If the ratios do not sum to 1.0 within tolerance.

    Note:
        The dataset timestamps are assumed to be sorted. The ratios are applied
        cumulatively to the total temporal span to determine absolute split boundaries.
    """

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self) -> None:
        for r in (self.train_ratio, self.val_ratio, self.test_ratio):
            if r < 0:
                raise ValueError('Ratios must all be non-negative')

        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f'train_ratio + val_ratio + test_ratio must sum to 1.0, got {total}'
            )

    def apply(self, data: 'DGData') -> Tuple['DGData', ...]:  # type: ignore
        min_time, max_time = data.timestamps[0], data.timestamps[-1]  # it's sorted
        total_span = max_time - min_time + 1

        val_time = min_time + int(total_span * self.train_ratio)
        test_time = val_time + int(total_span * self.val_ratio)

        logger.info(
            'TemporalRatioSplit (train=%.2f, val=%.2f, test=%.2f): '
            'train time range: [%s, %s), val time range: =[%s, %s), test time range: [%s, %s]',
            self.train_ratio,
            self.val_ratio,
            self.test_ratio,
            pretty_number_format(min_time),
            pretty_number_format(val_time),
            pretty_number_format(val_time),
            pretty_number_format(test_time),
            pretty_number_format(test_time),
            pretty_number_format(max_time),
        )

        time_split = TemporalSplit(val_time=val_time, test_time=test_time)
        return time_split.apply(data)


@dataclass
class TGBSplit(SplitStrategy):
    """Split a temporal graph dataset using pre-specified edge time bounds.

    Args:
        split_bounds (Dict[str, Tuple[int, int]]): Mapping from split names
            ('train', 'val', 'test') to (min_time, max_time) intervals for edges.

    Note:
        Nodes are included in a split if their timestamps fall within the
        corresponding edge interval (or slightly before the min_time of edges).
    """

    split_bounds: Dict[str, Tuple[int, int]]  # min/max edge times for each split

    def apply(self, data: 'DGData') -> Tuple['DGData', 'DGData', 'DGData']:  # type: ignore
        splits = []
        for split_name in ['train', 'val', 'test']:
            edge_start_time, edge_end_time = self.split_bounds[split_name]
            edge_mask = (data.timestamps[data.edge_event_idx] >= edge_start_time) & (
                data.timestamps[data.edge_event_idx] <= edge_end_time
            )

            node_mask = None
            if data.node_ids is not None:
                node_times = data.timestamps[data.node_event_idx]
                if edge_mask.any():
                    node_mask = (node_times >= (edge_start_time - 1)) & (
                        node_times < edge_end_time
                    )

            split_data = self._masked_copy(data, edge_mask, node_mask)
            splits.append(split_data)
            logger.info(
                'TGB %s split time range [%s, %s], %s edge events, %s node events',
                split_name,
                pretty_number_format(edge_start_time),
                pretty_number_format(edge_end_time),
                pretty_number_format(split_data.edge_index.size(0)),
                pretty_number_format(
                    0 if split_data.node_ids is None else split_data.node_ids.size(0)
                ),
            )

        return tuple(splits)
