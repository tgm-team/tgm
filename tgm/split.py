from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor


class SplitStrategy(ABC):
    @abstractmethod
    def apply(self, data: 'DGData') -> Tuple['DGData', ...]: ...  # type: ignore

    def _masked_copy(
        self,
        data: 'DGData',  # type: ignore
        edge_mask: Tensor,
        node_mask: Tensor | None = None,
    ) -> 'DGData':  # type: ignore
        from tgm import DGData  # avoid circular dependency

        if edge_mask.dtype != torch.bool:
            raise ValueError('edge_mask must be a boolean tensor')
        edge_index = data.edge_index[edge_mask]
        edge_feats = data.edge_feats[edge_mask] if data.edge_feats is not None else None
        edge_timestamps = data.timestamps[data.edge_event_idx[edge_mask]]

        node_ids, dynamic_node_feats, node_timestamps = None, None, None
        if data.node_ids is not None:
            if node_mask is None:
                node_mask = torch.ones(data.node_ids.shape[0], dtype=torch.bool)
            if node_mask.dtype != torch.bool:
                raise ValueError('node_mask must be a boolean tensor')
            node_ids = data.node_ids[node_mask]
            node_timestamps = data.timestamps[data.node_event_idx[node_mask]]
            if data.dynamic_node_feats is not None:
                dynamic_node_feats = data.dynamic_node_feats[node_mask]

        # Static features are shared across splits, do not clone
        static_node_feats = data.static_node_feats

        # In case we masked out to the point of empty node events, change to None
        if node_ids is not None and node_ids.numel() == 0:
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
        for start, end in ranges.values():
            edge_mask = (edge_times >= start) & (edge_times < end)
            if not edge_mask.any():
                continue

            node_mask = None
            if node_times is not None:
                node_mask = (node_times >= start) & (node_times < end)

            splits.append(self._masked_copy(data, edge_mask, node_mask))

        return tuple(splits)


@dataclass
class TemporalRatioSplit(SplitStrategy):
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
        edge_times = data.timestamps[data.edge_event_idx]
        if edge_times.numel() == 0:
            return tuple()  # no edges

        min_time, max_time = int(edge_times.min()), int(edge_times.max())
        total_span = max_time - min_time + 1

        val_time = min_time + int(total_span * self.train_ratio)
        test_time = val_time + int(total_span * self.val_ratio)

        time_split = TemporalSplit(val_time=val_time, test_time=test_time)
        return time_split.apply(data)


@dataclass
class TGBSplit(SplitStrategy):
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
                        node_times <= edge_end_time
                    )

            splits.append(self._masked_copy(data, edge_mask, node_mask))

        return tuple(splits)
