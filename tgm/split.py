from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

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
class TimeSplit(SplitStrategy):
    val_time: int
    test_time: int

    def __post_init__(self) -> None:
        if not (0 <= self.val_time <= self.test_time):
            raise ValueError(
                f'Expected 0 <= val_time <= test_time, got {self.val_time}, {self.test_time}'
            )

    def apply(self, data: 'DGData') -> Tuple['DGData', ...]:  # type: ignore
        edge_times = data.timestamps[data.edge_event_idx]
        train_mask = edge_times < self.val_time
        val_mask = (edge_times >= self.val_time) & (edge_times < self.test_time)
        test_mask = edge_times >= self.test_time

        node_mask = None
        if data.node_ids is not None:
            node_times = data.timestamps[data.node_event_idx]
            node_mask = node_times < self.test_time  # keep all nodes before end of test

        splits = []
        for mask in (train_mask, val_mask, test_mask):
            if not mask.any():
                continue
            node_mask = None
            if data.node_ids is not None:
                node_times = data.timestamps[data.node_event_idx]
                # include nodes between first edge-1 and last edge
                first_edge = edge_times[mask].min()
                last_edge = edge_times[mask].max()
                node_mask = (node_times >= (first_edge - 1)) & (
                    node_times < last_edge + 1
                )
            splits.append(self._masked_copy(data, mask, node_mask))

        return tuple(splits)


@dataclass
class RatioSplit(SplitStrategy):
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

        time_split = TimeSplit(val_time=val_time, test_time=test_time)
        return time_split.apply(data)


@dataclass
class TGBSplit(SplitStrategy):
    train_mask: Tensor
    val_mask: Tensor
    test_mask: Tensor

    def apply(self, data: 'DGData') -> Tuple['DGData', 'DGData', 'DGData']:  # type: ignore
        splits = []
        for mask in (self.train_mask, self.val_mask, self.test_mask):
            edge_mask = mask
            node_mask = None
            if data.node_ids is not None:
                node_times = data.timestamps[data.node_event_idx]
                edge_times = data.timestamps[data.edge_event_idx][mask]
                if edge_times.numel() > 0:
                    first_edge = edge_times.min()
                    last_edge = edge_times.max()
                    node_mask = (node_times >= (first_edge - 1)) & (
                        node_times <= last_edge
                    )
            splits.append(self._masked_copy(data, edge_mask, node_mask))
        return tuple(splits)
