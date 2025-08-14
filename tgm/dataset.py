from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Final, Generic, Optional, Tuple, TypeVar

import numpy as np
import torch

from tgm import DGraph
from tgm.data import DGData
from tgm.timedelta import TGB_TIME_DELTAS, TimeDeltaDG

T = TypeVar('T')


class _SplitsMixin(Generic[T]):
    def __init__(self, train: Optional[T], val: Optional[T], test: Optional[T]):
        self._train = train
        self._val = val
        self._test = test

    @property
    def train(self) -> Optional[T]:
        return self._train

    @property
    def val(self) -> Optional[T]:
        return self._val

    @property
    def test(self) -> Optional[T]:
        return self._test

    def get_splits(self) -> Tuple[Optional[T], Optional[T], Optional[T]]:
        """Return the train, validation and test splits as a fixed-order tuple.

        Returns:
            Tuple[Optional[T], Optional[T], Optional[T]]:
                A tuple of (train, val, test). If a split was not created or included, its position in the tuple will be None.
        """
        return self.train, self.val, self.test


class DGDataset(_SplitsMixin[DGraph]):
    r"""Immutable container storing some combination of train, validation and test DGraphs."""

    def __init__(
        self,
        data: DGData,
        time_delta: TimeDeltaDG | str = 'r',
        device: str | torch.device = 'cpu',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> None:
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f'train_ratio must be in (0,1), got {train_ratio}')
        if not (0.0 <= val_ratio < 1.0):
            raise ValueError(f'val_ratio must be in [0,1), got {val_ratio}')
        if train_ratio + val_ratio >= 1.0:
            raise ValueError(
                f'train_ratio + val_ratio must be less than 1.0, '
                f'got {train_ratio + val_ratio}'
            )

        train_data, val_data, test_data = self._split_data(data, train_ratio, val_ratio)
        super().__init__(
            train=DGraph(train_data, time_delta, device) if train_data else None,
            val=DGraph(val_data, time_delta, device) if val_data else None,
            test=DGraph(test_data, time_delta, device) if test_data else None,
        )

    @staticmethod
    def _split_data(
        data: DGData, train_ratio: float, val_ratio: float
    ) -> Tuple[Optional[DGData], ...]:
        num_events = data.timestamps.shape[0]
        train_end = int(num_events * train_ratio)
        val_end = train_end + int(num_events * val_ratio)

        def slice_data(start: int, end: int) -> Optional[DGData]:
            if start >= end:
                return None

            edge_mask = (data.edge_event_idx >= start) & (data.edge_event_idx < end)
            node_mask = None
            if data.node_event_idx is not None:
                node_mask = (data.node_event_idx >= start) & (data.node_event_idx < end)

            return DGData(
                timestamps=data.timestamps[start:end],
                edge_event_idx=data.edge_event_idx[edge_mask],
                edge_index=data.edge_index[edge_mask],
                edge_feats=data.edge_feats[edge_mask]
                if data.edge_feats is not None
                else None,
                node_event_idx=data.node_event_idx[node_mask]  # type: ignore
                if node_mask is not None
                else None,
                node_ids=data.node_ids[node_mask] if node_mask is not None else None,  # type: ignore
                dynamic_node_feats=data.dynamic_node_feats[node_mask]
                if node_mask is not None and data.dynamic_node_feats is not None
                else None,
                static_node_feats=data.static_node_feats,
            )

        train_data = slice_data(0, train_end)
        val_data = slice_data(train_end, val_end)
        test_data = slice_data(val_end, num_events)
        return train_data, val_data, test_data


class PreSplitDGDataset(_SplitsMixin[DGraph], ABC):
    r"""Base class for datasets with predefined train/val/test splits."""

    def __init__(
        self,
        time_delta: str | TimeDeltaDG = 'r',
        device: str | torch.device = 'cpu',
    ):
        train_data, val_data, test_data = self._load_splits()

        # TODO: Fix the discretization here from previous PR based on native_time_delta and user-requested
        # time_delta (which is really a discretization time_delta. Also, need to add reduce_op)
        super().__init__(
            train=DGraph(train_data, time_delta, device) if train_data else None,
            val=DGraph(val_data, time_delta, device) if val_data else None,
            test=DGraph(test_data, time_delta, device) if test_data else None,
        )

    @abstractmethod
    def _load_splits(
        self,
    ) -> Tuple[Optional[DGData], Optional[DGData], Optional[DGData]]: ...

    @property
    @abstractmethod
    def native_time_delta(self) -> TimeDeltaDG: ...


class TGBDataset(PreSplitDGDataset):
    TGB_TIME_DELTAS: Final[Dict[str, TimeDeltaDG]] = {
        'tgbl-wiki': TimeDeltaDG('s'),
        'tgbl-subreddit': TimeDeltaDG('s'),
        'tgbl-lastfm': TimeDeltaDG('s'),
        'tgbl-review': TimeDeltaDG('s'),
        'tgbl-coin': TimeDeltaDG('s'),
        'tgbl-flight': TimeDeltaDG('s'),
        'tgbl-comment': TimeDeltaDG('s'),
        'tgbn-trade': TimeDeltaDG('Y'),
        'tgbn-genre': TimeDeltaDG('s'),
        'tgbn-reddit': TimeDeltaDG('s'),
        'tgbn-token': TimeDeltaDG('s'),
    }

    def __init__(
        self,
        name: str,
        time_delta: str | TimeDeltaDG = 'r',
        device: str | torch.device = 'cpu',
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.kwargs = kwargs
        super().__init__(time_delta=time_delta, device=device)

    @property
    def native_time_delta(self) -> TimeDeltaDG:
        return TGB_TIME_DELTAS[self.name]

    def _load_splits(
        self,
    ) -> Tuple[Optional[DGData], Optional[DGData], Optional[DGData]]:
        def _check_tgb_import() -> tuple['LinkPropPredDataset', 'NodePropPredDataset']:  # type: ignore
            try:
                from tgb.linkproppred.dataset import LinkPropPredDataset
                from tgb.nodeproppred.dataset import NodePropPredDataset

                return LinkPropPredDataset, NodePropPredDataset
            except ImportError:
                err_msg = 'User requires tgb to initialize a DGraph from a tgb dataset '
                raise ImportError(err_msg)

        LinkPropPredDataset, NodePropPredDataset = _check_tgb_import()

        if self.name.startswith('tgbl-'):
            dataset = LinkPropPredDataset(name=self.name, **self.kwargs)  # type: ignore
        elif self.name.startswith('tgbn-'):
            dataset = NodePropPredDataset(name=self.name, **self.kwargs)  # type: ignore
        else:
            raise ValueError(f'Unknown dataset: {self.name}')

        data = dataset.full_data

        def _split_tgb_dataset(split: str) -> DGData:
            mask = getattr(dataset, f'{split}_mask')
            src = data['sources'][mask]

            dst = data['destinations'][mask]
            edge_index = torch.from_numpy(np.stack([src, dst], axis=1)).long()
            timestamps = torch.from_numpy(data['timestamps'][mask]).long()
            if data['edge_feat'] is None:
                edge_feats = None
            else:
                edge_feats = torch.from_numpy(data['edge_feat'][mask])

            node_timestamps, node_ids, dynamic_node_feats = None, None, None
            if self.name.startswith('tgbn-'):
                if 'node_label_dict' in data:
                    # in TGB, after passing a batch of edges, you find the nearest node event batch in the past
                    # in tgbn-trade, validation edge starts at 2010 while the first node event batch starts at 2009.
                    # therefore we do (timestamps[0] - 1) to account for this behaviour
                    node_label_dict = {
                        t: v
                        for t, v in data['node_label_dict'].items()
                        if (timestamps[0] - 1)
                        <= t
                        < timestamps[
                            -1
                        ]  # include the batch of labels even if they start before the edge events.
                    }
                else:
                    raise ValueError(
                        'please update your tgb package or install by source'
                    )

                if len(node_label_dict):
                    # Node events could be missing from the current data split (e.g. validation)
                    num_node_events = 0
                    node_label_dim = 0
                    for t in node_label_dict:
                        for node_id, label in node_label_dict[t].items():
                            num_node_events += 1
                            node_label_dim = label.shape[0]

                    temp_node_timestamps = np.zeros(num_node_events, dtype=np.int64)
                    temp_node_ids = np.zeros(num_node_events, dtype=np.int64)
                    temp_dynamic_node_feats = np.zeros(
                        (num_node_events, node_label_dim), dtype=np.float32
                    )
                    idx = 0
                    for t in node_label_dict:
                        for node_id, label in node_label_dict[t].items():
                            temp_node_timestamps[idx] = t
                            temp_node_ids[idx] = node_id
                            temp_dynamic_node_feats[idx] = label
                            idx += 1
                    node_timestamps = torch.from_numpy(temp_node_timestamps).long()
                    node_ids = torch.from_numpy(temp_node_ids).long()
                    dynamic_node_feats = torch.from_numpy(
                        temp_dynamic_node_feats
                    ).float()

            # Read static node features if they exist
            static_node_feats = None
            if dataset.node_feat is not None:
                static_node_feats = torch.from_numpy(dataset.node_feat)

            return DGData.from_raw(
                edge_timestamps=timestamps,
                edge_index=edge_index,
                edge_feats=edge_feats,
                node_timestamps=node_timestamps,
                node_ids=node_ids,
                dynamic_node_feats=dynamic_node_feats,
                static_node_feats=static_node_feats,
            )

        train_data = _split_tgb_dataset('train')
        val_data = _split_tgb_dataset('val')
        test_data = _split_tgb_dataset('test')
        return train_data, val_data, test_data
