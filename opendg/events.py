from abc import ABC
from typing import Optional, Tuple

from torch import Tensor


class Event(ABC):
    r"""An abstract event that occured in a dynamic graph."""

    def __init__(self, time: int) -> None:
        self._time = time

    @property
    def time(self) -> int:
        return self._time


class NodeEvent(Event):
    r"""A Node event that occured in a dynamic graph."""

    def __init__(
        self, time: int, node_id: int, features: Optional[Tensor] = None
    ) -> None:
        super().__init__(time)

        self._node_id = node_id
        self._features = features

    @property
    def node_id(self) -> int:
        return self._node_id

    @property
    def features(self) -> Optional[Tensor]:
        return self._features


class EdgeEvent(Event):
    r"""An Edge event that occured in a dynamic graph."""

    def __init__(
        self,
        time: int,
        edge: Tuple[int, int],
        features: Optional[Tensor] = None,
    ) -> None:
        super().__init__(time)

        self._edge = edge
        self._features = features

    @property
    def edge(self) -> Tuple[int, int]:
        return self._edge

    @property
    def features(self) -> Optional[Tensor]:
        return self._features
