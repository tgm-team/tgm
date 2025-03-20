from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from torch import Tensor


@dataclass(slots=True)
class NodeEvent:
    r"""A Node event that occured in a dynamic graph."""

    t: int
    src: int
    features: Optional[Tensor] = None


@dataclass(slots=True)
class EdgeEvent:
    r"""An Edge event that occured in a dynamic graph."""

    t: int
    src: int
    dst: int
    features: Optional[Tensor] = None

    @property
    def edge(self) -> Tuple[int, int]:
        return (self.src, self.dst)


Event = NodeEvent | EdgeEvent
