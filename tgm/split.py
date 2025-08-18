from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from torch import Tensor


class SplitStrategy(ABC):
    @abstractmethod
    def apply(self, data: 'DGData') -> Tuple['DGData', ...]: ...  # type: ignore


@dataclass
class TimeSplit(SplitStrategy):
    val_time: int
    test_time: int

    def apply(self, data: 'DGData') -> Tuple['DGData', 'DGData', 'DGData']:  # type: ignore
        return data, data, data


@dataclass
class RatioSplit(SplitStrategy):
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f'train_ratio + val_ratio + test_ratio must sum to 1.0, got {total}'
            )

    def apply(self, data: 'DGData') -> Tuple['DGData', 'DGData', 'DGData']:  # type: ignore
        return data, data, data


@dataclass
class TGBSplit(SplitStrategy):
    train_mask: Tensor
    val_mask: Tensor
    test_mask: Tensor

    def apply(self, data: 'DGData') -> Tuple['DGData', 'DGData', 'DGData']:  # type: ignore
        return data, data, data
