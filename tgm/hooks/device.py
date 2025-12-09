from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any

import torch

from tgm import DGBatch, DGraph
from tgm.hooks import StatelessHook
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


class PinMemoryHook(StatelessHook):
    """Pin all tensors in the DGBatch to page-locked memory for faster async CPU-GPU transfers."""

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        if torch.cuda.is_available():
            pin_if_needed = (
                lambda x: x.pin_memory() if not x.is_cuda and not x.is_pinned() else x
            )

            _apply_to_tensors_inplace(batch, pin_if_needed)
        return batch


class DeviceTransferHook(StatelessHook):
    """Moves all tensors in the DGBatch to the specified device."""

    def __init__(self, device: str | torch.device) -> None:
        self.device = torch.device(device)

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        move_if_needed = (
            lambda x: x.to(device=self.device, non_blocking=True)
            if x.device != self.device
            else x
        )

        _apply_to_tensors_inplace(batch, move_if_needed)
        return batch


def _apply_to_tensors_inplace(obj: Any, fn: Any) -> Any:
    if torch.is_tensor(obj):
        return fn(obj)
    elif is_dataclass(obj):
        for k, v in vars(obj).items():
            setattr(obj, k, _apply_to_tensors_inplace(v, fn))
        return obj
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = _apply_to_tensors_inplace(obj[i], fn)
        return obj
    elif isinstance(obj, tuple):
        # Tuples are immutable, so return a new tuple
        return tuple(_apply_to_tensors_inplace(x, fn) for x in obj)
    elif isinstance(obj, dict):
        for k in obj:
            obj[k] = _apply_to_tensors_inplace(obj[k], fn)
        return obj
    else:
        return obj
