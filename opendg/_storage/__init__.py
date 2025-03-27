from typing import Union, Type
import inspect

from .base import DGStorageBase, DGSliceTracker
from .backends import DGStorageBackends, DGStorage


def get_dg_storage_backend() -> Type:
    return DGStorage


def set_dg_storage_backend(backend: Union[str, DGStorageBase]) -> None:
    global DGStorage

    if inspect.isclass(backend) and issubclass(backend, DGStorageBase):
        DGStorage = backend
    elif isinstance(backend, str) and backend in DGStorageBackends:
        DGStorage = DGStorageBackends[backend]
    else:
        raise ValueError(
            f'Unrecognized DGStorage backend: {backend}, expected one of: {list(DGStorageBackends.keys())}'
        )


__all__ = [
    'DGStorage',
    'DGStorageBackends',
    'DGSliceTracker',
    'get_dg_storage_backend',
    'set_dg_storage_backend',
]
