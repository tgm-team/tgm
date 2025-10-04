from typing import Union, Type
import inspect

from .base import DGStorageBase, DGSliceTracker
from .backends import DGStorageBackends, DGStorage

from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


def get_dg_storage_backend() -> Type:
    return DGStorage


def set_dg_storage_backend(backend: Union[str, DGStorageBase]) -> None:
    global DGStorage

    if inspect.isclass(backend) and issubclass(backend, DGStorageBase):
        DGStorage = backend
        logger.debug('DGStorage backend set to: %s', DGStorage.__name__)
    elif isinstance(backend, str) and backend in DGStorageBackends:
        DGStorage = DGStorageBackends[backend]
        logger.debug('DGStorage backend set to: %s', DGStorage.__name__)
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
