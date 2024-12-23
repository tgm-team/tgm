from .base import DGStorageBase
from .dict_impl import DGStorageDictImpl

DGStorageImplementations = [
    DGStorageDictImpl,
]

DGStorage = DGStorageDictImpl

__all__ = [
    'DGStorage',
]
