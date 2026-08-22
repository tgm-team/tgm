from typing import Type
from tgm.core._storage.base import DGStorageBase
from tgm.core._storage.backends.array_backend import DGStorageArrayBackend

DGStorageBackends = {
    'ArrayBackend': DGStorageArrayBackend,
}

DGStorage: Type[DGStorageBase] = DGStorageArrayBackend
