from tgm.core._storage.backends.array_backend import DGStorageArrayBackend

DGStorageBackends = {
    'ArrayBackend': DGStorageArrayBackend,
}

DGStorage = DGStorageArrayBackend
