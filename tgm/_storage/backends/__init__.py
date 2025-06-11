from tgm._storage.backends.array_backend import DGStorageArrayBackend

DGStorageBackends = {
    'ArrayBackend': DGStorageArrayBackend,
}

DGStorage = DGStorageArrayBackend
