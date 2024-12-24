from opendg._storage.backends.dict_backend import DGStorageDictBackend

DGStorageBackends = {
    'DictionaryBackend': DGStorageDictBackend,
}

DGStorage = DGStorageDictBackend
