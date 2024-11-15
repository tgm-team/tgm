
import copy
from collections.abc import MutableMapping
from collections import defaultdict, namedtuple
import torch
from torch import Tensor
from typing_extensions import Self
from opendg.data.view import KeysView, ValuesView, ItemsView


from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
    overload,
)

class BaseTGStore(MutableMapping):
    r"""A base storage class for temporal graph objects.
    This class wraps a Python dictionary and extends it as follows:
    1. It allows attribute assignments, e.g.:
       `storage.x = ...` in addition to `storage['x'] = ...`
    2. It allows private attributes that are not exposed to the user, e.g.:
       `storage._{key} = ...` and accessible via `storage._{key}`
    3. It allows iterating over only a subset of keys, e.g.:
       `storage.values('x', 'y')` or `storage.items('x', 'y')
    4. It adds mandatory property such as timestamp
    (!TODO)5. It adds additional PyTorch Tensor functionality, e.g.:
       `storage.cpu()`, `storage.cuda()` or `storage.share_memory_()`.

    inspired by PyG BaseStorage class
    https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/storage.py
    # """
    def __init__(
        self,
        timestamp: int = None,
        _mapping: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._timestamp = timestamp
        self._mapping: Dict[str, Any] = {}
        for key, value in (_mapping or {}).items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def _key(self) -> Any:
        return None
    
    @property
    def timestamp(self) -> int:
        return self._timestamp


    def _pop_cache(self, key: str) -> None:
        for cache in getattr(self, '_cached_attr', {}).values():
            cache.discard(key)

    def __len__(self) -> int:
        return len(self._mapping)

    def __getattr__(self, key: str) -> Any:
        if key == '_mapping':
            self._mapping = {}
            return self._mapping
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            ) from None

    def __setattr__(self, key: str, value: Any) -> None:
        propobj = getattr(self.__class__, key, None)
        if propobj is not None and getattr(propobj, 'fset', None) is not None:
            propobj.fset(self, value)
        elif key[:1] == '_':
            self.__dict__[key] = value
        else:
            self[key] = value

    def __delattr__(self, key: str) -> None:
        if key[:1] == '_':
            del self.__dict__[key]
        else:
            del self[key]

    def __getitem__(self, key: str) -> Any:
        return self._mapping[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._pop_cache(key)
        if value is None and key in self._mapping:
            del self._mapping[key]
        elif value is not None:
            self._mapping[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self._mapping:
            self._pop_cache(key)
            del self._mapping[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._mapping)

    def __copy__(self) -> Self:
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            if key != '_cached_attr':
                out.__dict__[key] = value
        out._mapping = copy.copy(out._mapping)
        return out

    def __deepcopy__(self, memo: Optional[Dict[int, Any]]) -> Self:
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out._mapping = copy.deepcopy(out._mapping, memo)
        return out
    
    def __repr__(self) -> str:
        return repr(self._mapping)
    
    # Allow iterating over subsets ############################################

    # In contrast to standard `keys()`, `values()` and `items()` functions of
    # Python dictionaries, we allow to only iterate over a subset of items
    # denoted by a list of keys `args`.
    # This is especially useful for adding PyTorch Tensor functionality to the
    # storage object, e.g., in case we only want to transfer a subset of keys
    # to the GPU (i.e. the ones that are relevant to the deep learning model).

    def keys(self, *args: str) -> KeysView:  # type: ignore
        return KeysView(self._mapping, *args)

    def values(self, *args: str) -> ValuesView:  # type: ignore
        return ValuesView(self._mapping, *args)

    def items(self, *args: str) -> ItemsView:  # type: ignore
        return ItemsView(self._mapping, *args)
    
    
    # Additional functionality ################################################

    def get(self, key: str, value: Optional[Any] = None) -> Any:
        return self._mapping.get(key, value)
    

    def to_dict(self) -> Dict[str, Any]:
        r"""Returns a dictionary of stored key/value pairs."""
        out_dict = copy.copy(self._mapping)
        return out_dict
    
    def to_namedtuple(self) -> NamedTuple:
        r"""Returns a :obj:`NamedTuple` of stored key/value pairs."""
        field_names = list(self.keys())
        typename = f'{self.__class__.__name__}Tuple'
        StorageTuple = namedtuple(typename, field_names)  # type: ignore
        return StorageTuple(*[self[key] for key in field_names])
    
    def clone(self, *args: str) -> Self:
        r"""Performs a deep-copy of the object."""
        return copy.deepcopy(self)
    
    def contiguous(self, *args: str) -> Self:
        r"""Ensures a contiguous memory layout, either for all attributes or
        only the ones given in :obj:`*args`.
        """
        return self.apply(lambda x: x.contiguous(), *args)



class EventStore(BaseTGStore):
    r"""A storage class for event information at a given timestamp, 
    can contain edge or(and) node information"""
    def __init__(
        self,
        timestamp: int = None,
        edges: Optional[Tensor] = None,
        node_feats: Optional[Dict[int, Tensor]] = None,
        edge_feats: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._mapping: Dict[str, Any] = {}
        self._mapping['edges'] = edges
        self._mapping['node_feats'] = node_feats
        self._mapping['edge_feats'] = edge_feats

        if (edge_feats is not None):
            assert edges is not None, "Edge features provided but no edges are provided"
            assert edges.shape[1] == edge_feats.shape[0], "Number of edges and edge features do not match"
        
        #! Continue debug here
        # for key, value in self._mapping.items():
        #     setattr(self, key, value)
        # for key, value in kwargs.items():
        #     setattr(self, key, value)

    @property
    def num_edges(self) -> Optional[int]:
        r"""Returns the number of edges in the event."""
        return self._mapping['edges'].shape[1]
    
    @property
    def num_updated_nodes(self) -> Optional[int]:
        r"""Returns the number of nodes involved in node feature updates."""
        return len(self._mapping['node_feats'])
    
    #! TODO add # of num calculation from edge_index