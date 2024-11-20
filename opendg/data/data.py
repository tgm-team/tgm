from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
from opendg.data.storage import EventStore
from collections.abc import MutableMapping
import csv
import torch
import os.path
from abc import ABC


class BaseData(ABC):
    r"""Base class for all temporal graph data structures."""
    def __getattr__(self, key: str) -> Any:
        raise NotImplementedError

    def __setattr__(self, key: str, value: Any):
        raise NotImplementedError

    def __delattr__(self, key: str):
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        raise NotImplementedError

    def __setitem__(self, key: str, value: Any):
        raise NotImplementedError

    def __delitem__(self, key: str):
        raise NotImplementedError
    
    def __copy__(self):
        raise NotImplementedError

    def __deepcopy__(self, memo):
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError
    
    def aggregate_graph(self, start_time:int, end_time:int) -> Any:
        """Aggregates the graph between start_time and end_time."""
        raise NotImplementedError
    
    @property
    def start_time(self) -> Optional[int]:
        r"""Returns the start time of the temporal graph, could be a unix timestamp or a snapshot id."""
        pass
    
    @property
    def end_time(self) -> Optional[int]:
        r"""Returns the end time of the temporal graph, could be a unix timestamp or a snapshot id."""
        pass

    @property
    def num_nodes(self) -> int:
        r"""Returns the total number of unique nodes encountered over the entire temporal graph."""
        pass

    @property
    def num_edges(self) -> int:
        r"""Returns the total number of temporal edges encountered over the entire temporal graph."""
        pass

    @property
    def num_timestamps(self) -> int:
        r"""Returns the total number of unique timestamps encountered over the entire temporal graph."""
        pass


class CTDG(BaseData):
    r"""A data object describing a Continuous Time Dynamic Graph, often represented by a stream of edges
    The data object can hold link-level, node-level and graph-level attributes, inherents the BaseData class.
    In general, :class:`~opendg.data.CTDG` tries to mimic the
    behavior of a regular :python:`Python` dictionary with the key being the timestamps in TG.
    In addition, it provides useful functionality for analyzing temproal graph
    structures, sorting / slicing / subindexing temporal graph via timestamps 
    and provides basic PyTorch tensor functionalities.

    .. code-block:: python

        from opendg.data import CTDG

        # create a CTDG object either from a csv file or from a dictionary
        data = CTDG(file_path="path/to/ctdg.csv")
        # data = CTDG(data=edge_dict)

        # Analyzing the graph structure:
        data.num_nodes
        >>> 23

        data.is_directed()
        >>> False

        # # PyTorch tensor functionality:
        # data = data.pin_memory()
        # data = data.to('cuda:0', non_blocking=True)

    Args:
        file_path (str, optional): The path to the csv file to be loaded. (default: :obj:`None`)
        event_dict (Dict[str, Any], optional): The dictionary of temporal edges to be stored in the CTDG. (default: :obj:`None`)

        **kwargs (optional): Additional attributes.
    """
    def __init__(self, file_path: str=None, event_dict: Dict[str, Any]=None, graph_store: Dict[str,Any]=None, device='cpu'):
        r"""Initializes a :class:`~opendg.data.CTDG` object, must either initialize with a dictionary or a file path."""
        assert file_path is not None or data is not None or graph_store is not None, "Either file_path or data or copy should be provided"

        self.__dict__['_store'] = {}

        if graph_store is not None:
            self._store = graph_store
            return

        if file_path is not None:
            edge_dict = self._load_csv(file_path)
        else:
            edge_dict = data

        #* check if edge_dict is sorted
        t_list = list(edge_dict.keys())
        if (not all(t_list[i] <= t_list[i+1] for i in range(len(t_list) - 1))):
            edge_dict = dict(sorted(edge_dict.items()))

        #! add logic to insert node events too
        self.__dict__['_store']['events'] = self._to_eventStore(edge_dict)
        ts_list = list(edge_dict.keys())
        self.__dict__['_store']['min_time'] = ts_list[0]
        self.__dict__['_store']['max_time'] = ts_list[-1]
        self.__dict__['_store']['timestamps'] = ts_list
        

    def _to_eventStore(self, edge_dict:Dict[int,list]) -> Dict[int, EventStore]:
        r"""Converts a dictionary of edges to a dictionary of events."""
        event_dict = {}
        for t, edges in edge_dict.items():
            edge_index = torch.tensor(edges)
            edge_index = torch.transpose(edge_index, 0, 1)
            event_dict[t] = EventStore(timestamp=t,edges=edge_index,node_feats=None)
        return event_dict

    def _load_csv(self, file_path: str) -> Dict[str, Any]:
        r"""Loads a continuous time dynamic graph from a csv file."""
        os.path.exists(file_path), f"File {file_path} does not exist"
        edge_dict = {}
        with open('test_ctdg.csv', mode='r') as ctdg_file:
            ctdg_reader = csv.reader(ctdg_file)
            header = next(ctdg_reader)
            assert len(header) >= 3, "The csv file should have at least 3 columns (source, destination, timestamp)"
            #* infer the headers
            header_dict = {header[i]:i for i in range(len(header))}
            src_idx = header_dict.get('source',None)
            dst_idx = header_dict.get('destination',None)
            time_idx = header_dict.get('timestamp',None)

            for edge in ctdg_reader:
                u = int(edge[src_idx])
                v = int(edge[dst_idx])
                t = int(edge[time_idx])
                if t not in edge_dict:
                    edge_dict[t] = [(u,v)]
                else:
                    edge_dict[t].append((u,v))
        return edge_dict



    def __getattr__(self, key: str) -> Any:
        return getattr(self._store, key)
    
    def __setattr__(self, key: str, value: Any):
        setattr(self._store, key, value)

    def __delattr__(self, key: str):
        delattr(self._store, key)

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __setitem__(self, key: str, value: Any):
        self._store[key] = value

    def __delitem__(self, key: str):
        if key in self._store:
            del self._store[key]

    def __copy__(self):
        return CTDG(self._store.copy())

    def __deepcopy__(self, memo):
        return CTDG(self._store.copy())

    def __repr__(self) -> str:
        return f"CTDG({self._store.keys()})"
    
    ###########################################################################

    def aggregate_graph(self, start_time:int=None, end_time:int=None) -> torch.Tensor:
        r"""Aggregates the graph between start_time and end_time, inclusive.
        Args:
            start_time (int, optional): The start time of the aggregation. (default: :obj:`None`)
            end_time (int, optional): The end time of the aggregation. (default: :obj:`None`)
        Returns:
            :class:`torch.Tensor`: The aggregated edge index tensor."""
        if (start_time is None):
            start_time = self.min_time
        if (end_time is None):
            end_time = self.max_time
        assert start_time <= end_time, "start_time should be less or equal to end_time"
        assert isinstance(start_time, int) and isinstance(end_time, int), "start_time and end_time should be integers"
        out_index = self._store['events'][start_time].edges
        for t in self._store['events'].keys():
            if (t > start_time and t <= end_time):
                out_index = torch.cat((out_index, self._store['events'][t].edges), dim=1)
        return out_index

    def to_events(self) -> Any:
        r"""Converts a continuous time dynamic graph to a list of edge events.
        Returns: a tuple of timestamps and edge index tensors."""
        # Implement this method
        edge_index = self._store['events'][self.min_time].edges
        timestamps = [self.min_time]*(edge_index.shape[1])
        for t in self._store['events'].keys():
            if (t> self.min_time):
                edge_index = torch.cat((edge_index, self._store['events'][t].edges), dim=1)
                timestamps.extend([t]*(self._store['events'][t].edges.shape[1]))
        timestamps = torch.tensor(timestamps)
        return timestamps, edge_index
    
    def to_snapshots(self) -> Any:
        """Converts a continuous time dynamic graph to a list of snapshots."""
        # Implement this method
        pass

    def load_csv(self, file_path: str) -> Any:
        """Loads a continuous time dynamic graph from a csv file."""
        # Implement this method
        edge_dict = self._load_csv(file_path)
        self.__dict__['_store']['events'] = self._to_eventStore(edge_dict)

    @property
    def num_edges(self) -> Optional[int]:
        r"""Returns the number of edges in the temporal graph."""
        return sum([event.num_edges for event in self._store['events'].values()])
    
    @property
    def min_time(self) -> int:
        r"""Returns the minimum timestamp in the temporal graph."""
        return self.__dict__['_store']['min_time']
    
    @property
    def max_time(self) -> int:
        r"""Returns the maximum timestamp in the temporal graph."""
        return self.__dict__['_store']['max_time']
    
    @property
    def timestamps(self) -> List[int]:
        r"""Returns the list of timestamps in the temporal graph."""
        return self.__dict__['_store']['timestamps']
    
        
    

class DTDG(BaseData):
    """Class for Discrete Time Dynamic Graphs, often represented as a sequence of graph snapshots."""
    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def __getattr__(self, key: str) -> Any:
        return self.data[key]

    def __setattr__(self, key: str, value: Any):
        self.data[key] = value

    def __delattr__(self, key: str):
        del self.data[key]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any):
        self.data[key] = value

    def __delitem__(self, key: str):
        del self.data[key]

    def __copy__(self):
        return DTDG(self.data.copy())

    def __deepcopy__(self, memo):
        return DTDG(self.data.copy())

    def __repr__(self) -> str:
        return f"DTDG({self.data})"
    
    ###########################################################################

    def aggregate_graph(self, start_time:int, end_time:int):
        """Aggregates the graph between start_time and end_time."""
        assert start_time <= end_time, "start_time should be less or equal to end_time"
        assert isinstance(start_time, int) and isinstance(end_time, int), "start_time and end_time should be integers"
        pass
    
    def to_events(self) -> Any:
        """Converts a discrete time dynamic graph to a list of events."""
        # Implement this method
        pass

    def to_snapshots(self) -> Any:
        """Converts a discrete time dynamic graph to a list of snapshots."""
        # Implement this method
        pass

    def load_csv(self, file_path: str) -> Any:
        """Loads a discrete time dynamic graph from a csv file."""
        # Implement this method
        pass
    

    
    
