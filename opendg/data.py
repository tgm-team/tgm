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


class BaseData:
    """Base class for all temporal graph data structures."""
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
    """Class for Continuous Time Dynamic Graphs, often represented as a stream of edges."""
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
        return CTDG(self.data.copy())

    def __deepcopy__(self, memo):
        return CTDG(self.data.copy())

    def __repr__(self) -> str:
        return f"CTDG({self.data})"
    
    ###########################################################################

    def aggregate_graph(self, start_time:int, end_time:int):
        """Aggregates the graph between start_time and end_time."""
        assert start_time <= end_time, "start_time should be less or equal to end_time"
        assert isinstance(start_time, int) and isinstance(end_time, int), "start_time and end_time should be integers"
        pass

    def to_events(self) -> Any:
        """Converts a continuous time dynamic graph to a list of events."""
        # Implement this method
        pass
    
    def to_snapshots(self) -> Any:
        """Converts a continuous time dynamic graph to a list of snapshots."""
        # Implement this method
        pass
    

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

    
    
