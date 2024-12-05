from typing import Any, Dict, Optional


class BaseData:
    """Base class for all temporal graph data structures."""

    def __getattr__(self, key: str) -> Any:
        raise NotImplementedError

    def __setattr__(self, key: str, value: Any) -> None:
        raise NotImplementedError

    def __delattr__(self, key: str) -> None:
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        raise NotImplementedError

    def __setitem__(self, key: str, value: Any) -> None:
        raise NotImplementedError

    def __delitem__(self, key: str) -> None:
        raise NotImplementedError

    def __copy__(self) -> BaseData:
        raise NotImplementedError

    def __deepcopy__(self, memo: dict) -> BaseData:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def aggregate_graph(self, start_time: int, end_time: int) -> Any:
        """Aggregates the graph between start_time and end_time."""
        raise NotImplementedError

    @property
    def start_time(self) -> Optional[int]:
        """Returns the start time of the temporal graph.
        Could be a unix timestamp or a snapshot id.
        """

    @property
    def end_time(self) -> Optional[int]:
        """Returns the end time of the temporal graph
        Could be a unix timestamp or a snapshot id.
        """

    @property
    def num_nodes(self) -> int:
        """Returns the total number of unique nodes encountered over the entire
        temporal graph.
        """
        return -1

    @property
    def num_edges(self) -> int:
        """Returns the total number of temporal edges encountered over the entire
        temporal graph.
        """
        return -1

    @property
    def num_timestamps(self) -> int:
        """Returns the total number of unique timestamps encountered over the entire
        temporal graph.
        """
        return -1


class CTDG(BaseData):
    """Class for Continuous Time Dynamic Graphs, often represented as a stream of
    edges.
    """

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def __getattr__(self, key: str) -> Any:
        return self.data[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __delattr__(self, key: str) -> None:
        del self.data[key]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __copy__(self) -> 'CTDG':
        return CTDG(self.data.copy())

    def __deepcopy__(self, memo: dict) -> 'CTDG':
        return CTDG(self.data.copy())

    def __repr__(self) -> str:
        return f'CTDG({self.data})'

    ###########################################################################

    def aggregate_graph(self, start_time: int, end_time: int) -> Any:
        """Aggregates the graph between start_time and end_time."""
        assert start_time <= end_time, 'start_time should be less or equal to end_time'
        assert isinstance(start_time, int) and isinstance(
            end_time, int
        ), 'start_time and end_time should be integers'

    def to_events(self) -> Any:
        """Converts a continuous time dynamic graph to a list of events."""
        # Implement this method

    def to_snapshots(self) -> Any:
        """Converts a continuous time dynamic graph to a list of snapshots."""
        # Implement this method


class DTDG(BaseData):
    """Class for Discrete Time Dynamic Graphs, often represented as a sequence of
    graph snapshots.
    """

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def __getattr__(self, key: str) -> Any:
        return self.data[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __delattr__(self, key: str) -> None:
        del self.data[key]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __copy__(self) -> 'DTDG':
        return DTDG(self.data.copy())

    def __deepcopy__(self, memo: dict) -> 'DTDG':
        return DTDG(self.data.copy())

    def __repr__(self) -> str:
        return f'DTDG({self.data})'

    ###########################################################################

    def aggregate_graph(self, start_time: int, end_time: int) -> Any:
        """Aggregates the graph between start_time and end_time."""
        assert start_time <= end_time, 'start_time should be less or equal to end_time'
        assert isinstance(start_time, int) and isinstance(
            end_time, int
        ), 'start_time and end_time should be integers'

    def to_events(self) -> Any:
        """Converts a discrete time dynamic graph to a list of events."""
        # Implement this method

    def to_snapshots(self) -> Any:
        """Converts a discrete time dynamic graph to a list of snapshots."""
        # Implement this method
