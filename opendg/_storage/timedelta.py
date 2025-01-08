from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from opendg.typing import Event 


class TimeGranularity:
    r"""TimeGranularity class to represent time granularity in dynamic graph."""
    def __init__(self, interval_type: str, n: Optional[int] = 1):
        r"""
        Args:
            interval_type (str): Type of time granularity. Possible values are 's', 'm', 'h', 'd', 'w', 'M', 'y'.
            n (int, optional): Number of duration of the specified time granularity. Defaults to 1.
        """
        self.type = interval_type
        self.n = n

    def _is_valid_type(self) -> bool:
        r"""Check if the specified time granularity type is valid."""
        return self.type in ['s', 'm', 'h', 'd', 'w', 'M', 'y']

    def __str__(self):
        return f'time granularity is {self.value} {self.type}'

    def __len__(self) -> int:
        r"""Returns the number of duration of the specified time granularity."""
        return self.n