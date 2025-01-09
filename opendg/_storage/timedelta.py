from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from opendg.typing import Event, TimeDeltaTG 


class TimeDeltaTG:
    r"""TimeGranularity class to represent time granularity in dynamic graph."""
    def __init__(self, interval_type: str, duration: Optional[int] = 1):
        r"""
        Args:
            interval_type (str): Type of time granularity. Possible values are 'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'.
            duration (int, optional): Duration of the specified time granularity. Defaults to 1.
        """
        self.time_dict = {"r": "ordered", "Y": "year", "M": "month", "W": "week", "D": "day", "h": "hour", "m": "minute", "s": "second", "ms": "millisecond", "us": "microsecond", "ns": "nanosecond"}
        self.non_convert = {"r": "ordered"}

        SECONDS_IN_A_YEAR = 31536000
        SECONDS_IN_A_MONTH = 2592000
        SECONDS_IN_A_WEEK = 604800
        SECONDS_IN_A_DAY = 86400
        SECONDS_IN_AN_HOUR = 3600
        SECONDS_IN_A_MINUTE = 60
        self.time_constant = {"Y": SECONDS_IN_A_YEAR, 
                              "M": SECONDS_IN_A_MONTH, 
                              "W": SECONDS_IN_A_WEEK, 
                              "D": SECONDS_IN_A_DAY, 
                              "h": SECONDS_IN_AN_HOUR, 
                              "m": SECONDS_IN_A_MINUTE, 
                              "s": 1, 
                              "ms": 1/1000, 
                              "us": 1/1000000, 
                              "ns": 1/1000000000,}


        if (self._is_valid_type(interval_type)):
            self.type = interval_type
        else:
            raise ValueError(f"Invalid time granularity type: {interval_type}")

        if (isinstance(duration, int)):
            self.dur = duration
        else:
            raise ValueError(f"TimeDeltaTG duration should be an integer, got {type(duration)}")
        
    @property
    def type(self) -> str:
        r"""The time granularity type"""
        return self.type

    def _is_valid_type(self) -> bool:
        r"""Check if the specified time granularity type is valid."""
        return (self.type in self.time_dict)

    def __str__(self):
        return f'time granularity is {self.type} : {self.time_dict[self.type]}'

    def __len__(self) -> int:
        r"""Returns the number of duration of the specified time granularity."""
        return self.dur
    
    def _convertFromStr(self, interval_type: str, duration: int) -> float:
        r"""Convert the time granularity to the specified time granularity type."""
        new_secs = duration * self.time_constant[interval_type]
        cur_secs = self.get_seconds()
        return new_secs / cur_secs
    
    def _convertFromDelta(self, td: TimeDeltaTG) -> float:
        r"""Convert the time granularity to the specified time granularity type."""
        new_secs = td.get_seconds()
        cur_secs = self.get_seconds()
        return new_secs / cur_secs
    
    def get_seconds(self) -> float:
        r"""Returns the time granularity in seconds."""
        return self.dur * self.time_constant[self.type]

    

    def convert(self, time_delta: Union[str, TimeDeltaTG]) -> int:
        r"""Convert the time granularity to the specified time granularity type, can be either a string or a TimeDeltaTG object.
        Args:
            time_delta (str or TimeDeltaTG): Type of time granularity. Possible values are 'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns' or a TimeDeltaTG object
        """
        if (isinstance(time_delta, str)):
            if (time_delta in self.non_convert):
                raise ValueError(f"Conversion not allowed for time granularity type {time_delta}")
            elif (self._is_valid_type(time_delta)):
                return self._convert(time_delta, 1)
            else:
                raise ValueError(f"Invalid time granularity type: {time_delta}")
        elif (isinstance(time_delta, TimeDeltaTG)):
            if (time_delta.type in self.non_convert):
                raise ValueError(f"Conversion not allowed for time granularity type {time_delta.type}")
            else:
                return self._convertFromDelta(time_delta)
        else:
            raise ValueError(f"Invalid time granularity type for conversion: {time_delta}")