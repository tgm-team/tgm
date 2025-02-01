from typing import Optional, Union


class TimeDeltaDG:
    r"""TimeGranularity class to represent time granularity in dynamic graph."""

    def __init__(self, unit: str, value: Optional[int] = 1):
        r"""Args:
        unit (str): unit of time granularity. Possible values are 'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'r'.
        value (int, optional): value of the specified time granularity. Defaults to 1.
        """
        self.time_dict = {
            'r': 'ordered',
            'Y': 'year',
            'M': 'month',
            'W': 'week',
            'D': 'day',
            'h': 'hour',
            'm': 'minute',
            's': 'second',
            'ms': 'millisecond',
            'us': 'microsecond',
            'ns': 'nanosecond',
        }
        self.non_convert = {'r': 'ordered'}

        SECONDS_IN_A_YEAR = 31536000  # 365 days
        SECONDS_IN_A_MONTH = 2592000
        SECONDS_IN_A_WEEK = 604800
        SECONDS_IN_A_DAY = 86400
        SECONDS_IN_AN_HOUR = 3600
        SECONDS_IN_A_MINUTE = 60
        self.time_constant = {
            'Y': SECONDS_IN_A_YEAR,
            'M': SECONDS_IN_A_MONTH,
            'W': SECONDS_IN_A_WEEK,
            'D': SECONDS_IN_A_DAY,
            'h': SECONDS_IN_AN_HOUR,
            'm': SECONDS_IN_A_MINUTE,
            's': 1,
            'ms': 1 / 1000,
            'us': 1 / 1_000_000,
            'ns': 1 / 1_000_000_000,
        }

        if self._is_valid_unit(unit):
            self._unit = unit
        else:
            raise ValueError(f'Invalid time granularity unit: {unit}')

        if isinstance(value, int):
            self._value = value
        else:
            raise ValueError(
                f'TimeDeltaDG value should be an integer, got {str(value)}'
            )

    @property
    def unit(self) -> str:
        r"""The time granularity unit."""
        return self._unit

    @property
    def value(self) -> int:
        r"""The time granularity value."""
        return self._value

    def convert(self, time_delta: Union[str, 'TimeDeltaDG']) -> float:
        r"""Convert the time granularity to the specified time granularity unit, can be either a string or a TimeDeltaDG object.

        Args:
            time_delta (str or TimeDeltaDG): unit of time granularity. Possible values are 'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns' or a TimeDeltaDG object
        Returns:
            float: conversion rate
        """
        if isinstance(time_delta, str):
            time_delta = TimeDeltaDG(time_delta)

        if not isinstance(time_delta, TimeDeltaDG):
            raise ValueError(
                f'Invalid time granularity unit for conversion: {time_delta}'
            )

        if (time_delta.unit in self.non_convert) or (
            self._unit in time_delta.non_convert
        ):
            raise ValueError(
                f'Conversion not allowed for time granularity unit {self._unit} : {time_delta.unit}'
            )

        return self._convert_from_delta(time_delta)

    def get_seconds(self) -> float:
        r"""Returns the time granularity in seconds."""
        return self._value * self.time_constant[self._unit]

    def _is_valid_unit(self, unit: str) -> bool:
        r"""Check if the specified time granularity unit is valid."""
        return unit in self.time_dict

    def __str__(self) -> str:
        return f'time granularity is {self._unit} : {self.time_dict[self._unit]}'

    def __len__(self) -> int:
        r"""Returns the number of value of the specified time granularity."""
        return self._value

    def _convert_from_delta(self, td: 'TimeDeltaDG') -> float:
        r"""Convert the time granularity to the specified time granularity unit."""
        new_secs = td.get_seconds()
        cur_secs = self.get_seconds()
        return cur_secs / new_secs
