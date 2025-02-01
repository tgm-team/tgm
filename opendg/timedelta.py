from typing import Optional, Union


class TimeDeltaDG:
    r"""TimeGranularity class to represent time granularity in dynamic graph."""

    def __init__(self, unit: str, value: Optional[int] = 1) -> None:
        r"""Args:
        unit (str): unit of time granularity. Possible values are 'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'r'.
        value (int, optional): value of the specified time granularity. Defaults to 1.
        """
        self._time_dict = {
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
        self._non_convert = {'r': 'ordered'}
        self._time_constant = {
            'Y': 1000 * 1000 * 1000 * 60 * 60 * 24 * 365,
            'M': 1000 * 1000 * 1000 * 60 * 60 * 24 * 30,
            'W': 1000 * 1000 * 1000 * 60 * 60 * 24 * 7,
            'D': 1000 * 1000 * 1000 * 60 * 60 * 24,
            'h': 1000 * 1000 * 1000 * 60 * 60,
            'm': 1000 * 1000 * 1000 * 60,
            's': 1000 * 1000 * 1000,
            'ms': 1000 * 1000,
            'us': 1000,
            'ns': 1,
        }

        if unit in self._time_dict:
            self._unit = unit
        else:
            raise ValueError(f'Invalid time granularity unit: {unit}')

        if isinstance(value, int) and value > 0:
            self._value = value
        else:
            raise ValueError(
                f'TimeDeltaDG value must be a positive integer, got {value}'
            )

        if unit in self._non_convert and value != 1:
            raise ValueError(
                'Only value=1 is supported for time granularity unit: {unit}'
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

        if (time_delta.unit in self._non_convert) or (
            self._unit in time_delta._non_convert
        ):
            raise ValueError(
                f'Conversion not allowed for time granularity unit {self._unit} : {time_delta.unit}'
            )

        return self._convert_from_delta(time_delta)

    def __str__(self) -> str:
        if self.unit in self._non_convert:
            return f"TimeDeltaDG(unit='{self.unit}')"
        return f"TimeDeltaDG(unit='{self.unit}', value={self.value})"

    def _convert_from_delta(self, other: 'TimeDeltaDG') -> float:
        self_nanos = self._time_constant[self.unit]
        other_nanos = self._time_constant[other.unit]

        invert_unit_ratio = False
        if self_nanos > other_nanos:
            # The unit ratio is safe to integer divide without precision error
            unit_ratio = (
                self._time_constant[self.unit] // self._time_constant[other.unit]
            )
        else:
            invert_unit_ratio = True
            unit_ratio = (
                self._time_constant[other.unit] // self._time_constant[self.unit]
            )

        value_ratio = self.value / other.value

        if invert_unit_ratio:
            return value_ratio / unit_ratio

        return unit_ratio * value_ratio
