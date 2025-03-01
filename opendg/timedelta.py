from typing import Optional, Union


class TimeDeltaDG:
    r"""TimeGranularity class to represent time granularity in dynamic graph.

    Args:
        unit (str): The temporal unit of time granularity.
        value (Optional[int]): The value of the specified time granularity. Defaults to 1.

    Note:
        Possible Values for the temporal unit are:

            'r':  Ordered granularity (no conversions allowed, TimeDelta value must be 1)
            'Y':  Year
            'M':  Month
            'D':  Day
            'h':  Hour
            'm':  Minute
            's':  Second
            'ms': Millisecond
            'us': Microsecond
            'ns': Nanosecond
    """

    def __init__(self, unit: str, value: Optional[int] = 1) -> None:
        self._ordered_unit = 'r'
        self._unit_nano_ratio = {
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

        if unit not in self._unit_nano_ratio and unit != self._ordered_unit:
            raise ValueError(f'Invalid time granularity unit: {unit}')

        if not (isinstance(value, int)) or value <= 0:
            raise ValueError(f'Value must be a positive integer, got: {value}')

        if unit == self._ordered_unit and value != 1:
            raise ValueError(f"Only value=1 is supported for 'ordered' TimeDelta unit")

        self._unit = unit
        self._value = value

    @property
    def unit(self) -> str:
        r"""The time granularity unit."""
        return self._unit

    @property
    def value(self) -> int:
        r"""The time granularity value."""
        return self._value

    @property
    def is_ordered(self) -> bool:
        r"""Whether or not the time granularity is 'ordered', in which case conversions are prohibited."""
        return self.unit == self._ordered_unit

    def convert(self, time_delta: Union[str, 'TimeDeltaDG']) -> float:
        r"""Convert the current granularity to the specified time_delta (either a unit string or a TimeDeltaDG object).

        Args:
            time_delta (str or TimeDeltaDG): unit of time granularity.

        Returns:
            float: conversion rate
        """
        if isinstance(time_delta, str):
            time_delta = TimeDeltaDG(time_delta)

        if time_delta.is_ordered or self.is_ordered:
            raise ValueError(
                f"Conversion {self}->{time_delta} for 'ordered' unit not allowed"
            )

        return self._convert_from_delta(time_delta)

    def __str__(self) -> str:
        if self.is_ordered:
            return f"TimeDeltaDG(unit='{self.unit}')"
        else:
            return f"TimeDeltaDG(unit='{self.unit}', value={self.value})"

    def _convert_from_delta(self, other: 'TimeDeltaDG') -> float:
        self_nanos = self._unit_nano_ratio[self.unit]
        other_nanos = self._unit_nano_ratio[other.unit]

        invert_unit = False
        if self_nanos > other_nanos:
            # The unit ratio is safe to integer divide without precision error
            unit_ratio = self_nanos // other_nanos
        else:
            invert_unit = True
            unit_ratio = other_nanos // self_nanos

        value_ratio = self.value / other.value
        return value_ratio / unit_ratio if invert_unit else value_ratio * unit_ratio
