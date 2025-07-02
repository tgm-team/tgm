from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict


@dataclass(frozen=True, slots=True)
class TimeDeltaDG:
    r"""Time granularity for temporal index in a dynamic graph."""

    unit: str
    value: int = 1

    _ORDERED: ClassVar[str] = 'r'
    _UNIT_TO_NANOS: ClassVar[Dict[str, int]] = {
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

    def __post_init__(self) -> None:
        if not isinstance(self.value, int) or self.value <= 0:
            raise ValueError(f'Value must be a positive integer, got: {self.value}')
        if self.is_ordered and self.value != 1:
            raise ValueError(f'Only value=1 is supported for ordered TimeDeltaDG')
        if not self.is_ordered and self.unit not in TimeDeltaDG._UNIT_TO_NANOS:
            raise ValueError(
                f'Unknown unit: {self.unit}, expected one of {[TimeDeltaDG._ORDERED] + list(TimeDeltaDG._UNIT_TO_NANOS.keys())}'
            )

    @property
    def is_ordered(self) -> bool:
        return self.unit == TimeDeltaDG._ORDERED

    def is_coarser_than(self, other: str | TimeDeltaDG) -> bool:
        r"""Return True iff self is strictly coarser than other.

        Raises:
            ValueError if either self or other is ordered.
        """
        return self.convert(other) > 1

    def convert(self, time_delta: str | TimeDeltaDG) -> float:
        r"""Convert the current granularity to the specified time_delta.

        Raises:
            ValueError if either self or other is ordered.
        """
        if isinstance(time_delta, str):
            time_delta = TimeDeltaDG(time_delta)
        if self.is_ordered or time_delta.is_ordered:
            raise ValueError('Cannot compare granularity for ordered TimeDeltaDG')
        return self._convert(time_delta)

    def _convert(self, other: TimeDeltaDG) -> float:
        self_nanos = TimeDeltaDG._UNIT_TO_NANOS[self.unit]
        other_nanos = TimeDeltaDG._UNIT_TO_NANOS[other.unit]

        invert_unit = False
        if self_nanos > other_nanos:
            # The unit ratio is safe to integer divide without precision error
            unit_ratio = self_nanos // other_nanos
        else:
            invert_unit = True
            unit_ratio = other_nanos // self_nanos

        value_ratio = self.value / other.value
        return value_ratio / unit_ratio if invert_unit else value_ratio * unit_ratio


TGB_TIME_DELTAS = {
    'tgbl-wiki': TimeDeltaDG('s'),
    'tgbl-subreddit': TimeDeltaDG('s'),
    'tgbl-lastfm': TimeDeltaDG('s'),
    'tgbl-review': TimeDeltaDG('s'),
    'tgbl-coin': TimeDeltaDG('s'),
    'tgbl-flight': TimeDeltaDG('s'),
    'tgbl-comment': TimeDeltaDG('s'),
    'tgbn-trade': TimeDeltaDG('Y'),
    'tgbn-genre': TimeDeltaDG('s'),
    'tgbn-reddit': TimeDeltaDG('s'),
    'tgbn-token': TimeDeltaDG('s'),
}
