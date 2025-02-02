from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import numpy as np


def construct_sampling_func(
    arg: Union[str, Callable[[float], float], 'SamplingFunc'],
    **kwargs: Any,
) -> 'SamplingFunc':
    if isinstance(arg, SamplingFunc):
        return arg

    if callable(arg):  # TODO: Double check this doesn't have side-effects
        return CustomSamplingFunc(arg, **kwargs)

    if arg in _sampling_funcs:
        sampling_func_klass = _sampling_funcs[arg]
        return sampling_func_klass(**kwargs)
    else:
        raise ValueError(
            f'Unknown sampling function type: {arg}, use one of: {list(_sampling_funcs.keys())}'
        )


class SamplingFunc(ABC):
    r"""Encodes a non-increasing sampling density on the positive reals."""

    @abstractmethod
    def __call__(self, dt: float) -> float:
        r"""Return the appropriate sampling weight for the input dt >= 0."""


class CustomSamplingFunc(SamplingFunc):
    r"""Sampling using an arbitrary callable to assign sampling weight.

    Args:
        func: Callable[[float], float]: The functional to apply.
    """

    def __init__(self, func: Callable[[float], float]) -> None:
        # TODO: Should probably use inspect and check the signature here
        self._func = func

    @property
    def func(self) -> Callable[[float], float]:
        return self._func

    def __call__(self, dt: float) -> float:
        return self._func(dt)


class UniformSamplingFunc(SamplingFunc):
    r"""Sampling using uniform weight with piecewise cutoff sampling probability.

    Args:
        max_time (Optional[float]): The cutoff time beyond which no neighborhood is sampled.
    """

    def __init__(self, max_time: Optional[float]) -> None:
        self._max_time = max_time

    @property
    def max_time(self) -> Optional[float]:
        return self._max_time

    def __call__(self, dt: float) -> float:
        if self.max_time is not None and dt > self.max_time:
            return 0
        return 1


class DecayedSamplingFunc(SamplingFunc):
    def __init__(self, decay_rate: float = 1, max_time: Optional[float] = None) -> None:
        self._decay_rate = decay_rate
        self._max_time = max_time

    @property
    def decay_rate(self) -> float:
        return self._decay_rate

    @property
    def max_time(self) -> Optional[float]:
        return self._max_time

    def __call__(self, dt: float) -> float:
        if self.max_time is not None and dt > self.max_time:
            return 0
        return np.e ** (-self.decay_rate * dt)


_sampling_funcs = {
    'uniform': UniformSamplingFunc,
    'decayed': DecayedSamplingFunc,
}
