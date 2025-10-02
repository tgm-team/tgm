import contextlib
import cProfile
import pstats
from typing import Optional, Tuple

from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


class Profiling(contextlib.ContextDecorator):
    def __init__(self, filename: Optional[str] = None, frac: float = 0.3) -> None:
        self.filename = filename
        self.frac = frac

    def __enter__(self) -> 'Profiling':
        self.pr = cProfile.Profile()
        self.pr.enable()
        return self

    def __exit__(self, *_: None) -> None:
        self.pr.disable()
        if self.filename:
            self.pr.dump_stats(self.filename)

        stats = pstats.Stats(self.pr).strip_dirs().sort_stats('cumtime')
        for fcn in stats.fcn_list[0 : int(len(stats.fcn_list) * self.frac)]:  # type: ignore
            _, n_calls, tottime, cumtime, callers = stats.stats[fcn]  # type: ignore
            scallers = sorted(callers.items(), key=lambda x: -x[1][2])
            s = f'n:{n_calls:8d}  tm:{tottime * 1e3:7.2f}ms  tot:{cumtime * 1e3:7.2f}ms'
            s += _color(self._format_fcn(fcn).ljust(50), 'yellow')
            if scallers:
                perc = scallers[0][1][2] / tottime * 100
                caller = f'{self._format_fcn(scallers[0][0])}'
                s += _color(f'<- {perc:3.0f}% {caller}', 'BLACK')
            print(s)
            logger.info(s)

    @staticmethod
    def _format_fcn(fcn: Tuple[str, ...]) -> str:
        return f'{fcn[0]}:{fcn[1]}:{fcn[2]}'


def _color(s: str, color: str, background: bool = False) -> str:
    # Adapted from: https://github.com/tinygrad/tinygrad/blob/master/tinygrad/helpers.py
    colors = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
    return f'\u001b[{10 * background + 60 * (color.upper() == color) + 30 + colors.index(color.lower())}m{s}\u001b[0m'
