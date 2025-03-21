import contextlib
import cProfile
import pstats
import time
from typing import Any, Dict, Optional, Tuple

import torch


class Usage(contextlib.ContextDecorator):
    def __init__(self, gpu: bool = True, prefix: Optional[str] = None) -> None:
        self.prefix = '' if prefix is None else prefix.ljust(20)
        self.gpu = gpu

    def __enter__(self) -> None:
        if self.gpu:
            self.base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        self.st = time.perf_counter_ns()

    def __exit__(self, *_: None) -> None:
        duration = time.perf_counter_ns() - self.st
        s = f'{self.prefix}{duration * 1e-6:6.2f} ms'
        if self.gpu:
            gpu = torch.cuda.max_memory_allocated() - self.base
            s += f' ms ({gpu * 1e-9:2.2f} GB GPU)'
        print(s)


# Adapted from: https://github.com/tinygrad/tinygrad/blob/master/tinygrad/helpers.py
class Profiling(contextlib.ContextDecorator):
    def __init__(self, filename: Optional[str] = None, frac: float = 0.3) -> None:
        self.filename = filename
        self.frac = frac

    def __enter__(self) -> None:
        self.pr = cProfile.Profile()
        self.pr.enable()

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

    @staticmethod
    def _format_fcn(fcn: Tuple[str, ...]) -> str:
        return f'{fcn[0]}:{fcn[1]}:{fcn[2]}'


def compare_usage(
    f1: Dict[str, Any], f2: Dict[str, Any], prefix: Optional[str] = None
) -> None:
    def _usage(spec: Dict[str, Any]) -> Tuple[float, float]:
        f, args, kwargs = spec['func'], spec.get('args', []), spec.get('kwargs', {})
        base_mem = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        st = time.perf_counter_ns()
        f(*args, **kwargs)
        duration = 1e-6 * (time.perf_counter_ns() - st)
        gpu = 1e-9 * (torch.cuda.max_memory_allocated() - base_mem)
        return duration, gpu

    def _rank(x: float) -> str:
        c = 'green' if x < 0.75 else 'red' if x > 1.15 else 'yellow'
        return _color(f'{x:7.2f}x', c)

    prefix = '' if prefix is None else prefix.ljust(20)
    f1_time, f1_gpu, f2_time, f2_gpu = *_usage(f1), *_usage(f2)
    f1_res = f'[{f1["func"].__name__.rjust(20)}] {f1_time:6.2f} ms ({f1_gpu} GB GPU)'
    f2_res = f'[{f2["func"].__name__.rjust(20)}] {f2_time:6.2f} ms ({f2_gpu} GB GPU)'
    diff = f'{_rank(f1_time / f2_time)} {"faster" if f1_time < f2_time else "slower"}'
    print(f'\r{prefix} {f1_res} \t{f2_res} \t{diff}')


def _color(s: str, color: str, background: bool = False) -> str:
    # Adapted from: https://github.com/tinygrad/tinygrad/blob/master/tinygrad/helpers.py
    colors = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
    return f'\u001b[{10 * background + 60 * (color.upper() == color) + 30 + colors.index(color.lower())}m{s}\u001b[0m'
