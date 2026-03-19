import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

parser = argparse.ArgumentParser(
    description=(
        'Parse TGM debug logs and extract structured metrics. '
        'Supports multiple reduction modes and optional comparison between log files.'
    ),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('log_file_path', type=str, help='Path to the log file')
parser.add_argument(
    'json_save_path',
    type=str,
    nargs='?',
    default=None,
    help='Optional path to save parsed logs. If not provided, defaults to the log file path with .json suffix',
)
parser.add_argument(
    '--compare',
    type=str,
    default=None,
    metavar='LOG_FILE',
    help='Path to a second log file to compare against the primary log file',
)
parser.add_argument(
    '--reduction',
    type=str,
    default='mean',
    choices=['mean', 'median', 'min', 'max', 'last', 'full'],
    help='Reduction strategy applied to repeated metric values. "full" returns all values with statistics.',
)
parser.add_argument(
    '--percentiles',
    type=float,
    nargs='+',
    default=None,
    metavar='P',
    help='Percentiles to include in output, e.g. --percentiles 25 50 75 95',
)


@dataclass
class MetricSummary:
    """Statistical summary for a single named metric across repeated log entries."""

    metric: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    values: list[float] = field(default_factory=list, repr=False)
    percentiles: dict[str, float] = field(default_factory=dict)

    def to_dict(self, include_values: bool = False) -> dict:
        d: dict = {
            'count': self.count,
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
        }
        if self.percentiles:
            d['percentiles'] = self.percentiles
        if include_values:
            d['values'] = self.values
        return d


def compute_summary(
    metric: str,
    values: list[float],
    percentile_points: list[float] | None = None,
) -> MetricSummary:
    """Compute a full statistical summary for a list of metric values.

    Args:
        metric: Name of the metric.
        values: Raw recorded values for this metric.
        percentile_points: Optional list of percentile levels (0-100) to compute.

    Returns:
        MetricSummary with count, mean, std, min, max, and optional percentiles.
    """
    n = len(values)
    mean = sum(values) / n
    std = statistics.stdev(values) if n > 1 else 0.0
    pct: dict[str, float] = {}
    if percentile_points:
        sorted_vals = sorted(values)
        for p in percentile_points:
            idx = (p / 100.0) * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            pct[f'p{p:g}'] = sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (
                idx - lo
            )
    return MetricSummary(
        metric=metric,
        count=n,
        mean=mean,
        std=std,
        min=min(values),
        max=max(values),
        values=values,
        percentiles=pct,
    )


def reduce_metrics(
    raw: dict[str, list[float]],
    reduction: str = 'mean',
    percentile_points: list[float] | None = None,
) -> dict:
    """Reduce raw metric lists to scalar(s) using the requested strategy.

    Args:
        raw: Mapping from metric name to list of recorded values.
        reduction: One of 'mean', 'median', 'min', 'max', 'last', or 'full'.
                   'full' returns a summary dict with count/mean/std/min/max.
        percentile_points: If provided, include these percentiles in the output
                           regardless of reduction mode.

    Returns:
        Dict mapping metric name to reduced value or summary dict.
    """
    REDUCERS: dict[str, Callable[[list[float]], float]] = {
        'mean': lambda vs: sum(vs) / len(vs),
        'median': statistics.median,
        'min': min,
        'max': max,
        'last': lambda vs: vs[-1],
    }

    if reduction == 'full' or percentile_points:
        include_values = reduction == 'full'
        return {
            metric: compute_summary(metric, values, percentile_points).to_dict(
                include_values=include_values
            )
            for metric, values in raw.items()
        }

    reducer = REDUCERS[reduction]
    return {metric: reducer(values) for metric, values in raw.items()}


def compare_logs(
    primary: dict[str, list[float]],
    secondary: dict[str, list[float]],
    reduction: str = 'mean',
    percentile_points: list[float] | None = None,
) -> dict:
    """Produce a side-by-side comparison of two parsed log files.

    For each metric present in either file, the output contains 'primary' and
    'secondary' reduced values. For shared scalar metrics, also reports 'delta'
    (primary minus secondary) and 'delta_pct' (relative change as a percentage).

    Args:
        primary: Raw metrics from the first (reference) log file.
        secondary: Raw metrics from the second log file.
        reduction: Reduction strategy passed to reduce_metrics.
        percentile_points: Optional percentiles to include in the summary.

    Returns:
        Dict mapping each metric name to a comparison entry.
    """
    primary_reduced = reduce_metrics(primary, reduction, percentile_points)
    secondary_reduced = reduce_metrics(secondary, reduction, percentile_points)

    all_metrics = sorted(set(primary_reduced) | set(secondary_reduced))
    result = {}
    for metric in all_metrics:
        entry: dict = {}
        if metric in primary_reduced:
            entry['primary'] = primary_reduced[metric]
        if metric in secondary_reduced:
            entry['secondary'] = secondary_reduced[metric]
        if metric in primary_reduced and metric in secondary_reduced:
            p_val = primary_reduced[metric]
            s_val = secondary_reduced[metric]
            if isinstance(p_val, (int, float)) and isinstance(s_val, (int, float)):
                delta = p_val - s_val
                entry['delta'] = delta
                entry['delta_pct'] = (delta / s_val * 100) if s_val != 0 else None
        result[metric] = entry
    return result


def collect_raw_metrics(log_file_path: Path) -> dict[str, list[float]]:
    """Read a log file and return all metric values grouped by metric name.

    Args:
        log_file_path: Path to the TGM debug log file.

    Returns:
        Mapping from metric name to list of recorded float values in file order.

    Raises:
        ValueError: If a JSON log entry has unexpected keys.
        json.JSONDecodeError: If a JSON-looking line cannot be parsed.
    """
    raw: dict[str, list[float]] = defaultdict(list)
    for line in log_file_path.read_text().splitlines():
        try:
            msg = extract_log_msg(line)
        except Exception as e:
            print(f'Failed to extract log msg from line: {line}')
            raise e

        if not (msg.startswith('{') and msg.endswith('}')):
            continue  # Skip non-JSON messages (could have some false positives)

        try:
            msg_dict = json.loads(msg)
        except json.JSONDecodeError as e:
            print(f'Failed to decode JSON for line: {line}')
            raise e

        msg_dict_keys = tuple(sorted(msg_dict.keys()))
        if msg_dict_keys != ('metric', 'value'):
            raise ValueError(
                'We only support parsing structured logs with keys: "metric" and "value", '
                f'found: {msg_dict_keys} in msg {msg} from line: {line}'
            )
        raw[msg_dict['metric']].append(msg_dict['value'])
    return raw


def parse_log_file(log_file_path: Path) -> dict:
    """Parse a log file and return mean-reduced metrics.

    Kept for backward compatibility. Prefer collect_raw_metrics + reduce_metrics
    for more control over the reduction strategy.
    """
    return reduce_metrics(collect_raw_metrics(log_file_path), reduction='mean')


def main(
    log_file_str: str,
    json_save_str: str | None,
    reduction: str = 'mean',
    compare_path: str | None = None,
    percentile_points: list[float] | None = None,
) -> None:
    log_file_path = Path(log_file_str)
    print(f'Parsing log file: {log_file_path}')
    raw_metrics = collect_raw_metrics(log_file_path)

    if compare_path is not None:
        compare_file_path = Path(compare_path)
        print(f'Comparing against: {compare_file_path}')
        raw_secondary = collect_raw_metrics(compare_file_path)
        output = compare_logs(raw_metrics, raw_secondary, reduction, percentile_points)
    else:
        output = reduce_metrics(raw_metrics, reduction, percentile_points)

    if json_save_str is None:
        json_save_path = log_file_path.with_suffix('.json')
    else:
        json_save_path = Path(json_save_str)
    json_save_path.parent.mkdir(parents=True, exist_ok=True)
    json_save_path.write_text(json.dumps(output, indent=2))
    print(f'Saved parsed metrics to {json_save_path}')


def extract_log_msg(line: str) -> str:
    """Extract log message by skipping metadata.

    Logs have the form:
    [YYYY-MM-DD hh:mm:ss] LOG_NAME - LOG_LEVEL - [PROCESS THREAD FILE] LOG_MSG
    """
    return line[line.find(']', line.find(']') + 1) + 2 :]  # + 2 to skip '] '


if __name__ == '__main__':
    args = parser.parse_args()
    main(
        args.log_file_path,
        args.json_save_path,
        reduction=args.reduction,
        compare_path=args.compare,
        percentile_points=args.percentiles,
    )
