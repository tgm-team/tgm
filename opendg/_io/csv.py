import csv
import pathlib
from typing import List, Optional

import torch

from opendg.events import EdgeEvent, Event


def read_csv(
    file_path: str | pathlib.Path,
    src_col: str,
    dst_col: str,
    time_col: str,
    edge_feature_col: Optional[List[str]] = None,
) -> List[Event]:
    # TODO: Node Events not supported
    file_path = str(file_path) if isinstance(file_path, pathlib.Path) else file_path
    events: List[Event] = []
    with open(file_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row[src_col])
            dst = int(row[dst_col])
            t = int(row[time_col])

            if edge_feature_col is None:
                features = None
            else:
                msg_list = [float(row[feature_col]) for feature_col in edge_feature_col]
                features = torch.tensor(msg_list)

            event = EdgeEvent(t=t, src=src, dst=dst, features=features)
            events.append(event)
    return events
