import csv
from typing import Any, Dict, List, Optional

import torch

from opendg.events import EdgeEvent, Event, NodeEvent


def read_csv(
    file_path: str,
    src_col: str,
    dst_col: str,
    time_col: str,
    edge_feature_col: Optional[List[str]] = None,
) -> List[Event]:
    # TODO: Node Events not supported
    events: List[Event] = []
    with open(file_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row[src_col])
            dst = int(row[dst_col])
            t = int(row[time_col])

            if edge_feature_col is None:
                msg = None
            else:
                msg_list = [float(row[feature_col]) for feature_col in edge_feature_col]
                msg = torch.tensor(msg_list)

            event = EdgeEvent(t=t, src=src, dst=dst, msg=msg)
            events.append(event)
    return events


def write_csv(
    events: List[Event],
    file_path: str,
    src_col: str,
    dst_col: str,
    time_col: str,
    edge_feature_col: Optional[List[str]] = None,
) -> None:
    with open(file_path, 'w', newline='') as f:
        fieldnames = [src_col, dst_col, time_col]
        if edge_feature_col is not None:
            fieldnames += edge_feature_col
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for event in events:
            if isinstance(event, NodeEvent):
                raise NotImplementedError('Node feature IO not supported')

            assert isinstance(event, EdgeEvent)

            row: Dict[str, Any] = {
                src_col: event.src,
                dst_col: event.dst,
                time_col: event.t,
            }

            if event.msg is not None:
                if edge_feature_col is None:
                    raise ValueError(
                        'No feature column provided but events had features'
                    )

                if len(event.msg.shape) > 1:
                    raise ValueError('Multi-dimensional features not supported')

                if len(event.msg) != len(edge_feature_col):
                    raise ValueError(
                        f'Got {len(event.msg)}-dimensional feature tensor but only '
                        f'specified {len(edge_feature_col)} feature column names.'
                    )

                features_list = event.msg.tolist()
                for feature_col, feature_val in zip(edge_feature_col, features_list):
                    row[feature_col] = feature_val

            writer.writerow(row)
