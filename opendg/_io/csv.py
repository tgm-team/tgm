import csv
from typing import Any, Dict, List, Optional

import torch

from opendg.events import EdgeEvent, Event, NodeEvent


def read_csv(
    file_path: str,
    src_col: str,
    dst_col: str,
    time_col: Optional[str] = None,
    edge_feature_col: Optional[List[str]] = None,
) -> List[Event]:
    # TODO: Node Feature not supported
    events: List[Event] = []
    with open(file_path, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            src_id = int(row[src_col])
            dst_id = int(row[dst_col])

            time = int(row[time_col]) if time_col is not None else i

            if edge_feature_col is None:
                features = None  # TODO:: Infer the feature columns
            else:
                features_list = [
                    float(row[feature_col]) for feature_col in edge_feature_col
                ]
                features = torch.tensor(features_list)

            event = EdgeEvent(time, (src_id, dst_id), features)
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
            src_id, dst_id = event.edge
            time = event.time
            features = event.features

            row: Dict[str, Any] = {
                src_col: src_id,
                dst_col: dst_id,
                time_col: time,
            }

            if features is not None:
                if edge_feature_col is None:
                    raise ValueError(
                        'No feature column provided but events had features'
                    )

                if len(features.shape) > 1:
                    raise ValueError('Multi-dimensional features not supported')

                if len(features) != len(edge_feature_col):
                    raise ValueError(
                        f'Got {len(features)}-dimensional feature tensor but only '
                        f'specified {len(edge_feature_col)} feature column names.'
                    )

                features_list = features.tolist()
                for feature_col, feature_val in zip(edge_feature_col, features_list):
                    row[feature_col] = feature_val

            writer.writerow(row)
