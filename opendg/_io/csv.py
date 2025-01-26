import csv
from typing import Any, Dict, List, Optional

import torch

from opendg._events import EdgeEvent, Event, NodeEvent


def read_csv(
    file_path: str,
    src_node_id_col: str = 'src_id',
    dst_node_id_col: str = 'dst_id',
    time_col: str = 'timestamp',
    feature_cols: Optional[List[str]] = None,
) -> List[Event]:
    # TODO: Node Feature not supported
    events: List[Event] = []
    with open(file_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_id = int(row[src_node_id_col])
            dst_id = int(row[dst_node_id_col])
            time = int(row[time_col])

            if feature_cols is None:
                features = None
            else:
                # TODO: This doesn't work for multi dimensional array
                features = [float(row[feature_col]) for feature_col in feature_cols]
                features = torch.tensor(features)

            event = EdgeEvent(time, (src_id, dst_id), features)
            events.append(event)
    return events


def write_csv(
    events: List[Event],
    file_path: str,
    src_node_id_col: str = 'src_id',
    dst_node_id_col: str = 'dst_id',
    time_col: str = 'timestamp',
    feature_cols: Optional[List[str]] = None,
) -> None:
    with open(file_path, 'w', newline='') as f:
        fieldnames = [src_node_id_col, dst_node_id_col, time_col]
        if feature_cols is not None:
            fieldnames += feature_cols
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
                src_node_id_col: src_id,
                dst_node_id_col: dst_id,
                time_col: time,
            }

            if features is not None:
                if feature_cols is None:
                    raise ValueError(
                        'No feature column provided but events had features'
                    )

                if len(features.shape) > 1:
                    raise ValueError('Multi-dimensional features not supported')

                if len(features) != len(feature_cols):
                    raise ValueError(
                        f'Got {len(features)}-dimensional feature tensor but only '
                        f'specified {len(feature_cols)} feature column names.'
                    )

                features = features.tolist()
                for i, feature_col in enumerate(feature_cols):
                    row[feature_col] = features[i]

            writer.writerow(row)
