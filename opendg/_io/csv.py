import csv
from typing import Any, Dict, List

import torch

from opendg._events import EdgeEvent, Event, NodeEvent


def read_csv(file_path: str) -> List[Event]:
    # TODO: Node Feature not supported
    events: List[Event] = []
    with open(file_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # TODO: Make this configurable / implement auto-inference
            src_id = int(row['src_id'])
            dst_id = int(row['dst_id'])
            time = int(row['time'])
            features = row.get('features')
            if features == '':
                features = None
            else:
                # TODO: This doesn't work, need to parse a string
                features = torch.tensor(features)

            event = EdgeEvent(time, (src_id, dst_id), features)
            events.append(event)
    return events


def write_csv(events: List[Event], file_path: str) -> None:
    with open(file_path, 'w', newline='') as f:
        # TODO: Make this configurable / implement auto-inference
        fieldnames = ['src_id', 'dst_id', 'time', 'features']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for event in events:
            if isinstance(event, NodeEvent):
                raise NotImplementedError('Node feature IO not supported')

            assert isinstance(event, EdgeEvent)
            src_id, dst_id = event.edge
            time = event.time
            features = event.features

            row: Dict[str, Any] = {'src_id': src_id, 'dst_id': dst_id, 'time': time}
            if features is None:
                row['features'] = ''
            if features is not None:
                row['features'] = features.tolist()

            writer.writerow(row)
