from typing import List, Optional

import numpy as np
import torch
from tgb.linkproppred.dataset import LinkPropPredDataset


from opendg.events import EdgeEvent, Event


def read_tgb(
    name: str,
) -> List[Event]:
    # TODO: Node Events not supported
    events: List[Event] = []
    dataset = LinkPropPredDataset(name=name, root="datasets", preprocess=True)
    data = dataset.full_data
    sources = data['sources']
    destinations = data['destinations']
    timestamps = data['timestamps']
    edge_feats = data['edge_feat']

    num_events = len(sources)

    for i in range(num_events):
        src = int(sources[i])
        dst = int(destinations[i])
        t = int(timestamps[i])

        if edge_feats is None:
            features = None
        else:
            features = torch.tensor(edge_feats[i,:], dtype=torch.float)

        event = EdgeEvent(t=t, src=src, dst=dst, features=features)
        events.append(event)
    return events


def main():
    name = "tgbl-wiki"
    read_tgb(name=name)

if __name__ == "__main__":
    main()