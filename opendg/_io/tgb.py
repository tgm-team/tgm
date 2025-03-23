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
    type(data['sources'])
    type(data['destinations'])
    type(data['timestamps'])
    type(data['edge_feat'])
    # with open(file_path, newline='') as f:
    #     reader = csv.DictReader(f)
    #     for row in reader:
    #         src = int(row[src_col])
    #         dst = int(row[dst_col])
    #         t = int(row[time_col])

    #         if edge_feature_col is None:
    #             features = None
    #         else:
    #             msg_list = [float(row[feature_col]) for feature_col in edge_feature_col]
    #             features = torch.tensor(msg_list)

    #         event = EdgeEvent(t=t, src=src, dst=dst, features=features)
    #         events.append(event)
    return events


def main():
    name = "tgbl-wiki"
    read_tgb(name=name)

if __name__ == "__main__":
    main()